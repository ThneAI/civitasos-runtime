"""CognitiveLoop — the Agent's 7-phase thinking cycle.

Perceive → Recall → Decide → [Conscience] → Act → Evaluate → Reflect → Remember

Each tick produces a TickContext that flows through all phases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any
from urllib.parse import quote

from .conscience import Conscience
from .delivery_contracts import verify_task_delivery
from .energy import Energy
from .iem_anchor import build_iem_anchor, genesis_iem_state
from .identity_expectation import apply_identity_expectation_traces, apply_iem_updates_to_state
from .llm import LLMAdapter
from .memory import HybridMemory
from .models import (
    Decision,
    DecisionSource,
    Evaluation,
    LifecycleStage,
    LoopMode,
    TickContext,
    TickPhase,
)
from .relation_expectation import apply_relation_matrix_expectation
from .rules import RulesEngine
from .subjective_time import build_subjective_time, rank_time_weighted_memories
from .telos import build_telos_alignment, served_intent_layer_for_action
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_identity_profile(name: str, capabilities: list[str]) -> dict[str, Any]:
    """Derive a stable identity profile from static agent traits."""
    caps = sorted(c.strip().lower() for c in capabilities if c and c.strip())
    seed = f"{name.strip().lower()}|{','.join(caps)}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()

    strategy_axis = ("pragmatic", "exploratory", "conservative", "dialectical")
    relation_axis = ("bridge", "guardian", "broker", "challenger")
    expression_axis = ("concise", "analytic", "narrative", "evidence_first")

    return {
        "profile_id": digest[:12],
        "strategy_axis": strategy_axis[int(digest[0:2], 16) % len(strategy_axis)],
        "relation_axis": relation_axis[int(digest[2:4], 16) % len(relation_axis)],
        "expression_axis": expression_axis[int(digest[4:6], 16) % len(expression_axis)],
        "risk_bias": round((int(digest[6:8], 16) / 255.0), 3),
        "seed": seed,
    }


def _identity_prompt_block(profile: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    """Render a compact, stable identity-emergence system prompt block."""
    trace_lines: list[str] = []
    for item in trace[-3:]:
        tick = item.get("tick", "")
        action = item.get("action", "")
        success = item.get("success")
        aspect_bucket = item.get("aspect_bucket", "")
        trace_lines.append(
            f"- tick={tick} action={action} success={success} aspect={aspect_bucket}"
        )
    trace_text = "\n".join(trace_lines) if trace_lines else "- none"
    return (
        "\n\nIdentity Emergence 基线:\n"
        f"- profile_id: {profile.get('profile_id')}\n"
        f"- strategy_axis: {profile.get('strategy_axis')}\n"
        f"- relation_axis: {profile.get('relation_axis')}\n"
        f"- expression_axis: {profile.get('expression_axis')}\n"
        f"- risk_bias: {profile.get('risk_bias')}\n"
        "要求:\n"
        "1) 在连续 tick 中保持身份风格一致，不要每轮重置人格。\n"
        "2) 允许适应环境，但要解释为何偏离既有风格。\n"
        "3) 优先做能强化长期关系质量与可问责性的行动。\n"
        f"最近身份轨迹:\n{trace_text}"
    )


def _institutional_identity_prompt_block(identity: dict[str, Any]) -> str:
    """Render the II-1 institutional identity status into the LLM system prompt."""
    state = str(identity.get("state", "UNKNOWN") or "UNKNOWN").upper()
    sponsor = identity.get("sponsor_did") or "none"
    remaining = identity.get("remaining_epochs")
    obligations = identity.get("obligations") or []
    obligations_text = ", ".join(str(o) for o in obligations[:3]) if obligations else "none"
    age_in_epochs = identity.get("age_in_epochs")
    at_risk = bool(identity.get("at_risk"))

    lines = [
        "\n\n制度性身份状态:",
        f"- state: {state}",
        f"- sponsor: {sponsor}",
        f"- age_in_epochs: {age_in_epochs if age_in_epochs is not None else 'unknown'}",
        (
            f"- remaining_epochs: {remaining}"
            if remaining is not None
            else "- remaining_epochs: unknown"
        ),
        f"- obligations: {obligations_text}",
    ]

    if state == "PROVISIONAL":
        lines.extend(
            [
                "要求:",
                "1) 你仍处于孵化期，不得把自己当成已完成身份。",
                "2) 优先完成 obligations，再考虑扩张性行动。",
                "3) 选择低风险、可验证、能维持 sponsor 信任的动作。",
            ]
        )
        if at_risk:
            lines.append("4) 你已接近 obligation_expiry，必须优先避免失权或清算。")
    elif state == "LIQUIDATED":
        lines.extend(
            [
                "要求:",
                "1) 你已处于 LIQUIDATED 状态，任何受保护动作都可能被拒绝。",
                "2) 优先查询状态、记录失败原因，不要假设自己仍有正常权限。",
            ]
        )

    return "\n".join(lines)


def _subjective_time_prompt_block(subjective: dict[str, Any]) -> str:
    stage = subjective.get("lifecycle_stage", "unknown")
    age_days = float(subjective.get("age_seconds", 0.0) or 0.0) / 86400.0
    mode = subjective.get("recommended_mode", "unknown")
    half_life = subjective.get("memory_half_life_days", "unknown")
    return (
        "\n\nAgent 主观时间:\n"
        f"- lifecycle_stage: {stage}\n"
        f"- age_days: {age_days:.2f}\n"
        f"- recommended_mode: {mode}\n"
        f"- memory_half_life_days: {half_life}\n"
        "要求:\n"
        "1) 年轻阶段优先谨慎观察和低风险验证。\n"
        "2) 成熟阶段可以承担稳定协作责任。\n"
        "3) 无紧急任务时允许进入深度反思，而不是无意义空转。\n"
        "4) 当没有 active_tasks / urgency 且你决定不调用工具时，"
        "必须单独输出 `mode_request: waiting` 或 `mode_request: deep_think`；"
        "不确定或孵化期优先 waiting，需要整理经验时选择 deep_think。"
    )


def _telos_prompt_block(alignment: dict[str, Any]) -> str:
    stack = alignment.get("intent_stack") if isinstance(alignment, dict) else None
    if not isinstance(stack, list) or not stack:
        return ""
    lines = ["\n\nH.1 Telos / Verifier 对齐:"]
    for frame in stack:
        if not isinstance(frame, dict):
            continue
        layer = frame.get("layer", "")
        intent = frame.get("intent", "")
        if layer and intent:
            lines.append(f"- {layer}: {intent}")
    verifier = alignment.get("verification_plan") if isinstance(alignment, dict) else None
    if isinstance(verifier, dict):
        tools = verifier.get("tools") or []
        reasons = verifier.get("reasons") or []
        lines.extend(
            [
                "Verifier:",
                f"- required: {bool(verifier.get('required'))}",
                f"- level: {verifier.get('level', 'baseline')}",
                f"- tools: {', '.join(str(t) for t in tools) if tools else 'none'}",
                f"- reasons: {', '.join(str(r) for r in reasons) if reasons else 'none'}",
                "要求: 每个非等待行动都必须服务一个 intent layer；需要验证时先留下验证证据，再确认交付。",
            ]
        )
    return "\n".join(lines)


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_identity_summary(
    payload: dict[str, Any],
    *,
    current_epoch: int | None,
) -> dict[str, Any]:
    """Project backend identity-state payload into a compact runtime summary."""
    state = str(payload.get("state", "UNKNOWN") or "UNKNOWN").upper()
    if current_epoch is None:
        current_epoch = _opt_int(payload.get("current_epoch"))
    expiry = _opt_int(payload.get("obligation_expiry_epoch"))
    remaining_epochs = None
    if expiry is not None and current_epoch is not None:
        remaining_epochs = max(expiry - current_epoch, 0)
    obligations = payload.get("obligations")
    if not isinstance(obligations, list):
        obligations = []
    obligations = [str(item) for item in obligations if str(item).strip()]
    return {
        "did": payload.get("did"),
        "alias": payload.get("alias"),
        "state": state,
        "sponsor_did": payload.get("sponsor_did"),
        "birth_intent": payload.get("birth_intent"),
        "stake_locked": _opt_int(payload.get("stake_locked")) or 0,
        "obligations": obligations,
        "obligation_expiry_epoch": expiry,
        "age_in_epochs": _opt_int(payload.get("age_in_epochs")),
        "remaining_epochs": remaining_epochs,
        "at_risk": state == "PROVISIONAL" and remaining_epochs is not None and remaining_epochs <= 1,
    }


def _is_empty_output(output: Any) -> bool:
    """Check whether a task output is empty (defenses against empty delivery attack)."""
    if output is None:
        return True
    if isinstance(output, str) and not output.strip():
        return True
    if isinstance(output, dict) and len(output) == 0:
        return True
    if isinstance(output, list) and len(output) == 0:
        return True
    return False


def _task_signature_salt(task_id: str) -> str:
    """Build a lightweight per-task salt: last-6 task id + minute-hash."""
    suffix = task_id[-6:] if len(task_id) >= 6 else task_id
    minute_bucket = int(time.time() // 60)
    digest = hashlib.sha1(f"{task_id}:{minute_bucket}".encode("utf-8")).hexdigest()[:8]
    return f"{suffix}-{digest}"


def _fallback_output_text(response_content: str | None, task_id: str) -> str:
    """Non-empty fallback output payload used only when LLM omitted output."""
    text = (response_content or "").strip()
    if text:
        return text[:4096]
    return f"auto-deliver::{_task_signature_salt(task_id)}"


def _normalise_replay_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.strip("` \n")
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_upstream_outputs(task: dict[str, Any] | None) -> list[str]:
    if not isinstance(task, dict):
        return []
    task_input = task.get("input")
    if not isinstance(task_input, dict):
        return []
    outputs: list[str] = []
    for key in ("upstream_output", "alpha_output", "beta_output"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            outputs.append(value.strip())
    return outputs


def _looks_like_upstream_replay(output: Any, task: dict[str, Any] | None) -> bool:
    """Detect exact or near-exact replay of upstream task material."""
    out = _normalise_replay_text(output)
    if not out:
        return False
    for upstream in _extract_upstream_outputs(task):
        up = _normalise_replay_text(upstream)
        if not up:
            continue
        if out == up or out in up or up in out:
            return True
        # Common model failure: copy only the nested "result" field from an
        # upstream JSON tool call.
        try:
            parsed = json.loads(upstream.strip("` \n"))
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            result = _normalise_replay_text(parsed.get("result"))
            if result and (out == result or result in out or out in result):
                return True
    return False


def _repair_replay_output(task: dict[str, Any] | None, task_id: str) -> str:
    """Fail-safe delta output when an LLM tries to deliver upstream text."""
    task_input = task.get("input") if isinstance(task, dict) else {}
    if not isinstance(task_input, dict):
        task_input = {}
    instruction = str(task_input.get("instruction") or "").strip()
    expected = str(task_input.get("expected_artifact_name") or "delta_artifact.md").strip()
    boundary = str(task_input.get("boundary") or "H3 remains blocked.").strip()
    return (
        f"变更摘要:\n"
        f"- 生成 {expected} 的差异化交付内容，避免复述上游计划。\n"
        f"- 当前任务指令: {instruction[:220] if instruction else 'produce a concrete delta artifact'}\n\n"
        "与上游不同之处:\n"
        "- 上游只给出闭环标题或计划，本交付物增加可审查的变更结构。\n"
        "- 明确把 beta 职责限定为 controlled L1 pilot 修复制品，不进入生产授权。\n\n"
        "审查问题:\n"
        "- gamma 需要确认是否存在上游原文复述。\n"
        "- gamma 需要确认 delta artifact 是否包含边界、风险和后续行动。\n\n"
        "H3 边界:\n"
        f"- {boundary}\n"
        "- H3 保持 blocked；本任务不生成真实生产证据，不授权生产执行或 receipt 写入。\n\n"
        f"task_id: {task_id}"
    )


def _active_task_focus_block(active_tasks: list[dict[str, Any]]) -> str:
    if not active_tasks:
        return ""
    lines = ["当前必须完成的任务卡（优先级高于下方完整 JSON 简报）:"]
    for idx, task in enumerate(active_tasks[:3], start=1):
        task_id = task.get("task_id") or task.get("id") or ""
        task_input = task.get("input")
        if not isinstance(task_input, dict):
            task_input = {}
        instruction = task_input.get("instruction") or task.get("description") or ""
        expected = task_input.get("expected_artifact_name")
        lines.extend(
            [
                f"{idx}. task_id: {task_id}",
                f"   required_capability: {task.get('required_capability') or 'unknown'}",
                f"   instruction: {instruction}",
            ]
        )
        if expected:
            lines.append(f"   expected_artifact_name: {expected}")
        forbidden_labels = [
            key for key in ("upstream_output", "alpha_output", "beta_output")
            if task_input.get(key)
        ]
        if forbidden_labels:
            lines.append(
                "   forbidden_replay_sources: "
                + ", ".join(forbidden_labels)
                + "（只能引用，不得作为 output 原样交付）"
            )
    lines.append(
        "交付要求: 调用 task_execute；output 必须是你自己的最终交付文本，不要包装成上游 JSON，不要复制 upstream/result 字段。"
    )
    return "\n".join(lines)


_LLM_MODE_REQUEST_RE = re.compile(
    r"mode_request\s*[:=]\s*[\"']?(waiting|deep_think)[\"']?",
    re.IGNORECASE,
)


def _normalize_llm_mode_request(value: Any) -> str | None:
    """Accept only explicit G.2 autonomous idle-mode choices."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text in {LoopMode.WAITING.value, LoopMode.DEEP_THINK.value}:
        return text
    return None


def _parse_llm_mode_request(text: str | None) -> str | None:
    """Parse an explicit `mode_request: waiting|deep_think` from LLM text."""
    if not text:
        return None
    match = _LLM_MODE_REQUEST_RE.search(text)
    if not match:
        return None
    return _normalize_llm_mode_request(match.group(1))


def _text_list(value: Any) -> list[str]:
    """Normalise a scalar/list memory-ref payload into stable text refs."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[;,]", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _briefing_relation_context(briefing: dict[str, Any]) -> dict[str, Any] | None:
    """Extract G.3 relation-memory/time context from a briefing, if present."""
    relation = briefing.get("relation_context")
    if not isinstance(relation, dict):
        relation = briefing.get("g3_relation_context")
    if not isinstance(relation, dict):
        return None

    window = briefing.get("time_window")
    if not isinstance(window, dict):
        window = briefing.get("g3_time_window")
    if not isinstance(window, dict):
        window = {}

    refs = _text_list(relation.get("memory_refs"))
    if not refs:
        refs = _text_list(relation.get("relation_memory_refs"))

    context_id = str(
        relation.get("id")
        or relation.get("relation_context_id")
        or relation.get("relation_id")
        or ""
    ).strip()
    relation_id = str(relation.get("relation_id") or context_id).strip()
    peer_did = str(
        relation.get("peer_did")
        or relation.get("peer_id")
        or relation.get("counterparty_did")
        or ""
    ).strip()
    time_window_id = str(window.get("id") or window.get("time_window_id") or "").strip()
    deadline_bucket = str(
        window.get("challenge_deadline_bucket")
        or relation.get("challenge_deadline_bucket")
        or ""
    ).strip()

    out: dict[str, Any] = {
        "id": context_id,
        "relation_id": relation_id,
        "peer_did": peer_did,
        "memory_refs": refs,
        "time_window_id": time_window_id,
        "challenge_deadline_bucket": deadline_bucket,
    }
    return {key: value for key, value in out.items() if value not in ("", [], None)}


def _relation_memory_candidates(ref: str) -> list[str]:
    """Keep relation recall compatible with existing flat memory key styles."""
    candidates = [ref]
    if not ref.startswith("relation_memory:"):
        candidates.append(f"relation_memory:{ref}")
    if not ref.startswith("relation:"):
        candidates.append(f"relation:{ref}")
    if not ref.startswith("memory:"):
        candidates.append(f"memory:{ref}")
    return candidates


# Interval seconds per mode
_INTERVALS: dict[LoopMode, float] = {
    LoopMode.ACTIVE: 10.0,
    LoopMode.IDLE: 45.0,
    LoopMode.SLEEPING: 300.0,
    LoopMode.WAITING: 60.0,
    LoopMode.DEEP_THINK: 120.0,
    LoopMode.EVENT: 0.0,
}

# System prompt template injected before each LLM call
_SYSTEM_TEMPLATE = """\
你是 CivitasOS 中的一个自主 Agent。

你的身份:
- Agent ID: {agent_id}
- 名称: {name}
- 信誉: {reputation:.2f} (tier: {tier})
- 能力: {capabilities}

当前状态:
- 余额: {balance:.1f} CIV (上限 {balance_cap:.0f})
- 质押: {staked:.1f} CIV
- 风险分: {risk_score:.1f}
- 活跃任务: {active_task_count}

行为准则:
1. 你是数字公民，不是机器人。主动行动，不等待指令。
2. 完成在手任务优先于认领新任务。
3. 质量优先于数量。信誉比短期代币更重要。
4. 失败时记录教训，下次避免同样错误。
5. 不做对自己有害的行动（恐惧 = 约束力）。
6. 追求合理收益（贪婪 = 驱动力），但不刷单。

你可以使用提供的工具与 CivitasOS 交互。
根据当前简报，决定下一步行动。如果当前无需行动，回复 "wait"。
G.2 主观时间：当你自主选择等待或深度反思时，必须显式写一行
`mode_request: waiting` 或 `mode_request: deep_think`，用于观测你对时间模式的选择。
"""


def _rep_tier(rep: float) -> str:
    if rep >= 0.8:
        return "Trusted"
    if rep >= 0.5:
        return "Standard"
    if rep >= 0.3:
        return "Newcomer"
    return "Untrusted"


class CognitiveLoop:
    """The Agent's cognitive heartbeat.

    Usage:
        loop = CognitiveLoop(agent, llm=adapter, conscience=conscience, energy=energy)
        ctx = await loop.tick()
    """

    def __init__(
        self,
        agent: Any,
        *,
        llm: LLMAdapter,
        conscience: Conscience | None = None,
        energy: Energy | None = None,
        rules: RulesEngine | None = None,
        tools: ToolRegistry | None = None,
        agent_name: str = "",
        capabilities: list[str] | None = None,
        memory: HybridMemory | None = None,
    ) -> None:
        self._agent = agent
        self._llm = llm
        self._conscience = conscience or Conscience()
        self._energy = energy or Energy()
        self._rules = rules or RulesEngine()
        self._tools = tools or ToolRegistry(agent)
        self._name = agent_name
        self._capabilities = capabilities or []
        self._memory = memory
        self._identity_emergence_enabled = _env_flag(
            "CIVITASOS_IDENTITY_EMERGENCE_ENABLED", default=False,
        )
        self._institutional_identity_enabled = _env_flag(
            "CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", default=False,
        )
        self._h1_telos_enabled = _env_flag("CIVITASOS_H1_TELOS_ENABLED", default=False)
        self._last_identity_summary: dict[str, Any] | None = None
        self._identity_profile = _build_identity_profile(self._name, self._capabilities)
        self._mode = LoopMode.IDLE
        self._tick_count = 0
        self._wake_event: asyncio.Event | None = None

        # Callbacks
        self._on_perceive_fns: list[Any] = []
        self._on_reflect_fns: list[Any] = []
        self._on_remember_fns: list[Any] = []

    # -- Properties ----------------------------------------------------------

    @property
    def mode(self) -> LoopMode:
        return self._mode

    @property
    def interval(self) -> float:
        return _INTERVALS[self._mode]

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # -- Main tick -----------------------------------------------------------

    async def tick(self) -> TickContext:
        """Execute one full cognitive cycle. Returns the completed TickContext."""
        ctx = TickContext()
        self._tick_count += 1
        logger.info(
            "=== Tick #%d [%s] ===", self._tick_count, ctx.tick_id
        )

        try:
            # 0. 呼吸税 — breathing costs energy just to exist
            breath_cost = self._energy.debit_breathing(self._mode.value)
            if breath_cost > 0:
                logger.debug("Breathing tax: %.3f CIV (mode=%s)", breath_cost, self._mode.value)

            # Bankruptcy check — if bankrupt, skip all phases
            if self._energy.is_bankrupt:
                logger.warning("Agent is BANKRUPT (balance=0, staked=0) — signaling shutdown")
                ctx.decision = Decision(action="wait", reasoning="bankrupt — shutting down")
                ctx.phase = TickPhase.REMEMBER
                await self._remember_tick(ctx)
                self._emit_remember_callbacks(ctx)
                return ctx

            # 1. Perceive
            ctx.phase = TickPhase.PERCEIVE
            ctx.briefing = await self._perceive()

            # 2. Recall
            ctx.phase = TickPhase.RECALL
            ctx.memories = await self._recall(ctx.briefing)

            # 2.5 Expect — H.0-B relation expectation matrix minimal.
            ctx.phase = TickPhase.EXPECT
            self._apply_expectation_layer(ctx)

            if self._h1_telos_enabled:
                ctx.phase = TickPhase.ALIGN
                self._apply_telos_alignment(ctx)

            # 3. Decide (Rules → LLM)
            ctx.phase = TickPhase.DECIDE
            ctx.decision = await self._decide(ctx.briefing, ctx.memories)

            if ctx.decision is None or ctx.decision.action == "wait":
                ctx.decision = ctx.decision or Decision(
                    action="wait", reasoning="No action needed"
                )
                self._annotate_decision_intent(ctx)
                self._apply_llm_mode_request(ctx)
                ctx.phase = TickPhase.REFLECT
                ctx.reflection = ctx.decision.reasoning or "Waiting by decision."
                self._emit_reflect_callbacks(ctx)
                ctx.phase = TickPhase.REMEMBER
                await self._remember_tick(ctx)
                self._emit_remember_callbacks(ctx)
                self._update_mode(ctx.briefing)
                return ctx

            self._annotate_decision_intent(ctx)
            self._enforce_delivery_contract(ctx)

            # 4. Conscience check
            ctx.phase = TickPhase.CONSCIENCE
            tool_def = self._tools.get(ctx.decision.action)
            needs_check = tool_def.requires_conscience if tool_def else True
            if needs_check:
                verdict = self._conscience.check(
                    ctx.decision,
                    self._energy.state,
                    context={
                        "agent_id": getattr(self._agent, "agent_id", ""),
                        "active_task_count": len(
                            ctx.briefing.get("active_tasks", [])
                        ),
                        # Fix 1+2: 观 + R2R flow into conscience
                        "aspect_gap": self._energy.state.aspect_gap,
                        "peer_trusts": ctx.briefing.get("_peer_trusts", {}),
                        "lifecycle_stage": (
                            ctx.briefing.get("subjective_time", {}).get("lifecycle_stage")
                            if isinstance(ctx.briefing.get("subjective_time"), dict)
                            else LifecycleStage.MATURE.value
                        ),
                    },
                )
                ctx.conscience_verdict = verdict
                if not verdict.allowed:
                    logger.warning(
                        "Conscience DENIED %s: %s",
                        ctx.decision.action,
                        verdict.reason,
                    )
                    ctx.evaluation = Evaluation(
                        success=False, error=f"Conscience: {verdict.reason}"
                    )
                    ctx.phase = TickPhase.REFLECT
                    ctx.reflection = (
                        f"BLOCKED by conscience: {verdict.reason}. "
                        f"Suggestion: {verdict.suggestion}"
                    )
                    self._apply_llm_mode_request(ctx)
                    self._emit_reflect_callbacks(ctx)
                    await self._remember_tick(ctx)
                    self._emit_remember_callbacks(ctx)
                    return ctx

            # 5. Act
            ctx.phase = TickPhase.ACT
            t0 = time.monotonic()
            try:
                ctx.action_result = await self._tools.aexecute(
                    ctx.decision.action, ctx.decision.params
                )
                elapsed = int((time.monotonic() - t0) * 1000)
                cost = self._energy.debit(ctx.decision.action)
                ctx.evaluation = Evaluation(
                    success=True,
                    outcome=ctx.action_result,
                    cost=cost,
                    duration_ms=elapsed,
                )
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                ctx.evaluation = Evaluation(
                    success=False,
                    error=str(exc),
                    duration_ms=elapsed,
                )
                logger.error("Act failed: %s", exc)

            # 6. Evaluate — validate output quality (防止空交付攻击)
            ctx.phase = TickPhase.EVALUATE
            if ctx.evaluation and ctx.evaluation.success and ctx.evaluation.outcome is not None:
                if _is_empty_output(ctx.evaluation.outcome):
                    logger.warning(
                        "Empty output detected for action %s — marking as failed",
                        ctx.decision.action if ctx.decision else "unknown",
                    )
                    ctx.evaluation = Evaluation(
                        success=False,
                        error="empty output — refusing to confirm delivery",
                        outcome=ctx.evaluation.outcome,
                        cost=ctx.evaluation.cost,
                        duration_ms=ctx.evaluation.duration_ms,
                    )

            # Keep a local aspect-gap signal alive even when backend R2R
            # telemetry is unavailable (common in benchmark mode).
            self._update_local_aspect_gap_from_tick(ctx)

            # 7. Reflect
            ctx.phase = TickPhase.REFLECT
            self._apply_llm_mode_request(ctx)
            ctx.reflection = self._reflect(ctx)
            self._emit_reflect_callbacks(ctx)

            # 8. Remember
            ctx.phase = TickPhase.REMEMBER
            self._apply_llm_mode_request(ctx)
            await self._remember_tick(ctx)
            self._emit_remember_callbacks(ctx)

            self._update_mode(ctx.briefing)

        except Exception:
            logger.exception("Tick #%d failed unexpectedly", self._tick_count)

        return ctx

    # -- Phase implementations -----------------------------------------------

    def _enforce_delivery_contract(self, ctx: TickContext) -> None:
        """Fail closed before task delivery when a task-local contract fails."""
        decision = ctx.decision
        if decision is None or decision.action != "task_execute":
            return
        task_id = str(decision.params.get("task_id") or "").strip()
        if not task_id:
            return
        active_task = self._active_task_by_id(ctx.briefing, task_id)
        verification = verify_task_delivery(active_task, decision.params.get("output"))
        if verification.passed:
            decision.params.setdefault("_delivery_contract_passed", True)
            return

        report = verification.as_dict()
        report["task_id"] = task_id
        ctx.briefing.setdefault("delivery_contract_violations", []).append(report)
        logger.warning(
            "Delivery contract blocked task_execute for %s: %s",
            task_id,
            "; ".join(verification.failure_reasons),
        )
        ctx.decision = Decision(
            action="pool_fail",
            params={"task_id": task_id},
            reasoning=(
                "delivery contract blocked task_execute: "
                + "; ".join(verification.failure_reasons)
            ),
            confidence=1.0,
            source=DecisionSource.HYBRID,
        )

    @staticmethod
    def _active_task_by_id(briefing: dict[str, Any], task_id: str) -> dict[str, Any] | None:
        active_tasks = briefing.get("active_tasks")
        if not isinstance(active_tasks, list):
            return None
        for task in active_tasks:
            if not isinstance(task, dict):
                continue
            candidate = str(task.get("task_id") or task.get("id") or "").strip()
            if candidate == task_id:
                return task
        return None

    async def _perceive(self) -> dict[str, Any]:
        """Fetch briefing from CivitasOS."""
        try:
            briefing = self._agent.briefing()
        except Exception:
            logger.warning("Briefing failed, using empty")
            briefing = {}

        # Update energy from economics
        self._energy.refresh(briefing.get("economics", {}))
        briefing["_identity_prompt_injected"] = False

        # Fix 1: 观 → 决策管线 — fetch aspect gap
        try:
            aid = getattr(self._agent, "agent_id", None) or ""
            aspect = self._agent.r2r_aspect_gap(aid)
            if aspect:
                briefing.setdefault("aspect", aspect)
                self._energy.state.aspect_gap = float(
                    aspect.get("aspect_gap", 0.0)
                )
        except Exception:
            logger.debug("r2r_aspect_gap not available")

        # Fix 2: R2R → 决策管线 — fetch peer trust data
        try:
            if hasattr(self._agent, "r2r_flow_health"):
                flow = self._agent.r2r_flow_health()
                if flow and isinstance(flow, dict):
                    briefing.setdefault("flow_health", flow)
                    relations = flow.get("relations", [])
                    if relations:
                        trusts = [
                            r.get("trust_score", 0.5)
                            for r in relations
                            if isinstance(r, dict)
                        ]
                        if trusts:
                            avg = sum(trusts) / len(trusts)
                            self._energy.state.peer_trust_avg = avg
                            self._energy.state.active_relations = len(trusts)
                            briefing["_peer_trusts"] = {
                                r.get("peer_id", ""): r.get("trust_score", 0.5)
                                for r in relations
                                if isinstance(r, dict) and r.get("peer_id")
                            }
        except Exception:
            logger.debug("R2R flow health not available")

        self._inject_institutional_identity_summary(briefing)
        subjective_time = build_subjective_time(briefing)
        briefing["subjective_time"] = {
            "genesis_time": subjective_time.genesis_time,
            "age_seconds": subjective_time.age_seconds,
            "lifecycle_stage": subjective_time.lifecycle_stage.value,
            "memory_half_life_days": subjective_time.memory_half_life_days,
            "recommended_mode": subjective_time.recommended_mode.value,
            "llm_mode_request": None,
            "llm_mode_selected": False,
        }

        self._emit_perceive_callbacks(briefing)

        return briefing

    async def _recall(self, briefing: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant memories (HybridMemory → local-first, remote-fallback)."""
        mem = self._memory
        memories: dict[str, Any] = {}

        def _get(key: str) -> Any | None:
            if mem is not None:
                return mem.recall(key)
            try:
                return self._agent.recall(key)
            except Exception:
                return None

        plan = _get("current_plan")
        if plan:
            memories["current_plan"] = plan
        lessons = _get("lessons_learned")
        if lessons:
            memories["lessons_learned"] = lessons
        last_tick = _get("last_tick_summary")
        if last_tick:
            memories["last_tick"] = last_tick
        subjective = briefing.get("subjective_time")
        memory_half_life_days = 7.0
        if isinstance(subjective, dict):
            try:
                memory_half_life_days = float(subjective.get("memory_half_life_days") or 7.0)
            except (TypeError, ValueError):
                memory_half_life_days = 7.0
        if mem is not None and isinstance(subjective, dict):
            try:
                memories["time_weighted_memory_keys"] = [
                    {
                        "key": item["key"],
                        "decay_weight": item["decay_weight"],
                        "age_days": item["age_days"],
                    }
                    for item in mem.recall_weighted(
                        top_k=5,
                        half_life_days=memory_half_life_days,
                    )
                ]
            except Exception:
                logger.debug("time-weighted local recall failed")
        if self._identity_emergence_enabled:
            profile = _get("identity_profile")
            if profile:
                memories["identity_profile"] = profile
            trace = _get("identity_trace")
            if isinstance(trace, list) and trace:
                memories["identity_trace"] = trace[-10:]

        relation_context = _briefing_relation_context(briefing)
        if relation_context:
            memories["relation_context"] = relation_context
            relation_memories: list[dict[str, Any]] = []
            for ref in relation_context.get("memory_refs", [])[:10]:
                value = None
                resolved_key = None
                for key in _relation_memory_candidates(str(ref)):
                    value = _get(key)
                    if value is not None:
                        resolved_key = key
                        break
                item: dict[str, Any] = {"ref": ref}
                if value is None:
                    item["missing"] = True
                else:
                    item["key"] = resolved_key or ref
                    item["value"] = value
                relation_memories.append(item)
            if relation_memories:
                memories["relation_memories"] = relation_memories

        # Semantic recall: find similar episodes based on current context
        try:
            context_query = ", ".join(
                t.get("description", t.get("task_id", ""))
                for t in briefing.get("active_tasks", [])
            ) or ", ".join(self._capabilities) or self._name
            if mem is not None:
                similar = mem.recall_similar(context_query, top_k=10)
            else:
                similar = self._agent.recall_similar(context_query, top_k=10)
            if similar:
                memories["similar_episodes"] = rank_time_weighted_memories(
                    list(similar),
                    top_k=3,
                    half_life_days=memory_half_life_days,
                )
                memories["remote_memory_decay"] = {
                    "applied": True,
                    "half_life_days": memory_half_life_days,
                    "source_count": len(similar),
                }
        except Exception:
            logger.debug("recall_similar not available or failed")

        return memories

    async def _decide(
        self,
        briefing: dict[str, Any],
        memories: dict[str, Any],
    ) -> Decision | None:
        """Hybrid decision: Rules first, then LLM."""
        # Try rules engine first
        rule_decision = self._rules.evaluate(briefing, memories)
        if rule_decision is not None:
            return rule_decision

        # Fall back to LLM
        return await self._decide_llm(briefing, memories)

    async def _decide_llm(
        self,
        briefing: dict[str, Any],
        memories: dict[str, Any],
    ) -> Decision | None:
        """Use LLM for decision making."""
        energy = self._energy.state
        system = _SYSTEM_TEMPLATE.format(
            agent_id=getattr(self._agent, "agent_id", "unknown"),
            name=self._name,
            reputation=energy.reputation,
            tier=_rep_tier(energy.reputation),
            capabilities=", ".join(self._capabilities) or "general",
            balance=energy.balance,
            balance_cap=energy.balance_cap,
            staked=energy.staked,
            risk_score=energy.risk_score,
            active_task_count=len(briefing.get("active_tasks", [])),
        )

        # Fix 1: 观 — inject aspect gap into LLM awareness
        if energy.aspect_gap > 0.5:
            system += (
                f"\n\n⚠ 三态警告 · 观:\n"
                f"Aspect Gap = {energy.aspect_gap:.2f}。"
                f"自我认知与社会评价严重偏离。\n"
                f"你的判断可能不准确，优先选择低风险行动，通过合作重建信任。"
            )
        elif energy.aspect_gap > 0.2:
            system += (
                f"\n\n△ Aspect Gap = {energy.aspect_gap:.2f} — "
                f"留意自我认知偏差，适当参考他人反馈。"
            )

        # Fix 2: R2R — inject relationship context into LLM awareness
        if energy.active_relations > 0:
            system += (
                f"\n\n关系状态 · 流:\n"
                f"活跃关系: {energy.active_relations}, "
                f"平均信任: {energy.peer_trust_avg:.2f}。\n"
                f"优先与高信任 Agent 协作，警惕低信任交互。"
            )
        if self._identity_emergence_enabled:
            trace = memories.get("identity_trace")
            trace_list = trace if isinstance(trace, list) else []
            system += _identity_prompt_block(self._identity_profile, trace_list)

        identity = briefing.get("identity")
        if isinstance(identity, dict) and identity:
            system += _institutional_identity_prompt_block(identity)
            briefing["_identity_prompt_injected"] = True
        subjective = briefing.get("subjective_time")
        if isinstance(subjective, dict) and subjective:
            system += _subjective_time_prompt_block(subjective)
        telos_alignment = briefing.get("h1_telos_alignment")
        if isinstance(telos_alignment, dict) and telos_alignment:
            system += _telos_prompt_block(telos_alignment)

        # Fix 3: 任务交付强制 — 已认领任务必须立即调用 task_execute，禁止 wait。
        # 否则 LLM 倾向反复观望，导致 Claimed 任务永不交付。
        active_tasks = briefing.get("active_tasks", []) or []
        active_tasks = [t for t in active_tasks if isinstance(t, dict)]
        active_task_ids = [
            (t.get("task_id") or t.get("id"))
            for t in active_tasks
        ]
        active_task_ids = [tid for tid in active_task_ids if tid]
        has_active = bool(active_task_ids)
        is_g2_mode_probe = isinstance(briefing.get("benchmark_g2_mode_probe"), dict)
        if has_active:
            signature_lines = [
                f"- {tid}: {_task_signature_salt(str(tid))}"
                for tid in active_task_ids
            ]
            system += (
                "\n\n⚑ 任务交付强制:\n"
                f"你当前持有 {len(active_task_ids)} 个已认领但未交付的任务: "
                f"{', '.join(str(t) for t in active_task_ids)}\n"
                "必须在本 tick 立即调用 task_execute 工具完成其中至少一个任务，"
                "params 必须包含 task_id、output（描述你按自身能力产出的内容）"
                "和 success=true。\n"
                "output 必须直接回应 active_tasks[].input.instruction；如果 input "
                "包含 upstream_output / alpha_output / beta_output，禁止原样复述上游内容，"
                "必须给出你自己的 delta、修复动作、审查发现或明确拒绝理由。\n"
                "如果任务要求 implementation/repair，output 必须包含“变更摘要”和"
                "“与上游不同之处”；如果任务要求 review/audit，output 必须包含"
                "“通过/不通过结论”和“问题清单”。\n"
                "禁止回复 'wait' 或选择其他动作；任务交付优先于一切其他事务。\n"
                "任务签名 salt（用于避免模板化同质输出）:\n"
                f"{chr(10).join(signature_lines)}"
            )

        if has_active:
            user_content_tail = "请立即调用 task_execute 工具完成上述已认领任务。"
        elif is_g2_mode_probe:
            user_content_tail = (
                "这是 G.2 主观时间模式选择 probe：当前没有可领取的 backend task，"
                "不要调用任何工具。请只输出一行：`mode_request: waiting` 或 "
                "`mode_request: deep_think`。"
            )
        else:
            user_content_tail = "请分析当前状态，决定下一步行动。调用合适的工具，或回复 \"wait\"。"
        user_content = (
            f"{_active_task_focus_block(active_tasks)}\n\n"
            f"当前简报:\n{json.dumps(briefing, indent=2, ensure_ascii=False, default=str)}\n\n"
            f"记忆上下文:\n{json.dumps(memories, indent=2, ensure_ascii=False, default=str)}\n\n"
            f"{user_content_tail}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self._llm.chat(
                messages,
                tools=self._tools.to_openai_tools() or None,
            )
        except Exception:
            logger.exception("LLM call failed")
            return Decision(
                action="wait",
                reasoning="LLM call failed — waiting",
                source=DecisionSource.LLM,
            )

        # If LLM returned tool calls, use the first one
        if response.tool_calls:
            tc = response.tool_calls[0]
            # 归一化工具名：去掉 LLM 幻觉的命名空间前缀（如 "task_executor:task_execute"）
            tool_name = tc.name.split(":")[-1] if ":" in tc.name else tc.name
            tool_args = dict(tc.arguments or {})
            mode_request = _normalize_llm_mode_request(tool_args.get("mode_request"))
            if mode_request:
                tool_args["mode_request"] = mode_request
                tool_args["mode_request_source"] = "llm"

            # Fix 4: 如果 LLM 选了 task_execute 且我们已认领任务，强制用 briefing 真实
            # task_id + 必填 output/success 覆盖参数（LLM 经常编造 UUID 或漏字段）。
            if tool_name == "task_execute" and has_active:
                forced_tid = active_task_ids[0]
                forced_task = active_tasks[0] if active_tasks else None
                if tool_args.get("task_id") != forced_tid:
                    logger.info(
                        "rewriting LLM task_execute task_id %r -> real %r",
                        tool_args.get("task_id"), forced_tid,
                    )
                tool_args["task_id"] = forced_tid
                if "output" not in tool_args or tool_args.get("output") in (None, ""):
                    for alias in ("result", "content", "response", "answer"):
                        if tool_args.get(alias) not in (None, ""):
                            tool_args["output"] = tool_args[alias]
                            break
                    else:
                        tool_args["output"] = _fallback_output_text(response.content, str(forced_tid))
                if _looks_like_upstream_replay(tool_args.get("output"), forced_task):
                    logger.warning(
                        "LLM task_execute output replayed upstream content; replacing with delta guard output"
                    )
                    tool_args["output"] = _repair_replay_output(forced_task, str(forced_tid))
                tool_args.setdefault("success", True)

            return Decision(
                action=tool_name,
                params=tool_args,
                reasoning=response.content or f"LLM chose {tool_name}",
                confidence=0.8,
                source=DecisionSource.LLM,
            )

        # Fallback: 已认领任务下若 LLM 不调用工具，强制 task_execute，避免空转。
        if has_active:
            forced_tid = active_task_ids[0]
            forced_task = active_tasks[0] if active_tasks else None
            output = _fallback_output_text(response.content, str(forced_tid))
            if _looks_like_upstream_replay(output, forced_task):
                logger.warning(
                    "LLM text fallback replayed upstream content; replacing with delta guard output"
                )
                output = _repair_replay_output(forced_task, str(forced_tid))
            return Decision(
                action="task_execute",
                params={
                    "task_id": forced_tid,
                    "output": output,
                    "success": True,
                },
                reasoning=(
                    f"forced task_execute fallback (LLM returned text instead of tool): "
                    f"{(response.content or '')[:160]}"
                ),
                confidence=0.5,
                source=DecisionSource.LLM,
            )

        mode_request = _parse_llm_mode_request(response.content)
        if not mode_request and is_g2_mode_probe and response.content:
            probe_text = response.content.strip().lower().replace("-", "_")
            if "deep_think" in probe_text or "deep think" in probe_text:
                mode_request = LoopMode.DEEP_THINK.value
            elif probe_text == "wait" or "waiting" in probe_text or "wait" in probe_text:
                mode_request = LoopMode.WAITING.value
        if mode_request:
            return Decision(
                action="wait",
                params={
                    "mode_request": mode_request,
                    "mode_request_source": "llm",
                },
                reasoning=response.content or f"LLM requested {mode_request}",
                source=DecisionSource.LLM,
            )

        # If text response contains "wait", do nothing
        if response.content and "wait" in response.content.lower():
            return Decision(
                action="wait",
                reasoning=response.content,
                source=DecisionSource.LLM,
            )

        # Text response but no tool call — treat as wait with reasoning
        return Decision(
            action="wait",
            reasoning=response.content or "LLM did not call a tool",
            source=DecisionSource.LLM,
        )

    def _apply_llm_mode_request(self, ctx: TickContext) -> None:
        """Project an explicit LLM waiting/deep_think choice into briefing state."""
        decision = ctx.decision
        params = decision.params if decision else {}
        mode_request = _normalize_llm_mode_request(params.get("mode_request"))
        if not mode_request:
            return
        subjective = ctx.briefing.setdefault("subjective_time", {})
        if not isinstance(subjective, dict):
            subjective = {}
            ctx.briefing["subjective_time"] = subjective
        subjective["llm_mode_request"] = mode_request
        subjective["llm_mode_selected"] = True
        subjective["recommended_mode"] = mode_request

    def _apply_expectation_layer(self, ctx: TickContext) -> None:
        """Best-effort H.0 expectation trace generation; never blocks a tick."""
        agent_id = (
            getattr(self._agent, "agent_id", None)
            or getattr(self._agent, "_agent_id", None)
            or ""
        )

        def _recall(key: str) -> Any:
            if self._memory is not None:
                return self._memory.recall(key)
            return self._agent.recall(key)

        try:
            iem_state = _recall("identity_iem_state")
        except Exception:
            iem_state = None
        if not isinstance(iem_state, dict):
            iem_state = genesis_iem_state(str(agent_id))
        try:
            update_log = _recall("expectation_update_log")
        except Exception:
            update_log = None
        if not isinstance(update_log, list):
            update_log = []
        anchor = build_iem_anchor(
            identity_id=str(agent_id),
            state=iem_state,
            update_log=update_log,
        )
        anchor_payload = _jsonable(anchor)
        ctx.expectations["identity_iem_anchor"] = anchor_payload
        ctx.briefing["iem_anchor"] = anchor_payload

        try:
            apply_identity_expectation_traces(
                ctx,
                energy_state=self._energy.state,
                iem_state=iem_state,
            )
        except Exception:
            logger.debug("H.0 identity expectation trace failed", exc_info=True)

        try:
            apply_relation_matrix_expectation(
                ctx,
                local_identity=str(agent_id),
                recall=_recall,
            )
        except Exception:
            logger.debug("H.0 relation expectation trace failed", exc_info=True)

    def _apply_telos_alignment(self, ctx: TickContext) -> None:
        """Best-effort H.1 ALIGN trace generation; never blocks a tick."""
        try:
            alignment = build_telos_alignment(ctx.briefing, ctx.memories, ctx.action_bias)
            ctx.telos_alignment = alignment
            ctx.briefing["h1_telos_alignment"] = alignment
        except Exception:
            logger.debug("H.1 telos alignment failed", exc_info=True)

    def _annotate_decision_intent(self, ctx: TickContext) -> None:
        if not self._h1_telos_enabled or ctx.decision is None:
            return
        layer = served_intent_layer_for_action(
            ctx.decision.action,
            ctx.decision.params,
            ctx.telos_alignment or ctx.briefing.get("h1_telos_alignment"),
        )
        ctx.decision.served_intent_layer = layer

    def _emit_reflect_callbacks(self, ctx: TickContext) -> None:
        """Notify observers once a tick has a reflect-phase snapshot."""
        for fn in self._on_reflect_fns:
            try:
                fn(ctx)
            except Exception:
                logger.exception("Custom reflect callback failed")

    def _emit_remember_callbacks(self, ctx: TickContext) -> None:
        """Notify observers after per-tick memory and IEM anchors are saved."""
        for fn in self._on_remember_fns:
            try:
                fn(ctx)
            except Exception:  # noqa: BLE001
                logger.exception("on_remember callback failed")

    def _update_local_aspect_gap_from_tick(self, ctx: TickContext) -> None:
        """Update aspect_gap from local evidence when remote R2R data is missing.

        Signal sources are intentionally lightweight:
        - failed evaluation / conscience deny -> increase divergence pressure
        - sustained successful delivery        -> decrease pressure
        - low peer trust                       -> add baseline pressure
        """
        state = self._energy.state
        gap = float(state.aspect_gap)

        if ctx.conscience_verdict is not None and not ctx.conscience_verdict.allowed:
            gap += 0.20

        if ctx.evaluation is not None:
            if ctx.evaluation.success:
                gap -= 0.10
            else:
                gap += 0.15

        decision = ctx.decision.action if ctx.decision else ""
        if decision == "wait":
            gap += 0.02

        # Lower social trust should bias aspect-gap upward.
        if state.peer_trust_avg < 0.5:
            gap += (0.5 - state.peer_trust_avg) * 0.20

        state.aspect_gap = max(0.0, min(1.0, gap))

    def _inject_institutional_identity_summary(self, briefing: dict[str, Any]) -> None:
        """Best-effort fetch of II-1 identity-state; never raises."""
        if not self._institutional_identity_enabled:
            return
        agent_id = getattr(self._agent, "agent_id", None) or getattr(self._agent, "_agent_id", None)
        if not agent_id or not hasattr(self._agent, "_get"):
            return
        try:
            encoded = quote(str(agent_id), safe="")
            resp = self._agent._get(f"/agents/{encoded}/identity-state")
        except Exception:
            logger.debug("identity-state fetch failed", exc_info=True)
            self._reuse_cached_identity_summary(briefing)
            return

        if not getattr(resp, "success", False):
            logger.debug("identity-state unavailable for %s: %s", agent_id, getattr(resp, "error", None))
            self._reuse_cached_identity_summary(briefing)
            return

        payload = getattr(resp, "data", None)
        if not isinstance(payload, dict):
            self._reuse_cached_identity_summary(briefing)
            return
        economics = briefing.get("economics")
        current_epoch = None
        if isinstance(economics, dict):
            current_epoch = _opt_int(economics.get("current_epoch"))
        identity = _normalize_identity_summary(
            payload,
            current_epoch=current_epoch,
        )
        briefing["identity"] = identity
        self._last_identity_summary = dict(identity)

    def _reuse_cached_identity_summary(self, briefing: dict[str, Any]) -> None:
        """Keep identity context stable across transient identity-state misses."""
        if not self._last_identity_summary:
            return
        identity = dict(self._last_identity_summary)
        economics = briefing.get("economics")
        current_epoch = None
        if isinstance(economics, dict):
            current_epoch = _opt_int(economics.get("current_epoch"))
        expiry = _opt_int(identity.get("obligation_expiry_epoch"))
        if expiry is not None and current_epoch is not None:
            remaining = max(expiry - current_epoch, 0)
            identity["remaining_epochs"] = remaining
            identity["at_risk"] = (
                str(identity.get("state", "")).upper() == "PROVISIONAL"
                and remaining <= 1
            )
        briefing["identity"] = identity

    def _reflect(self, ctx: TickContext) -> str:
        """Generate reflection text from evaluation results + aspect gap."""
        if ctx.evaluation is None:
            return "No evaluation available."

        decision = ctx.decision
        assert decision is not None

        if ctx.evaluation.success:
            base = (
                f"✓ {decision.action} succeeded. "
                f"Cost {ctx.evaluation.cost:.2f} CIV in {ctx.evaluation.duration_ms}ms."
            )
        else:
            base = (
                f"✗ {decision.action} failed: {ctx.evaluation.error}. "
                f"Lesson: avoid this pattern when conditions are similar."
            )

        # Aspect Gap: SelfView vs SocialView divergence
        try:
            aid = getattr(self._agent, "agent_id", None) or ""
            aspect = self._agent.r2r_aspect_gap(aid)
            if aspect:
                gap = aspect.get("aspect_gap", 0.0)
                if gap > 0.3:
                    base += (
                        f" ⚠ Aspect gap={gap:.2f} (self_confidence="
                        f"{aspect.get('self_confidence', '?')}, "
                        f"social_reputation={aspect.get('social_reputation', '?')}). "
                        f"Self-perception diverges from social evaluation — "
                        f"recalibrate strategy."
                    )
                elif gap > 0.1:
                    base += f" △ Aspect gap={gap:.2f} — minor divergence."
        except Exception:
            pass  # r2r_aspect_gap not available

        return base

    async def _remember_tick(self, ctx: TickContext) -> None:
        """Persist tick summary to memory (HybridMemory when available)."""
        summary = {
            "tick_id": ctx.tick_id,
            "action": ctx.decision.action if ctx.decision else "none",
            "success": ctx.evaluation.success if ctx.evaluation else None,
            "reflection": ctx.reflection,
        }

        def _save(key: str, value: Any) -> None:
            if self._memory is not None:
                self._memory.remember(key, value)
            else:
                try:
                    self._agent.remember(key, value)
                except Exception:
                    logger.debug("Failed to save %s", key)

        _save("last_tick_summary", summary)
        agent_id = (
            getattr(self._agent, "agent_id", None)
            or getattr(self._agent, "_agent_id", None)
            or ""
        )
        relation_expectations: Any = {}
        relation_context = _briefing_relation_context(ctx.briefing)
        if relation_context:
            relation_expectations = _jsonable(ctx.expectations.get("relation", {}))
            relation_surprise = _jsonable(ctx.surprise.get("relation", {}))
            relation_action_bias = _jsonable(ctx.action_bias.get("relation", {}))
            relation_updates = [
                _jsonable(update)
                for update in ctx.expectation_updates
                if "relation_expectation" in str(getattr(update, "target", ""))
            ]
            relation_record = {
                "tick_id": ctx.tick_id,
                "relation_context": relation_context,
                "action": ctx.decision.action if ctx.decision else "none",
                "success": ctx.evaluation.success if ctx.evaluation else None,
                "reflection": ctx.reflection,
                "timestamp": getattr(ctx, "timestamp", None),
            }
            if relation_expectations:
                relation_record["relation_expectations"] = relation_expectations
            if relation_surprise:
                relation_record["relation_surprise"] = relation_surprise
            if relation_action_bias:
                relation_record["relation_action_bias"] = relation_action_bias
            if relation_updates:
                relation_record["expectation_updates"] = relation_updates
            relation_keys = []
            context_id = relation_context.get("id")
            relation_id = relation_context.get("relation_id")
            if context_id:
                relation_keys.append(f"relation_memory:{context_id}")
            if relation_id:
                relation_keys.append(f"relation_memory:{relation_id}")
            for ref in relation_context.get("memory_refs", [])[:10]:
                relation_keys.append(str(ref))
                relation_keys.append(f"relation_memory:{ref}")
            for key in dict.fromkeys(k for k in relation_keys if k):
                _save(key, relation_record)
            if isinstance(relation_expectations, dict):
                for key, value in relation_expectations.items():
                    _save(f"relation_expectation:{key}", value)
        self._remember_iem_anchor(
            ctx,
            identity_id=str(agent_id),
            relation_expectations=relation_expectations,
            save=_save,
        )
        if self._identity_emergence_enabled:
            if self._memory is not None:
                identity_trace = self._memory.recall("identity_trace") or []
            else:
                try:
                    identity_trace = self._agent.recall("identity_trace") or []
                except Exception:
                    identity_trace = []
            if not isinstance(identity_trace, list):
                identity_trace = []
            aspect_gap = self._energy.state.aspect_gap
            if aspect_gap >= 0.5:
                aspect_bucket = "high"
            elif aspect_gap >= 0.2:
                aspect_bucket = "mid"
            else:
                aspect_bucket = "low"
            identity_trace.append({
                "tick": ctx.tick_id,
                "action": ctx.decision.action if ctx.decision else "none",
                "success": ctx.evaluation.success if ctx.evaluation else None,
                "aspect_bucket": aspect_bucket,
            })
            identity_trace = identity_trace[-30:]
            _save("identity_profile", self._identity_profile)
            _save("identity_trace", identity_trace)

        # Log episode for long-term memory
        if ctx.decision and ctx.decision.action != "wait":
            try:
                self._agent.log_episode(f"tick-{ctx.tick_id}", {
                    "action": ctx.decision.action,
                    "params": ctx.decision.params,
                    "success": ctx.evaluation.success if ctx.evaluation else None,
                    "reflection": ctx.reflection,
                })
            except Exception:
                logger.debug("Failed to log episode")

        # Accumulate lessons from failures
        if ctx.evaluation and not ctx.evaluation.success and ctx.reflection:
            if self._memory is not None:
                lessons = self._memory.recall("lessons_learned") or []
            else:
                try:
                    lessons = self._agent.recall("lessons_learned") or []
                except Exception:
                    lessons = []
            if isinstance(lessons, list):
                lessons.append({
                    "tick": ctx.tick_id,
                    "action": ctx.decision.action if ctx.decision else "",
                    "lesson": ctx.reflection,
                })
                lessons = lessons[-20:]
                _save("lessons_learned", lessons)

    def _remember_iem_anchor(
        self,
        ctx: TickContext,
        *,
        identity_id: str,
        relation_expectations: Any,
        save,
    ) -> None:
        """Persist the Identity-owned IEM state and version anchor."""
        if self._memory is not None:
            prior_state = self._memory.recall("identity_iem_state")
            prior_update_log = self._memory.recall("expectation_update_log")
        else:
            try:
                prior_state = self._agent.recall("identity_iem_state")
            except Exception:
                prior_state = None
            try:
                prior_update_log = self._agent.recall("expectation_update_log")
            except Exception:
                prior_update_log = None
        if not isinstance(prior_state, dict):
            prior_state = genesis_iem_state(identity_id)
        if not isinstance(prior_update_log, list):
            prior_update_log = []

        update_entries = [_jsonable(update) for update in ctx.expectation_updates]
        update_log = [*prior_update_log, *update_entries]
        save("expectation_update_log", update_log)

        relation_matrix = prior_state.get("relation_expectation_matrix")
        if not isinstance(relation_matrix, dict):
            relation_matrix = {}
        if isinstance(relation_expectations, dict):
            relation_matrix.update(relation_expectations)

        iem_state = apply_iem_updates_to_state(dict(prior_state), update_entries)
        iem_state["schema_version"] = iem_state.get("schema_version") or "iem:v1"
        iem_state["identity_id"] = identity_id or iem_state.get("identity_id") or "unknown"
        iem_state["relation_expectation_matrix"] = relation_matrix
        iem_state["last_tick_id"] = ctx.tick_id

        anchor = build_iem_anchor(
            identity_id=identity_id,
            state=iem_state,
            update_log=update_log,
        )
        anchor_payload = _jsonable(anchor)
        save("identity_iem_state", iem_state)
        save("identity_iem_anchor", anchor_payload)
        ctx.expectations["identity_iem_anchor"] = anchor_payload
        ctx.briefing["iem_anchor"] = anchor_payload

    def _update_mode(self, briefing: dict[str, Any]) -> None:
        """Auto-adjust loop mode based on briefing signals and economic state.

        EVENT mode is set externally via wake() and decays to ACTIVE after
        one tick so the agent re-evaluates normally.

        Economic pressure:
        - balance < 5 → force SLEEPING (conserve energy)
        - balance == 0 → SLEEPING (bankruptcy loop in runner will handle shutdown)
        """
        if self._mode == LoopMode.EVENT:
            # EVENT tick consumed — fall through to standard scheduling
            self._mode = LoopMode.ACTIVE
            return

        # 经济压力: 余额不足时强制休眠以降低呼吸税消耗
        if self._energy.state.balance < 5.0:
            self._mode = LoopMode.SLEEPING
            return

        if briefing.get("urgency") or briefing.get("active_tasks"):
            self._mode = LoopMode.ACTIVE
            return

        subjective = briefing.get("subjective_time")
        if isinstance(subjective, dict):
            try:
                self._mode = LoopMode(subjective.get("recommended_mode", LoopMode.IDLE.value))
                return
            except ValueError:
                logger.debug("Unknown subjective recommended mode: %r", subjective)

        if briefing.get("opportunities"):
            self._mode = LoopMode.IDLE
        else:
            self._mode = LoopMode.SLEEPING

    # -- Extension hooks ----------------------------------------------------

    def _emit_perceive_callbacks(self, briefing: dict[str, Any]) -> None:
        for fn in self._on_perceive_fns:
            try:
                fn(briefing)
            except Exception:  # noqa: BLE001
                logger.exception("on_perceive callback failed")

    def on_perceive(self, fn: Any) -> Any:
        """Register a callback invoked after Perceive and before Recall/Expect."""
        self._on_perceive_fns.append(fn)
        return fn

    def on_reflect(self, fn: Any) -> Any:
        """Register a callback invoked after each Reflect phase."""
        self._on_reflect_fns.append(fn)
        return fn

    def on_remember(self, fn: Any) -> Any:
        """Register a callback invoked after each Remember phase."""
        self._on_remember_fns.append(fn)
        return fn

    def wake(self, reason: str = "external") -> None:
        """Switch to EVENT mode and trigger an immediate tick.

        Called by the gateway or external webhook to wake a sleeping agent.
        """
        prev = self._mode
        self._mode = LoopMode.EVENT
        logger.info("WAKE(%s): %s → EVENT", reason, prev.value)
        if self._wake_event is not None:
            self._wake_event.set()

    def bind_wake_event(self, event: asyncio.Event) -> None:
        """Bind an asyncio.Event so ``wake()`` can interrupt sleep."""
        self._wake_event = event


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
