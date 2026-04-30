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
from typing import Any
from urllib.parse import quote

from .conscience import Conscience
from .energy import Energy
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
from .rules import RulesEngine
from .subjective_time import build_subjective_time
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
        "3) 无紧急任务时允许进入深度反思，而不是无意义空转。"
    )


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
        return text[:512]
    return f"auto-deliver::{_task_signature_salt(task_id)}"


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
        self._last_identity_summary: dict[str, Any] | None = None
        self._identity_profile = _build_identity_profile(self._name, self._capabilities)
        self._mode = LoopMode.IDLE
        self._tick_count = 0
        self._wake_event: asyncio.Event | None = None

        # Callbacks
        self._on_reflect_fns: list[Any] = []

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
                return ctx

            # 1. Perceive
            ctx.phase = TickPhase.PERCEIVE
            ctx.briefing = await self._perceive()

            # 2. Recall
            ctx.phase = TickPhase.RECALL
            ctx.memories = await self._recall(ctx.briefing)

            # 3. Decide (Rules → LLM)
            ctx.phase = TickPhase.DECIDE
            ctx.decision = await self._decide(ctx.briefing, ctx.memories)

            if ctx.decision is None or ctx.decision.action == "wait":
                ctx.decision = ctx.decision or Decision(
                    action="wait", reasoning="No action needed"
                )
                self._apply_llm_mode_request(ctx)
                ctx.phase = TickPhase.REFLECT
                ctx.reflection = ctx.decision.reasoning or "Waiting by decision."
                self._emit_reflect_callbacks(ctx)
                ctx.phase = TickPhase.REMEMBER
                await self._remember_tick(ctx)
                self._update_mode(ctx.briefing)
                return ctx

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

            self._update_mode(ctx.briefing)

        except Exception:
            logger.exception("Tick #%d failed unexpectedly", self._tick_count)

        return ctx

    # -- Phase implementations -----------------------------------------------

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
                        half_life_days=float(subjective.get("memory_half_life_days") or 7.0),
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

        # Semantic recall: find similar episodes based on current context
        try:
            context_query = ", ".join(
                t.get("description", t.get("task_id", ""))
                for t in briefing.get("active_tasks", [])
            ) or ", ".join(self._capabilities) or self._name
            if mem is not None:
                similar = mem.recall_similar(context_query, top_k=3)
            else:
                similar = self._agent.recall_similar(context_query, top_k=3)
            if similar:
                memories["similar_episodes"] = similar
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

        # Fix 3: 任务交付强制 — 已认领任务必须立即调用 task_execute，禁止 wait。
        # 否则 LLM 倾向反复观望，导致 Claimed 任务永不交付。
        active_tasks = briefing.get("active_tasks", []) or []
        active_task_ids = [
            (t.get("task_id") or t.get("id"))
            for t in active_tasks
            if isinstance(t, dict)
        ]
        active_task_ids = [tid for tid in active_task_ids if tid]
        has_active = bool(active_task_ids)
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
                "禁止回复 'wait' 或选择其他动作；任务交付优先于一切其他事务。\n"
                "任务签名 salt（用于避免模板化同质输出）:\n"
                f"{chr(10).join(signature_lines)}"
            )

        user_content_tail = (
            "请立即调用 task_execute 工具完成上述已认领任务。"
            if has_active
            else "请分析当前状态，决定下一步行动。调用合适的工具，或回复 \"wait\"。"
        )
        user_content = (
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
                if tool_args.get("task_id") != forced_tid:
                    logger.info(
                        "rewriting LLM task_execute task_id %r -> real %r",
                        tool_args.get("task_id"), forced_tid,
                    )
                tool_args["task_id"] = forced_tid
                if "output" not in tool_args or tool_args.get("output") in (None, ""):
                    tool_args["output"] = _fallback_output_text(response.content, str(forced_tid))
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
            return Decision(
                action="task_execute",
                params={
                    "task_id": forced_tid,
                    "output": _fallback_output_text(response.content, str(forced_tid)),
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

    def _emit_reflect_callbacks(self, ctx: TickContext) -> None:
        """Notify observers once a tick has a reflect-phase snapshot."""
        for fn in self._on_reflect_fns:
            try:
                fn(ctx)
            except Exception:
                logger.exception("Custom reflect callback failed")

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

    def on_reflect(self, fn: Any) -> Any:
        """Register a callback invoked after each Reflect phase."""
        self._on_reflect_fns.append(fn)
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
