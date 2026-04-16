"""CognitiveLoop — the Agent's 7-phase thinking cycle.

Perceive → Recall → Decide → [Conscience] → Act → Evaluate → Reflect → Remember

Each tick produces a TickContext that flows through all phases.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .conscience import Conscience
from .energy import Energy
from .llm import LLMAdapter
from .memory import HybridMemory
from .models import (
    Decision,
    DecisionSource,
    Evaluation,
    LoopMode,
    TickContext,
    TickPhase,
)
from .rules import RulesEngine
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


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

# Interval seconds per mode
_INTERVALS: dict[LoopMode, float] = {
    LoopMode.ACTIVE: 10.0,
    LoopMode.IDLE: 45.0,
    LoopMode.SLEEPING: 300.0,
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

            # 7. Reflect
            ctx.phase = TickPhase.REFLECT
            ctx.reflection = self._reflect(ctx)
            for fn in self._on_reflect_fns:
                try:
                    fn(ctx)
                except Exception:
                    logger.exception("Custom reflect callback failed")

            # 8. Remember
            ctx.phase = TickPhase.REMEMBER
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

        user_content = (
            f"当前简报:\n{json.dumps(briefing, indent=2, ensure_ascii=False, default=str)}\n\n"
            f"记忆上下文:\n{json.dumps(memories, indent=2, ensure_ascii=False, default=str)}\n\n"
            "请分析当前状态，决定下一步行动。调用合适的工具，或回复 \"wait\"。"
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
            return Decision(
                action=tc.name,
                params=tc.arguments,
                reasoning=response.content or f"LLM chose {tc.name}",
                confidence=0.8,
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
        elif briefing.get("opportunities"):
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
