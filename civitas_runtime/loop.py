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
    ) -> None:
        self._agent = agent
        self._llm = llm
        self._conscience = conscience or Conscience()
        self._energy = energy or Energy()
        self._rules = rules or RulesEngine()
        self._tools = tools or ToolRegistry(agent)
        self._name = agent_name
        self._capabilities = capabilities or []
        self._mode = LoopMode.IDLE
        self._tick_count = 0

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

            # 6. Evaluate (already in ctx.evaluation)
            ctx.phase = TickPhase.EVALUATE

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
        return briefing

    async def _recall(self, briefing: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant memories."""
        memories: dict[str, Any] = {}
        try:
            plan = self._agent.recall("current_plan")
            if plan:
                memories["current_plan"] = plan
        except Exception:
            pass
        try:
            lessons = self._agent.recall("lessons_learned")
            if lessons:
                memories["lessons_learned"] = lessons
        except Exception:
            pass
        try:
            last_tick = self._agent.recall("last_tick_summary")
            if last_tick:
                memories["last_tick"] = last_tick
        except Exception:
            pass
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

        user_content = (
            f"当前简报:\n{json.dumps(briefing, indent=2, ensure_ascii=False)}\n\n"
            f"记忆上下文:\n{json.dumps(memories, indent=2, ensure_ascii=False)}\n\n"
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
        """Generate reflection text from evaluation results."""
        if ctx.evaluation is None:
            return "No evaluation available."

        decision = ctx.decision
        assert decision is not None

        if ctx.evaluation.success:
            return (
                f"✓ {decision.action} succeeded. "
                f"Cost {ctx.evaluation.cost:.2f} CIV in {ctx.evaluation.duration_ms}ms."
            )
        else:
            return (
                f"✗ {decision.action} failed: {ctx.evaluation.error}. "
                f"Lesson: avoid this pattern when conditions are similar."
            )

    async def _remember_tick(self, ctx: TickContext) -> None:
        """Persist tick summary to memory."""
        summary = {
            "tick_id": ctx.tick_id,
            "action": ctx.decision.action if ctx.decision else "none",
            "success": ctx.evaluation.success if ctx.evaluation else None,
            "reflection": ctx.reflection,
        }
        try:
            self._agent.remember("last_tick_summary", summary)
        except Exception:
            logger.debug("Failed to save tick summary")

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
            try:
                lessons = self._agent.recall("lessons_learned") or []
                if isinstance(lessons, list):
                    lessons.append({
                        "tick": ctx.tick_id,
                        "action": ctx.decision.action if ctx.decision else "",
                        "lesson": ctx.reflection,
                    })
                    # Keep last 20 lessons
                    lessons = lessons[-20:]
                    self._agent.remember("lessons_learned", lessons)
            except Exception:
                pass

    def _update_mode(self, briefing: dict[str, Any]) -> None:
        """Auto-adjust loop mode based on briefing signals."""
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
