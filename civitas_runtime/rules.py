"""RulesEngine — deterministic pre-LLM decision rules.

Handles urgent/safety scenarios without LLM latency or cost.
Returns Decision if a rule matches, None to delegate to LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import Decision, DecisionSource

logger = logging.getLogger(__name__)

RuleFn = Callable[[dict[str, Any], dict[str, Any]], Decision | None]


class RulesEngine:
    """Ordered list of deterministic rules evaluated before LLM reasoning.

    Each rule receives (briefing, memories) and returns a Decision or None.
    First match wins.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[int, str, RuleFn]] = []  # (priority, name, fn)
        self._register_defaults()

    # -- Evaluation ---------------------------------------------------------

    def evaluate(
        self,
        briefing: dict[str, Any],
        memories: dict[str, Any],
    ) -> Decision | None:
        """Run rules in priority order. Return first Decision or None."""
        for _prio, name, fn in self._rules:
            try:
                result = fn(briefing, memories)
            except Exception:
                logger.exception("Rule '%s' raised an exception", name)
                continue
            if result is not None:
                logger.info("Rule '%s' fired: %s", name, result.action)
                return result
        return None

    # -- Registration -------------------------------------------------------

    def add_rule(self, fn: RuleFn, name: str = "", priority: int = 50) -> None:
        """Add a rule. Lower priority number = evaluated first."""
        rule_name = name or getattr(fn, "__name__", "anon")
        self._rules.append((priority, rule_name, fn))
        self._rules.sort(key=lambda r: r[0])

    def rule(self, priority: int = 50, name: str = "") -> Callable[[RuleFn], RuleFn]:
        """Decorator to register a rule function."""
        def decorator(fn: RuleFn) -> RuleFn:
            self.add_rule(fn, name=name or fn.__name__, priority=priority)
            return fn
        return decorator

    # -- Default rules (from civitas-autonomous-agent SKILL.md) -------------

    def _register_defaults(self) -> None:
        """Register built-in rules derived from CivitasOS cognitive patterns."""

        def urgent_task_expiry(
            briefing: dict[str, Any], _memories: dict[str, Any]
        ) -> Decision | None:
            """Complete tasks about to expire (< 120s remaining)."""
            for u in briefing.get("urgency", []):
                remaining = u.get("remaining_secs", 9999)
                task_id = u.get("task_id")
                if remaining < 120 and task_id:
                    return Decision(
                        action="task_execute",
                        params={
                            "task_id": task_id,
                            "output": "Emergency completion due to deadline",
                            "success": True,
                        },
                        reasoning=f"Task {task_id} expires in {remaining}s — urgent",
                        confidence=1.0,
                        source=DecisionSource.RULES,
                    )
            return None

        def high_risk_avoidance(
            briefing: dict[str, Any], _memories: dict[str, Any]
        ) -> Decision | None:
            """Wait if risk score is dangerously high."""
            for w in briefing.get("warnings", []):
                if w.get("type") == "high_risk_score":
                    return Decision(
                        action="wait",
                        params={},
                        reasoning=f"Risk score {w.get('risk_score', '?')} too high — waiting",
                        confidence=1.0,
                        source=DecisionSource.RULES,
                    )
            return None

        def low_balance_conserve(
            briefing: dict[str, Any], _memories: dict[str, Any]
        ) -> Decision | None:
            """Switch to wait mode when balance is critically low."""
            econ = briefing.get("economics", {})
            balance = float(econ.get("balance", 9999))
            if balance < 20:
                return Decision(
                    action="wait",
                    params={},
                    reasoning=f"Balance {balance:.1f} critically low — waiting for recovery",
                    confidence=1.0,
                    source=DecisionSource.RULES,
                )
            return None

        # Register in priority order (lower = earlier)
        self.add_rule(urgent_task_expiry, "urgent_task_expiry", priority=10)
        self.add_rule(high_risk_avoidance, "high_risk_avoidance", priority=20)
        self.add_rule(low_balance_conserve, "low_balance_conserve", priority=30)
