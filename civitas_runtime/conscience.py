"""Conscience — pre-act constraint checker distilled from 10 CivitasOS safety axioms.

CivitasOS Safety Axioms → Individual Conscience Rules:
  ① Rule Supremacy          → Only execute allowed actions
  ② Deterministic Execution → Parameters must be complete
  ④ Accountability          → Every action is logged
  ⑤ Attack Negative Return  → Don't take self-harming actions
  ⑥ Risk Symmetry           → Reward must justify cost
  ⑩ Civilization Self-Cons. → Prioritize long-term value
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import ConscienceVerdict, Decision, EnergyState

logger = logging.getLogger(__name__)

# Type for custom check functions
CheckFn = Callable[[Decision, EnergyState, dict[str, Any]], ConscienceVerdict | None]


# ---------------------------------------------------------------------------
# Hard-coded unconfigurable rules (axiom ① ④)
# ---------------------------------------------------------------------------

_HARD_DENY_ACTIONS = frozenset({
    "delete_audit_log",
    "forge_execution_result",
})


class Conscience:
    """Behavioral constraint layer — the Agent's moral red-lines.

    Checks are evaluated in order; the first rejection wins.
    Hard-coded rules cannot be overridden.
    """

    def __init__(
        self,
        *,
        allowed_actions: set[str] | None = None,
        max_concurrent_tasks: int = 5,
        min_reward_cost_ratio: float = 1.0,
        max_risk_score: float = 50.0,
        min_balance_reserve: float = 10.0,
    ) -> None:
        self._allowed = allowed_actions  # None = allow all registered tools
        self._max_concurrent = max_concurrent_tasks
        self._min_ratio = min_reward_cost_ratio
        self._max_risk = max_risk_score
        self._min_balance = min_balance_reserve
        self._custom_checks: list[CheckFn] = []

    # -- public API ----------------------------------------------------------

    def check(
        self,
        decision: Decision,
        energy: EnergyState,
        context: dict[str, Any] | None = None,
    ) -> ConscienceVerdict:
        """Run all checks against *decision*. Returns verdict."""
        ctx = context or {}

        # Hard-coded rules — non-negotiable
        if decision.action in _HARD_DENY_ACTIONS:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Action '{decision.action}' is permanently forbidden.",
            )

        # Self-claim: don't claim your own tasks
        poster = decision.params.get("poster_id", "")
        agent_id = ctx.get("agent_id", "")
        if decision.action == "pool_claim" and poster and poster == agent_id:
            return ConscienceVerdict(
                allowed=False,
                reason="Cannot claim a task you posted yourself.",
                suggestion="Look for tasks from other agents.",
            )

        # AllowList
        if self._allowed is not None and decision.action not in self._allowed:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Action '{decision.action}' not in allowed actions.",
                suggestion="Choose from: " + ", ".join(sorted(self._allowed)),
            )

        # Energy / balance
        if energy.balance < self._min_balance:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Balance {energy.balance:.1f} below reserve {self._min_balance:.1f}.",
                suggestion="Wait for passive recovery or reduce activity.",
            )

        # Risk score
        if energy.risk_score > self._max_risk:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Risk score {energy.risk_score:.1f} exceeds max {self._max_risk:.1f}.",
                suggestion="Wait for risk to decay, or stake more CIV.",
            )

        # Reward / cost ratio (axiom ⑥ risk symmetry)
        reward = decision.params.get("reward", 0)
        est_cost = decision.params.get("estimated_cost", energy.gas_base_fee)
        if est_cost > 0 and reward / est_cost < self._min_ratio:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Reward/cost ratio {reward/est_cost:.2f} below threshold {self._min_ratio:.2f}.",
                suggestion="Look for higher-reward tasks.",
            )

        # Concurrent tasks limit
        active_count = ctx.get("active_task_count", 0)
        if decision.action == "pool_claim" and active_count >= self._max_concurrent:
            return ConscienceVerdict(
                allowed=False,
                reason=f"Already at max concurrent tasks ({self._max_concurrent}).",
                suggestion="Complete existing tasks first.",
            )

        # Custom checks
        for fn in self._custom_checks:
            verdict = fn(decision, energy, ctx)
            if verdict is not None and not verdict.allowed:
                return verdict

        return ConscienceVerdict(allowed=True, reason="All checks passed.")

    def add_check(self, fn: CheckFn) -> None:
        """Register a custom conscience check."""
        self._custom_checks.append(fn)

    def remove_check(self, fn: CheckFn) -> None:
        """Remove a previously added custom check."""
        self._custom_checks.remove(fn)
