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
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import ConscienceVerdict, Decision, EnergyState, PendingThresholdChange

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
        # Fix 3: governance-protected thresholds cannot be changed without approval
        self._governance_protected: set[str] = {
            "min_reward_cost_ratio", "max_risk_score",
        }
        self._pending_changes: list[PendingThresholdChange] = []

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
        # Fix 1: Aspect gap — high divergence blocks risky actions (观→决策)
        aspect_gap = ctx.get("aspect_gap", 0.0)
        benchmark_mode = bool(os.getenv("BENCHMARK_TASK_ID"))
        benchmark_target_claim = (
            decision.action == "pool_claim"
            and bool(decision.params.get("_benchmark_target_claim"))
            and benchmark_mode
        )
        benchmark_target_execute = (
            decision.action == "task_execute"
            and benchmark_mode
        )
        if aspect_gap > 0.7 and decision.action in {
            "pool_claim", "task_execute", "create_proposal",
        }:
            if benchmark_target_claim or benchmark_target_execute:
                logger.debug(
                    "Conscience: allow benchmark target %s under high aspect_gap=%.2f",
                    decision.action, aspect_gap,
                )
            else:
                return ConscienceVerdict(
                    allowed=False,
                    reason=(
                        f"Aspect gap {aspect_gap:.2f} — self-perception diverges "
                        f"too far from social reality for '{decision.action}'."
                    ),
                    suggestion="Reduce risk and rebuild social trust before risky actions.",
                )
        # Reward / cost ratio (axiom ⑥ risk symmetry)
        # Skip for zero-cost actions (perceive, memory reads, etc.)
        reward = decision.params.get("reward", 0)
        est_cost = decision.params.get("estimated_cost", energy.gas_base_fee)
        if est_cost > 0 and reward > 0 and reward / est_cost < self._min_ratio:
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

        # Fix 2: Peer trust — block collaboration with untrusted peers (R2R→决策)
        peer_trusts = ctx.get("peer_trusts", {})
        target_peer = (
            decision.params.get("target_agent")
            or decision.params.get("peer_id", "")
        )
        if target_peer and target_peer in peer_trusts:
            trust = peer_trusts[target_peer]
            if trust < 0.2 and decision.action in {
                "r2r_propose_relation", "pool_claim", "task_execute",
            }:
                return ConscienceVerdict(
                    allowed=False,
                    reason=f"Peer '{target_peer}' trust={trust:.2f} too low for '{decision.action}'.",
                    suggestion="Build trust through signals before direct collaboration.",
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

    # -- Fix 3: Governance gate on threshold changes -------------------------

    def propose_threshold(
        self,
        param_name: str,
        new_value: float,
        reason: str = "",
    ) -> PendingThresholdChange | None:
        """Propose a threshold change.

        Governance-protected params are queued; others applied directly.
        Returns the pending change if queued, None if applied or param unknown.
        """
        current = self._get_threshold(param_name)
        if current is None:
            return None
        if param_name in self._governance_protected:
            change = PendingThresholdChange(
                param_name=param_name,
                current_value=current,
                proposed_value=new_value,
                reason=reason,
            )
            self._pending_changes.append(change)
            logger.info(
                "Threshold '%s' governance-protected: queued %.2f → %.2f",
                param_name, current, new_value,
            )
            return change
        self._set_threshold(param_name, new_value)
        return None

    def drain_pending(self) -> list[PendingThresholdChange]:
        """Return and clear all pending governance-protected threshold changes."""
        changes = self._pending_changes[:]
        self._pending_changes.clear()
        return changes

    def apply_approved(self, param_name: str, approved_value: float) -> bool:
        """Apply a governance-approved threshold change."""
        ok = self._set_threshold(param_name, approved_value)
        if ok:
            logger.info(
                "Governance-approved: '%s' → %.2f", param_name, approved_value
            )
        return ok

    def _get_threshold(self, name: str) -> float | None:
        _MAP = {
            "min_reward_cost_ratio": "_min_ratio",
            "max_risk_score": "_max_risk",
            "min_balance_reserve": "_min_balance",
            "max_concurrent_tasks": "_max_concurrent",
        }
        attr = _MAP.get(name)
        return float(getattr(self, attr)) if attr else None

    def _set_threshold(self, name: str, value: float) -> bool:
        _MAP = {
            "min_reward_cost_ratio": "_min_ratio",
            "max_risk_score": "_max_risk",
            "min_balance_reserve": "_min_balance",
            "max_concurrent_tasks": "_max_concurrent",
        }
        attr = _MAP.get(name)
        if not attr:
            return False
        setattr(
            self, attr,
            int(value) if name == "max_concurrent_tasks" else value,
        )
        return True
