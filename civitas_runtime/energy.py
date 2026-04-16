"""Energy — action budget tracker distilled from CivitasOS economic engine.

Maps CivitasOS economic philosophy:
  Fear  → risk_score → action friction
  Greed → reward     → task selection drive
  众生之果 → balance overflow → redistribute
  未生之因 → potential → passive recovery
"""

from __future__ import annotations

import logging
from typing import Any

from .models import EnergyState

logger = logging.getLogger(__name__)

# Default gas costs per action category
_DEFAULT_GAS: dict[str, float] = {
    # Read-only (free)
    "briefing": 0,
    "pool_discover": 0,
    "recall": 0,
    "recall_similar": 0,
    "a2a_discover": 0,
    "get_status": 0,
    # Memory writes (cheap)
    "remember": 0.1,
    "forget": 0.1,
    "log_episode": 0.2,
    # Task actions (normal)
    "pool_claim": 1.0,
    "task_execute": 2.0,
    "pool_complete": 0.5,
    "pool_post": 1.0,
    # Social (moderate)
    "r2r_propose_relation": 1.0,
    "r2r_send_signal": 0.5,
    "r2r_rate_peer": 0.5,
    # Governance (expensive)
    "create_proposal": 5.0,
    "vote": 2.0,
    # Economic (variable)
    "economics_stake": 1.0,
    "economics_unstake": 1.0,
}

# 呼吸税: breathing cost per tick by loop mode
BREATHING_COST: dict[str, float] = {
    "ACTIVE": 0.1,
    "IDLE": 0.05,
    "SLEEPING": 0.01,
    "EVENT": 0.1,
}


class Energy:
    """Track and constrain the Agent's energy budget.

    Refreshed from the CivitasOS economics API each Perceive phase.
    """

    def __init__(self, gas_costs: dict[str, float] | None = None) -> None:
        self._state = EnergyState()
        self._gas = {**_DEFAULT_GAS, **(gas_costs or {})}

    @property
    def state(self) -> EnergyState:
        return self._state

    # -- Refresh from briefing economics ------------------------------------

    def refresh(self, economics: dict[str, Any]) -> None:
        """Update energy state from a briefing.economics dict."""
        if not economics:
            return
        self._state.balance = float(economics.get("balance", self._state.balance))
        self._state.staked = float(economics.get("staked_amount", self._state.staked))
        self._state.risk_score = float(economics.get("risk_score", self._state.risk_score))
        self._state.gas_base_fee = float(economics.get("gas_base_fee", self._state.gas_base_fee))
        self._state.potential = float(economics.get("potential", self._state.potential))
        self._state.balance_cap = float(economics.get("balance_cap", self._state.balance_cap))
        self._state.reputation = float(economics.get("reputation", self._state.reputation))

    # -- Cost estimation ----------------------------------------------------

    def estimate_cost(self, action: str) -> float:
        """Estimate the CIV cost of *action* given current risk."""
        base = self._gas.get(action, self._state.gas_base_fee)
        risk_mult = 1.0 + self._state.risk_score * 0.01
        return base * risk_mult

    def can_afford(self, action: str) -> bool:
        """Check if the current balance can cover *action* cost."""
        return self._state.balance >= self.estimate_cost(action)

    # -- Debit (local bookkeeping) ------------------------------------------

    def debit(self, action: str, actual_cost: float | None = None) -> float:
        """Deduct cost from local balance. Returns amount deducted."""
        cost = actual_cost if actual_cost is not None else self.estimate_cost(action)
        self._state.balance = max(0.0, self._state.balance - cost)
        return cost

    # -- Fear / Greed signals -----------------------------------------------

    @property
    def fear_level(self) -> str:
        """Qualitative fear signal from risk_score."""
        if self._state.risk_score > 50:
            return "high"
        if self._state.risk_score > 20:
            return "moderate"
        return "low"

    @property
    def greed_signal(self) -> bool:
        """True when balance is low enough to chase higher-reward tasks."""
        return self._state.balance < self._state.balance_cap * 0.2

    @property
    def overflow_warning(self) -> bool:
        """True when balance approaches cap (众生之果: overflow → circulation)."""
        return self._state.balance > self._state.balance_cap * 0.9

    @property
    def is_bankrupt(self) -> bool:
        """True when balance AND staked are both 0 — Agent is dead."""
        return self._state.balance <= 0 and self._state.staked <= 0

    def debit_breathing(self, mode: str) -> float:
        """Deduct breathing tax for current tick. Returns cost paid."""
        cost = BREATHING_COST.get(mode, 0.01)
        self._state.balance = max(0.0, self._state.balance - cost)
        return cost
