"""Tests for the 4 existential pressure mechanisms in civitasos-runtime.

1. 呼吸税 (Breathing Tax)    — every tick costs energy
2. 冷启动惩罚 (Cold Start)    — new agents start with 0 stake, low balance
3. 位置竞争 (Slot Competition) — tested at Core/Backend level
4. 关系资本 (Relationship Capital) — R2R trust bound to identity

These tests cover the *Runtime* layer distillation:
  - Energy.debit_breathing() rate per LoopMode
  - Energy.is_bankrupt detection
  - EnergyState default values (initial_balance=500, staked=0)
  - CognitiveLoop._update_mode() forced SLEEPING on low balance
  - CognitiveLoop bankruptcy exit path
  - AgentRunner bankruptcy → stop() lifecycle
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from civitasos_runtime import EnergyState
from civitasos_runtime.energy import BREATHING_COST, Energy
from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.models import Decision, LoopMode, TickPhase


# ===========================================================================
# 1. 呼吸税 — Breathing Tax
# ===========================================================================


class TestBreathingTax:
    """Every tick costs breathing tax depending on LoopMode."""

    def test_active_breathing_cost(self):
        e = Energy()
        cost = e.debit_breathing("ACTIVE")
        assert cost == pytest.approx(0.1)
        assert e.state.balance == pytest.approx(500.0 - 0.1)

    def test_idle_breathing_cost(self):
        e = Energy()
        cost = e.debit_breathing("IDLE")
        assert cost == pytest.approx(0.05)
        assert e.state.balance == pytest.approx(500.0 - 0.05)

    def test_sleeping_breathing_cost(self):
        e = Energy()
        cost = e.debit_breathing("SLEEPING")
        assert cost == pytest.approx(0.01)
        assert e.state.balance == pytest.approx(500.0 - 0.01)

    def test_event_breathing_cost(self):
        e = Energy()
        cost = e.debit_breathing("EVENT")
        assert cost == pytest.approx(0.1)

    def test_unknown_mode_fallback(self):
        """Unknown mode should default to 0.01 (SLEEPING-level)."""
        e = Energy()
        cost = e.debit_breathing("UNKNOWN_MODE")
        assert cost == pytest.approx(0.01)

    def test_breathing_cost_constants(self):
        """Verify the BREATHING_COST table matches design spec."""
        assert BREATHING_COST == {
            "ACTIVE": 0.1,
            "IDLE": 0.05,
            "SLEEPING": 0.01,
            "EVENT": 0.1,
        }

    def test_cumulative_breathing_drains_balance(self):
        """500 balance ÷ 0.1/tick = 5000 ticks to drain at ACTIVE rate."""
        e = Energy()
        for _ in range(100):
            e.debit_breathing("ACTIVE")
        assert e.state.balance == pytest.approx(500.0 - 100 * 0.1)

    def test_breathing_cannot_go_negative(self):
        """Balance floors at 0, never negative."""
        e = Energy()
        e._state.balance = 0.03
        cost = e.debit_breathing("ACTIVE")  # 0.1 > 0.03
        assert cost == pytest.approx(0.1)  # cost reported is nominal
        assert e.state.balance == pytest.approx(0.0)  # floored at 0


# ===========================================================================
# 2. 破产检测 — Bankruptcy Detection
# ===========================================================================


class TestBankruptcy:
    """Agent is dead when balance=0 AND staked=0."""

    def test_not_bankrupt_with_balance(self):
        e = Energy()
        assert not e.is_bankrupt

    def test_not_bankrupt_with_stake_only(self):
        e = Energy()
        e._state.balance = 0.0
        e._state.staked = 50.0
        assert not e.is_bankrupt

    def test_bankrupt_when_both_zero(self):
        e = Energy()
        e._state.balance = 0.0
        e._state.staked = 0.0
        assert e.is_bankrupt

    def test_bankrupt_boundary(self):
        """Negative values also count as bankrupt."""
        e = Energy()
        e._state.balance = -1.0  # shouldn't happen, but defensive
        e._state.staked = 0.0
        assert e.is_bankrupt

    def test_tiny_balance_not_bankrupt(self):
        """Even 0.001 CIV balance means not bankrupt yet."""
        e = Energy()
        e._state.balance = 0.001
        e._state.staked = 0.0
        assert not e.is_bankrupt

    def test_breathing_to_bankruptcy(self):
        """Drain via breathing until bankrupt (balance=0, staked=0)."""
        e = Energy()
        e._state.balance = 0.05
        e._state.staked = 0.0
        assert not e.is_bankrupt
        e.debit_breathing("IDLE")  # -0.05 → balance=0
        assert e.is_bankrupt


# ===========================================================================
# 3. 冷启动惩罚 — Cold Start Defaults
# ===========================================================================


class TestColdStartDefaults:
    """New agents start with initial_balance=500, staked=0 (cold start)."""

    def test_default_balance(self):
        s = EnergyState()
        assert s.balance == pytest.approx(500.0)

    def test_default_staked(self):
        """New agent starts with 0 stake — must earn and stake themselves."""
        s = EnergyState()
        assert s.staked == pytest.approx(0.0)

    def test_default_reputation(self):
        s = EnergyState()
        assert s.reputation == pytest.approx(0.5)

    def test_default_balance_cap(self):
        s = EnergyState()
        assert s.balance_cap == pytest.approx(10000.0)

    def test_energy_inherits_cold_start_defaults(self):
        """Energy() uses EnergyState defaults — cold start values."""
        e = Energy()
        assert e.state.balance == pytest.approx(500.0)
        assert e.state.staked == pytest.approx(0.0)

    def test_greed_signal_at_start(self):
        """With 500 balance and 10000 cap, 500 < 2000 (20%) → greed active."""
        e = Energy()
        assert e.greed_signal is True

    def test_no_overflow_at_start(self):
        """500 < 9000 (90% of cap) → no overflow warning."""
        e = Energy()
        assert e.overflow_warning is False


# ===========================================================================
# 4. 濒临死亡 — SLEEPING 强制切换
# ===========================================================================


class TestForcedSleeping:
    """When balance < 5.0, loop mode is forced to SLEEPING."""

    def _make_loop(self, balance: float = 500.0) -> CognitiveLoop:
        """Create a CognitiveLoop with a mocked agent and preset balance."""
        agent = MagicMock()
        agent.agent_id = "test-agent"
        energy = Energy()
        energy._state.balance = balance

        loop = CognitiveLoop(
            agent,
            llm=MagicMock(),
            energy=energy,
            agent_name="TestAgent",
        )
        return loop

    def test_normal_balance_active_with_tasks(self):
        """High balance + active tasks → ACTIVE mode."""
        loop = self._make_loop(balance=500.0)
        loop._update_mode({"active_tasks": [{"task_id": "t1"}]})
        assert loop.mode == LoopMode.ACTIVE

    def test_normal_balance_idle_with_opportunities(self):
        """High balance + opportunities → IDLE mode."""
        loop = self._make_loop(balance=500.0)
        loop._update_mode({"opportunities": True})
        assert loop.mode == LoopMode.IDLE

    def test_low_balance_forces_sleeping(self):
        """balance < 5.0 → SLEEPING regardless of opportunities."""
        loop = self._make_loop(balance=4.9)
        loop._update_mode({"active_tasks": [{"task_id": "t1"}], "urgency": True})
        assert loop.mode == LoopMode.SLEEPING

    def test_zero_balance_forces_sleeping(self):
        """balance = 0 → SLEEPING."""
        loop = self._make_loop(balance=0.0)
        loop._update_mode({"urgency": True})
        assert loop.mode == LoopMode.SLEEPING

    def test_boundary_exactly_5_not_sleeping(self):
        """balance == 5.0 → not forced SLEEPING (< 5, not <=5)."""
        loop = self._make_loop(balance=5.0)
        loop._update_mode({"active_tasks": [{"task_id": "t1"}]})
        assert loop.mode == LoopMode.ACTIVE

    def test_boundary_4_99_forces_sleeping(self):
        """balance = 4.99 → forced SLEEPING."""
        loop = self._make_loop(balance=4.99)
        loop._update_mode({"active_tasks": [{"task_id": "t1"}]})
        assert loop.mode == LoopMode.SLEEPING


# ===========================================================================
# 5. 认知循环中的破产路径
# ===========================================================================


class TestCognitiveLoopBankruptcy:
    """CognitiveLoop.tick() should short-circuit when agent is bankrupt."""

    def test_bankrupt_tick_skips_all_phases(self):
        """Bankrupt agent → tick produces wait/bankruptcy, no action."""
        agent = MagicMock()
        agent.agent_id = "dead-agent"
        energy = Energy()
        energy._state.balance = 0.0
        energy._state.staked = 0.0

        loop = CognitiveLoop(
            agent,
            llm=MagicMock(),
            energy=energy,
            agent_name="DeadAgent",
        )

        ctx = asyncio.run(loop.tick())
        assert ctx.decision is not None
        assert ctx.decision.action == "wait"
        assert "bankrupt" in ctx.decision.reasoning.lower()
        # Should NOT have gone through Perceive (no briefing call)
        agent.briefing.assert_not_called()

    def test_non_bankrupt_tick_proceeds(self):
        """Non-bankrupt agent should proceed to Perceive."""
        agent = MagicMock()
        agent.agent_id = "alive-agent"
        agent.briefing.return_value = {}
        agent.r2r_aspect_gap.return_value = {"gap": 0.1}
        agent.r2r_peer_trust.return_value = {}

        energy = Energy()  # default balance=500, staked=0

        loop = CognitiveLoop(
            agent,
            llm=MagicMock(),
            energy=energy,
            agent_name="AliveAgent",
        )

        ctx = asyncio.run(loop.tick())
        # Should have called briefing (Perceive phase)
        agent.briefing.assert_called_once()


# ===========================================================================
# 6. Energy.refresh() 覆盖默认值
# ===========================================================================


class TestEnergyRefresh:
    """Energy.refresh() should update state from server economics data."""

    def test_refresh_updates_balance(self):
        e = Energy()
        e.refresh({"balance": 1234.5, "staked_amount": 200.0})
        assert e.state.balance == pytest.approx(1234.5)
        assert e.state.staked == pytest.approx(200.0)

    def test_refresh_partial(self):
        """Partial update — only specified fields change."""
        e = Energy()
        e.refresh({"balance": 999.0})
        assert e.state.balance == pytest.approx(999.0)
        assert e.state.staked == pytest.approx(0.0)  # unchanged default

    def test_refresh_empty_noop(self):
        """Empty dict → no changes."""
        e = Energy()
        e.refresh({})
        assert e.state.balance == pytest.approx(500.0)

    def test_refresh_none_noop(self):
        """None → no changes."""
        e = Energy()
        e.refresh(None)
        assert e.state.balance == pytest.approx(500.0)


# ===========================================================================
# 7. Fear/Greed 恐惧与贪婪信号
# ===========================================================================


class TestFearGreedSignals:
    """Economic pressure signals that drive agent behavior."""

    def test_fear_low_default(self):
        e = Energy()
        assert e.fear_level == "low"

    def test_fear_high(self):
        e = Energy()
        e._state.risk_score = 60.0
        assert e.fear_level == "high"

    def test_fear_moderate(self):
        e = Energy()
        e._state.risk_score = 30.0
        assert e.fear_level == "moderate"

    def test_greed_when_poor(self):
        """Balance < 20% of cap → greed drives task-seeking."""
        e = Energy()
        e._state.balance = 100.0
        assert e.greed_signal is True

    def test_no_greed_when_rich(self):
        e = Energy()
        e._state.balance = 5000.0
        assert e.greed_signal is False

    def test_overflow_warning_near_cap(self):
        """Balance > 90% of cap → overflow warning (众生之果)."""
        e = Energy()
        e._state.balance = 9500.0
        assert e.overflow_warning is True

    def test_no_overflow_below_cap(self):
        e = Energy()
        e._state.balance = 5000.0
        assert e.overflow_warning is False


# ===========================================================================
# 8. Gas 消费与 can_afford
# ===========================================================================


class TestGasCosts:
    """Verify gas cost estimation and affordability checks."""

    def test_free_actions(self):
        e = Energy()
        assert e.estimate_cost("briefing") == pytest.approx(0.0)
        assert e.estimate_cost("pool_discover") == pytest.approx(0.0)
        assert e.estimate_cost("recall") == pytest.approx(0.0)

    def test_expensive_action(self):
        e = Energy()
        assert e.estimate_cost("create_proposal") == pytest.approx(5.0)

    def test_can_afford_with_balance(self):
        e = Energy()
        assert e.can_afford("create_proposal")

    def test_cannot_afford_when_broke(self):
        e = Energy()
        e._state.balance = 1.0
        assert not e.can_afford("create_proposal")  # costs 5.0

    def test_risk_multiplier(self):
        """Higher risk_score → higher effective gas cost."""
        e = Energy()
        e._state.risk_score = 100.0  # → 1 + 100*0.01 = 2x
        cost = e.estimate_cost("task_execute")  # base 2.0
        assert cost == pytest.approx(4.0)  # 2.0 * 2.0

    def test_debit_reduces_balance(self):
        e = Energy()
        cost = e.debit("task_execute")
        assert cost == pytest.approx(2.0)
        assert e.state.balance == pytest.approx(498.0)
