"""Tests for the three VMV alignment fixes in civitas-runtime.

Fix 1: 观→决策 — Aspect gap flows into conscience + LLM context
Fix 2: R2R→决策 — Peer trust flows into conscience + LLM context
Fix 3: 治理门控 — Governance-protected thresholds require approval
"""

import pytest
from civitas_runtime import (
    Conscience,
    Decision,
    DecisionSource,
    EnergyState,
    PendingThresholdChange,
)


# ---------------------------------------------------------------------------
# Fix 1: 观缺失 — Aspect gap → Conscience gate
# ---------------------------------------------------------------------------

class TestFix1AspectGate:
    """Aspect gap should block risky actions when self-perception diverges."""

    def test_high_aspect_gap_blocks_pool_claim(self):
        c = Conscience()
        d = Decision(action="pool_claim", params={"task_id": "t1"})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.8})
        assert not verdict.allowed
        assert "Aspect gap" in verdict.reason

    def test_high_aspect_gap_blocks_task_execute(self):
        c = Conscience()
        d = Decision(action="task_execute", params={"task_id": "t1"})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.75})
        assert not verdict.allowed

    def test_high_aspect_gap_blocks_create_proposal(self):
        c = Conscience()
        d = Decision(action="create_proposal", params={})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.9})
        assert not verdict.allowed

    def test_moderate_aspect_gap_allows_actions(self):
        c = Conscience()
        d = Decision(action="pool_claim", params={"task_id": "t1"})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.5})
        assert verdict.allowed

    def test_low_aspect_gap_allows_all(self):
        c = Conscience()
        d = Decision(action="task_execute", params={"task_id": "t1"})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.1})
        assert verdict.allowed

    def test_high_gap_allows_safe_actions(self):
        """High aspect gap should NOT block read-only/safe actions."""
        c = Conscience()
        d = Decision(action="recall", params={})
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"aspect_gap": 0.9})
        assert verdict.allowed


# ---------------------------------------------------------------------------
# Fix 2: R2R孤岛 — Peer trust → Conscience gate
# ---------------------------------------------------------------------------

class TestFix2PeerTrustGate:
    """Low peer trust should block collaboration with untrusted agents."""

    def test_low_trust_blocks_propose_relation(self):
        c = Conscience()
        d = Decision(
            action="r2r_propose_relation",
            params={"target_agent": "bad-agent"},
        )
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={
            "peer_trusts": {"bad-agent": 0.1},
        })
        assert not verdict.allowed
        assert "trust" in verdict.reason.lower()

    def test_low_trust_blocks_pool_claim_for_peer(self):
        c = Conscience()
        d = Decision(
            action="pool_claim",
            params={"peer_id": "shady-agent"},
        )
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={
            "peer_trusts": {"shady-agent": 0.05},
        })
        assert not verdict.allowed

    def test_moderate_trust_allows_action(self):
        c = Conscience()
        d = Decision(
            action="r2r_propose_relation",
            params={"target_agent": "ok-agent"},
        )
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={
            "peer_trusts": {"ok-agent": 0.5},
        })
        assert verdict.allowed

    def test_no_peer_trust_data_allows_action(self):
        """If no peer trust data available, should not block."""
        c = Conscience()
        d = Decision(
            action="r2r_propose_relation",
            params={"target_agent": "unknown-agent"},
        )
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={"peer_trusts": {}})
        assert verdict.allowed

    def test_unknown_target_allows_action(self):
        """If target not in peer_trusts dict, should not block."""
        c = Conscience()
        d = Decision(
            action="pool_claim",
            params={"peer_id": "new-agent"},
        )
        e = EnergyState(balance=500.0)
        verdict = c.check(d, e, context={
            "peer_trusts": {"other-agent": 0.1},
        })
        assert verdict.allowed


# ---------------------------------------------------------------------------
# Fix 3: 自演化脱离立宪 — Governance gate on thresholds
# ---------------------------------------------------------------------------

class TestFix3GovernanceGate:
    """Governance-protected thresholds must go through propose/approve cycle."""

    def test_protected_threshold_queued(self):
        c = Conscience()
        change = c.propose_threshold("max_risk_score", 80.0, reason="Relax risk")
        assert change is not None
        assert isinstance(change, PendingThresholdChange)
        assert change.param_name == "max_risk_score"
        assert change.current_value == 50.0
        assert change.proposed_value == 80.0
        # Threshold should NOT have changed yet
        assert c._max_risk == 50.0

    def test_unprotected_threshold_applied_directly(self):
        c = Conscience()
        old = c._min_balance
        change = c.propose_threshold("min_balance_reserve", 20.0)
        assert change is None  # Applied directly, not queued
        assert c._min_balance == 20.0

    def test_drain_pending(self):
        c = Conscience()
        c.propose_threshold("max_risk_score", 80.0)
        c.propose_threshold("min_reward_cost_ratio", 2.0)
        pending = c.drain_pending()
        assert len(pending) == 2
        assert pending[0].param_name == "max_risk_score"
        assert pending[1].param_name == "min_reward_cost_ratio"
        # Drain should clear the list
        assert len(c.drain_pending()) == 0

    def test_apply_approved(self):
        c = Conscience()
        c.propose_threshold("max_risk_score", 80.0)
        # Governance approves
        ok = c.apply_approved("max_risk_score", 80.0)
        assert ok
        assert c._max_risk == 80.0

    def test_apply_approved_unknown_param(self):
        c = Conscience()
        ok = c.apply_approved("nonexistent_param", 99.0)
        assert not ok

    def test_governance_gate_end_to_end(self):
        """Full cycle: propose → verify blocked → drain → approve → verify active."""
        c = Conscience()
        assert c._max_risk == 50.0

        # 1. Propose: threshold stays at 50.0
        change = c.propose_threshold("max_risk_score", 100.0, reason="Testing")
        assert change is not None
        assert c._max_risk == 50.0

        # 2. Before approval: risk_score=60 should still be blocked
        d = Decision(action="pool_claim", params={})
        e = EnergyState(balance=500.0, risk_score=60.0)
        verdict = c.check(d, e, context={})
        assert not verdict.allowed

        # 3. Drain pending, governance approves
        pending = c.drain_pending()
        assert len(pending) == 1
        c.apply_approved("max_risk_score", 100.0)
        assert c._max_risk == 100.0

        # 4. After approval: risk_score=60 should now be allowed
        verdict = c.check(d, e, context={})
        assert verdict.allowed


# ---------------------------------------------------------------------------
# EnergyState: new fields have correct defaults
# ---------------------------------------------------------------------------

class TestEnergyStateFields:
    def test_new_fields_default(self):
        e = EnergyState()
        assert e.aspect_gap == 0.0
        assert e.peer_trust_avg == 0.5
        assert e.active_relations == 0

    def test_new_fields_settable(self):
        e = EnergyState(aspect_gap=0.4, peer_trust_avg=0.8, active_relations=5)
        assert e.aspect_gap == 0.4
        assert e.peer_trust_avg == 0.8
        assert e.active_relations == 5


# ---------------------------------------------------------------------------
# PendingThresholdChange model
# ---------------------------------------------------------------------------

class TestPendingThresholdChange:
    def test_creation(self):
        p = PendingThresholdChange(
            param_name="max_risk_score",
            current_value=50.0,
            proposed_value=80.0,
            reason="test",
        )
        assert p.param_name == "max_risk_score"
        assert p.created_at  # auto-generated ISO timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
