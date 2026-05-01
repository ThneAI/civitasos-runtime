from __future__ import annotations

from civitasos_runtime.identity_expectation import (
    apply_identity_expectation_traces,
    apply_iem_updates_to_state,
)
from civitasos_runtime.models import EnergyState, TickContext


def _iem_state() -> dict:
    return {
        "schema_version": "iem:v1",
        "identity_id": "agent-alpha",
        "expectation_vector": {
            "survival_probability": 0.90,
            "economic_balance_ratio": 0.50,
        },
        "precision_vector": {
            "survival_probability": 0.50,
            "economic_balance_ratio": 0.40,
        },
        "desire_vector": {"risk_aversion": 0.50},
        "domain_weight_matrix": {},
        "drift_parameters": {
            "predicted_learning_rate": 0.20,
            "desired_learning_rate": 0.02,
            "negative_pressure_streak": 2,
        },
        "relation_expectation_matrix": {},
    }


def test_identity_expectation_emits_survival_economic_and_normative_traces() -> None:
    ctx = TickContext(
        briefing={
            "identity": {"state": "PROVISIONAL", "remaining_epochs": 0, "at_risk": True},
            "normative_context": {
                "breach": True,
                "rule_id": "challenge_window",
                "expected_value": 1.0,
                "actual_value": 0.0,
            },
        }
    )
    energy = EnergyState(balance=2.0, balance_cap=100.0)

    assert apply_identity_expectation_traces(ctx, energy_state=energy, iem_state=_iem_state()) is True

    assert ctx.surprise["survival"]["survival_probability"]["state_kind"] == "predicted"
    assert ctx.surprise["survival"]["survival_probability"]["valence"] == "negative"
    assert ctx.action_bias["survival"]["survival_probability"] == "reduce_action_intensity"
    assert ctx.surprise["economic"]["economic_balance_ratio"]["valence"] == "negative"
    assert ctx.action_bias["economic"]["economic_balance_ratio"] == "conserve_energy"
    assert ctx.surprise["constitutional"]["challenge_window"]["state_kind"] == "normative"
    assert ctx.action_bias["normative"]["challenge_window"] == "governance_trigger"
    assert any(update.local_update_blocked for update in ctx.expectation_updates)
    assert any(update.rule.value == "slow_trait_drift" for update in ctx.expectation_updates)


def test_iem_replay_applies_non_normative_updates_only() -> None:
    ctx = TickContext(
        briefing={
            "identity": {"state": "PROVISIONAL", "remaining_epochs": 0, "at_risk": True},
            "normative_context": {"breach": True, "rule_id": "constitution_rule"},
        }
    )
    state = _iem_state()

    apply_identity_expectation_traces(
        ctx,
        energy_state=EnergyState(balance=2.0, balance_cap=100.0),
        iem_state=state,
    )
    update_entries = [
        {
            "target": update.target,
            "parameter_name": update.parameter_name,
            "new_value": update.new_value,
            "local_update_blocked": update.local_update_blocked,
        }
        for update in ctx.expectation_updates
    ]

    replayed = apply_iem_updates_to_state(state, update_entries)

    assert replayed["expectation_vector"]["survival_probability"] < 0.90
    assert replayed["precision_vector"]["survival_probability"] < 0.50
    assert replayed["desire_vector"]["risk_aversion"] > 0.50
    assert "constitution_rule" not in replayed.get("normative_state", {})