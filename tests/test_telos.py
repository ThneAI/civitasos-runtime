from __future__ import annotations

from civitasos_runtime.models import Decision, IntentLayer, TickPhase
from civitasos_runtime.telos import build_telos_alignment, served_intent_layer_for_action


def test_build_telos_alignment_creates_five_layer_stack() -> None:
    alignment = build_telos_alignment(
        {
            "active_tasks": [
                {
                    "task_id": "backend-task-1",
                    "telos": "ship a verified deliverable",
                    "verifier_tools": ["static_analyze", "test_runner"],
                }
            ]
        },
        {},
        {"relation": {"r": {"verification_level": "elevated"}}},
    )

    assert alignment["schema_version"] == "h1_telos_alignment.v1"
    assert alignment["active_layer"] == IntentLayer.SHORT.value
    assert [frame["layer"] for frame in alignment["intent_stack"]] == [
        "immediate",
        "short",
        "mid",
        "long",
        "telos",
    ]
    assert alignment["verification_plan"]["required"] is True
    assert alignment["verification_plan"]["tools"] == ["static_analyze", "test_runner"]


def test_served_intent_layer_for_action_maps_verifier_and_delivery() -> None:
    alignment = build_telos_alignment(
        {"active_tasks": [{"task_id": "t1", "verifier_tools": ["test_runner"]}]},
        {},
        {},
    )

    assert served_intent_layer_for_action("test_runner", {}, alignment) == "short"
    assert served_intent_layer_for_action("task_execute", {}, alignment) == "immediate"
    assert served_intent_layer_for_action("r2r_propose_relation", {}, alignment) == "mid"
    assert served_intent_layer_for_action("create_proposal", {}, alignment) == "long"


def test_models_expose_h1_align_phase_and_decision_layer() -> None:
    decision = Decision(action="task_execute", served_intent_layer="immediate")
    assert TickPhase.ALIGN.value == "align"
    assert decision.served_intent_layer == "immediate"
