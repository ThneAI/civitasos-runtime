from __future__ import annotations

from civitasos_runtime.models import RelationExpectationVector
from civitasos_runtime.relation_learning import (
    MAX_ABS_DELTA,
    RelationEvidence,
    calculate_relation_update,
    learning_provenance,
)


def _evidence(
    ref: str,
    outcome_kind: str,
    *,
    provider: str = "provider-a",
    owner_id: str = "owner-a",
    risk_class: str = "normal",
) -> RelationEvidence:
    return RelationEvidence(
        ref=ref,
        outcome_kind=outcome_kind,
        task_kind="bounded-review",
        required_capability="analysis",
        provider=provider,
        owner_id=owner_id,
        risk_class=risk_class,
        confidence=1.0,
        upstream_event_id=f"event:{ref}",
    )


def test_outcomes_produce_ordered_relation_deltas() -> None:
    before = RelationExpectationVector()
    repaired = calculate_relation_update(
        before,
        [_evidence("repair:1", "settlement_confirmed")],
    )
    disputed = calculate_relation_update(
        before,
        [_evidence("failure:1", "post_delivery_dispute")],
    )
    failed = calculate_relation_update(
        before,
        [_evidence("failure:2", "post_delivery_failure", risk_class="high")],
    )

    assert repaired.applied_deltas["expected_trust"] > 0
    assert disputed.applied_deltas["expected_trust"] < 0
    assert failed.applied_deltas["expected_trust"] < disputed.applied_deltas["expected_trust"]
    assert failed.applied_deltas["expected_betrayal_risk"] > (
        disputed.applied_deltas["expected_betrayal_risk"]
    )


def test_provider_and_owner_are_neutral_dimensions() -> None:
    before = RelationExpectationVector()
    left = calculate_relation_update(
        before,
        [
            _evidence(
                "failure:left",
                "post_delivery_dispute",
                provider="deepseek-api-agent",
                owner_id="audit-owner",
            )
        ],
    )
    right = calculate_relation_update(
        before,
        [
            _evidence(
                "failure:right",
                "post_delivery_dispute",
                provider="local-gpu-agent",
                owner_id="observability-owner",
            )
        ],
    )

    assert left.applied_deltas == right.applied_deltas
    assert left.after == right.after


def test_replayed_source_event_is_not_learned_twice() -> None:
    before = RelationExpectationVector()
    result = calculate_relation_update(
        before,
        [_evidence("failure:already-seen", "post_delivery_failure")],
        prior_source_event_ids=["failure:already-seen"],
        prior_sample_count=4,
    )

    assert not result.novel_evidence
    assert result.duplicate_refs == ("failure:already-seen",)
    assert result.after == before
    assert result.sample_count == 4


def test_accumulated_evidence_is_bounded_and_explained() -> None:
    before = RelationExpectationVector()
    result = calculate_relation_update(
        before,
        [
            _evidence(
                f"failure:{index}",
                "relation_repair_relapse",
                risk_class="critical",
            )
            for index in range(10)
        ],
    )
    provenance = learning_provenance(result)

    for parameter, cap in MAX_ABS_DELTA.items():
        assert abs(result.bounded_deltas[parameter]) <= cap
    assert result.bounded_deltas["expected_trust"] == -MAX_ABS_DELTA["expected_trust"]
    assert provenance["raw_deltas"]["expected_trust"] < (
        provenance["bounded_deltas"]["expected_trust"]
    )
    assert len(provenance["components"]) == 10
