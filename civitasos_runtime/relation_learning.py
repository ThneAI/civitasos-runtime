"""Evidence-driven, bounded relation expectation learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .models import RelationExpectationVector

PARAMETERS = (
    "expected_trust",
    "expected_delivery_quality",
    "expected_cooperation",
    "expected_betrayal_risk",
    "expected_repair_probability",
    "precision",
)

OUTCOME_DELTAS: dict[str, dict[str, float]] = {
    "settlement_confirmed": {
        "expected_trust": 0.04,
        "expected_delivery_quality": 0.05,
        "expected_cooperation": 0.04,
        "expected_betrayal_risk": -0.05,
        "expected_repair_probability": 0.14,
        "precision": 0.06,
    },
    "post_delivery_dispute": {
        "expected_trust": -0.08,
        "expected_delivery_quality": -0.12,
        "expected_cooperation": -0.07,
        "expected_betrayal_risk": 0.14,
        "expected_repair_probability": -0.03,
        "precision": 0.05,
    },
    "post_delivery_failure": {
        "expected_trust": -0.12,
        "expected_delivery_quality": -0.18,
        "expected_cooperation": -0.10,
        "expected_betrayal_risk": 0.20,
        "expected_repair_probability": -0.04,
        "precision": 0.08,
    },
    "relation_repair_relapse": {
        "expected_trust": -0.16,
        "expected_delivery_quality": -0.20,
        "expected_cooperation": -0.14,
        "expected_betrayal_risk": 0.24,
        "expected_repair_probability": -0.10,
        "precision": 0.08,
    },
}

OUTCOME_ACTUAL_DELIVERY = {
    "settlement_confirmed": 0.90,
    "post_delivery_dispute": 0.35,
    "post_delivery_failure": 0.0,
    "relation_repair_relapse": 0.0,
}

RISK_WEIGHTS = {
    "low": 0.75,
    "normal": 1.0,
    "medium": 1.0,
    "high": 1.20,
    "critical": 1.35,
}

MAX_ABS_DELTA = {
    "expected_trust": 0.18,
    "expected_delivery_quality": 0.22,
    "expected_cooperation": 0.18,
    "expected_betrayal_risk": 0.24,
    "expected_repair_probability": 0.18,
    "precision": 0.10,
}


@dataclass(frozen=True)
class RelationEvidence:
    """One immutable relation outcome observation."""

    ref: str
    outcome_kind: str
    task_kind: str = ""
    required_capability: str = ""
    provider: str = ""
    owner_id: str = ""
    risk_class: str = "normal"
    confidence: float = 1.0
    upstream_event_id: str = ""


@dataclass(frozen=True)
class RelationLearningResult:
    """Auditable output of one bounded relation-learning step."""

    after: RelationExpectationVector
    novel_evidence: tuple[RelationEvidence, ...]
    duplicate_refs: tuple[str, ...]
    raw_deltas: dict[str, float]
    bounded_deltas: dict[str, float]
    applied_deltas: dict[str, float]
    components: tuple[dict[str, object], ...]
    prior_sample_count: int
    sample_count: int
    weighted_actual_delivery: float


def calculate_relation_update(
    before: RelationExpectationVector,
    evidence: Iterable[RelationEvidence],
    *,
    prior_source_event_ids: Iterable[str] = (),
    prior_sample_count: int = 0,
) -> RelationLearningResult:
    """Apply novel evidence with explicit provenance and per-step delta caps."""
    prior_refs = {str(ref) for ref in prior_source_event_ids if str(ref)}
    novel: list[RelationEvidence] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        if not item.ref or item.ref in prior_refs or item.ref in seen:
            if item.ref:
                duplicates.append(item.ref)
            continue
        seen.add(item.ref)
        novel.append(item)

    raw = {parameter: 0.0 for parameter in PARAMETERS}
    components: list[dict[str, object]] = []
    delivery_weighted_sum = 0.0
    delivery_weight_sum = 0.0
    for item in novel:
        outcome_kind = _normalized_outcome(item.outcome_kind)
        base = OUTCOME_DELTAS[outcome_kind]
        risk_weight = RISK_WEIGHTS.get(
            str(item.risk_class or "normal").strip().lower(),
            RISK_WEIGHTS["normal"],
        )
        confidence = _clamp(float(item.confidence), 0.0, 1.0)
        prior_precision = _clamp(float(before.precision), 0.05, 0.95)
        history_adaptation = _clamp(1.0 - 0.50 * prior_precision, 0.50, 0.95)
        actual_delivery = OUTCOME_ACTUAL_DELIVERY[outcome_kind]
        surprise = abs(actual_delivery - float(before.expected_delivery_quality))
        surprise_weight = _clamp(0.75 + 0.50 * surprise, 0.75, 1.25)
        effective_weight = risk_weight * confidence * history_adaptation * surprise_weight
        component_deltas = {
            parameter: round(base[parameter] * effective_weight, 6)
            for parameter in PARAMETERS
        }
        for parameter, delta in component_deltas.items():
            raw[parameter] += delta
        delivery_weighted_sum += actual_delivery * effective_weight
        delivery_weight_sum += effective_weight
        components.append(
            {
                "source_ref": item.ref,
                "upstream_event_id": item.upstream_event_id,
                "outcome_kind": outcome_kind,
                "task_kind": item.task_kind,
                "required_capability": item.required_capability,
                "provider": item.provider,
                "owner_id": item.owner_id,
                "risk_class": item.risk_class,
                "confidence": confidence,
                "risk_weight": risk_weight,
                "history_adaptation": round(history_adaptation, 6),
                "surprise_weight": round(surprise_weight, 6),
                "effective_weight": round(effective_weight, 6),
                "base_deltas": dict(base),
                "component_deltas": component_deltas,
            }
        )

    raw = {key: round(value, 6) for key, value in raw.items()}
    bounded = {
        parameter: round(
            _clamp(raw[parameter], -MAX_ABS_DELTA[parameter], MAX_ABS_DELTA[parameter]),
            6,
        )
        for parameter in PARAMETERS
    }
    before_payload = asdict(before)
    after_payload = {
        parameter: round(
            _clamp(
                float(before_payload[parameter]) + bounded[parameter],
                0.05 if parameter == "precision" else 0.0,
                0.95 if parameter == "precision" else 1.0,
            ),
            6,
        )
        for parameter in PARAMETERS
    }
    applied = {
        parameter: round(after_payload[parameter] - float(before_payload[parameter]), 6)
        for parameter in PARAMETERS
    }
    weighted_actual_delivery = (
        delivery_weighted_sum / delivery_weight_sum
        if delivery_weight_sum > 0
        else float(before.expected_delivery_quality)
    )
    return RelationLearningResult(
        after=RelationExpectationVector(**after_payload),
        novel_evidence=tuple(novel),
        duplicate_refs=tuple(dict.fromkeys(duplicates)),
        raw_deltas=raw,
        bounded_deltas=bounded,
        applied_deltas=applied,
        components=tuple(components),
        prior_sample_count=max(int(prior_sample_count), 0),
        sample_count=max(int(prior_sample_count), 0) + len(novel),
        weighted_actual_delivery=round(weighted_actual_delivery, 6),
    )


def learning_provenance(result: RelationLearningResult) -> dict[str, object]:
    """Return the stable audit payload stored beside relation state updates."""
    return {
        "schema_version": "relation-learning-provenance:v1",
        "source_event_ids": [item.ref for item in result.novel_evidence],
        "duplicate_source_event_ids": list(result.duplicate_refs),
        "prior_sample_count": result.prior_sample_count,
        "sample_count": result.sample_count,
        "raw_deltas": result.raw_deltas,
        "per_step_abs_caps": dict(MAX_ABS_DELTA),
        "bounded_deltas": result.bounded_deltas,
        "applied_deltas": result.applied_deltas,
        "weighted_actual_delivery": result.weighted_actual_delivery,
        "components": list(result.components),
        "identity_neutral_dimensions": ["owner_id", "provider"],
    }


def _normalized_outcome(value: str) -> str:
    outcome = str(value or "").strip().lower()
    aliases = {
        "completed": "settlement_confirmed",
        "repair": "settlement_confirmed",
        "disputed": "post_delivery_dispute",
        "failed": "post_delivery_failure",
        "failure": "post_delivery_failure",
    }
    outcome = aliases.get(outcome, outcome)
    return outcome if outcome in OUTCOME_DELTAS else "post_delivery_failure"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
