"""H.0-B directed relation expectation matrix minimal implementation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from .models import (
    DirectedRelationExpectation,
    ExpectationDomain,
    ExpectationLifecycleState,
    ExpectationStateKind,
    ExpectationTrace,
    ExpectationUpdate,
    ExpectationUpdateRule,
    RelationExpectationVector,
    TickContext,
)

RecallFn = Callable[[str], Any]


def apply_relation_matrix_expectation(
    ctx: TickContext,
    *,
    local_identity: str = "",
    recall: RecallFn | None = None,
) -> bool:
    """Populate H.0-B relation expectation traces from G.3.5 relation facts.

    Returns True only when a relation-pair failure or repair evidence item was
    converted into a directed predicted-state update and an auditable action bias.
    """
    relation = _relation_context(ctx.briefing)
    if not relation:
        return False

    relation_id = _first_text(relation, "relation_id", "id", "relation_context_id")
    relation_pair = relation.get("relation_pair")
    direction = _direction(relation_pair, local_identity=local_identity)
    if not relation_id or direction is None:
        return False

    from_identity, to_identity = direction
    failure_refs = _failure_refs(relation, relation_id=relation_id)
    repair_refs = _repair_refs(relation, relation_id=relation_id)
    if not failure_refs and not repair_refs:
        return False

    key = _relation_key(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        state_kind=ExpectationStateKind.PREDICTED,
    )
    prior = _prior_vector(relation=relation, key=key, recall=recall)
    after = _updated_vector(prior, failures=len(failure_refs), repairs=len(repair_refs))
    source_event_ids = _dedupe([*failure_refs, *repair_refs])

    expectation = DirectedRelationExpectation(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        state_kind=ExpectationStateKind.PREDICTED,
        expectation=after,
        source_event_ids=source_event_ids,
    )
    expectation_payload = asdict(expectation)
    expectation_payload["state_kind"] = expectation.state_kind.value

    relation_expectations = ctx.expectations.setdefault("relation", {})
    if isinstance(relation_expectations, dict):
        relation_expectations[key] = expectation_payload

    actual_delivery = 0.0 if failure_refs else 0.85
    surprise_score = round(abs(actual_delivery - prior.expected_delivery_quality) * after.precision, 4)
    valence = "negative" if failure_refs else "positive"
    training_invariants = _training_invariants(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        before=prior,
        after=after,
        failure_refs=failure_refs,
        repair_refs=repair_refs,
        source_event_ids=source_event_ids,
    )
    surprise = ExpectationTrace(
        domain=ExpectationDomain.RELATION,
        state_kind=ExpectationStateKind.PREDICTED,
        expected_value=prior.expected_delivery_quality,
        actual_value=actual_delivery,
        surprise_score=surprise_score,
        precision=after.precision,
        valence=valence,
        stake=float(max(len(source_event_ids), 1)),
        source_event_id=";".join(source_event_ids),
        lifecycle_state=(
            ExpectationLifecycleState.VIOLATED
            if failure_refs
            else ExpectationLifecycleState.CONFIRMED
        ),
    )
    relation_surprise = ctx.surprise.setdefault("relation", {})
    if isinstance(relation_surprise, dict):
        relation_surprise[key] = _trace_payload(surprise)

    action_bias = _action_bias(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        vector=after,
        source_event_ids=source_event_ids,
    )
    relation_bias = ctx.action_bias.setdefault("relation", {})
    if isinstance(relation_bias, dict):
        relation_bias[key] = action_bias

    drive = ctx.drive.setdefault("relation", {})
    if isinstance(drive, dict):
        drive[key] = {
            "drive_score": round(surprise_score * action_bias["required_stake_multiplier"], 4),
            "actionability": 1.0,
            "action_bias": action_bias,
            "training_invariants": training_invariants,
            "constitution_verdict": "allowed: predicted relation update only",
        }

    ctx.expectation_updates.extend(
        _update_log(
            key=key,
            before=prior,
            after=after,
            relation_id=relation_id,
            from_identity=from_identity,
            to_identity=to_identity,
            source_event_ids=source_event_ids,
        )
    )
    ctx.expectation_updates.append(
        _normative_guard_update(key=key, source_event_ids=source_event_ids)
    )

    ctx.briefing["h0_relation_expectation"] = expectation_payload
    ctx.briefing["h0_relation_action_bias"] = action_bias
    ctx.briefing["h0_relation_training_invariants"] = training_invariants
    return True


def _relation_context(briefing: dict[str, Any]) -> dict[str, Any]:
    relation = briefing.get("relation_context")
    if not isinstance(relation, dict):
        relation = briefing.get("g3_relation_context")
    return relation if isinstance(relation, dict) else {}


def _direction(value: Any, *, local_identity: str) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    requester = str(value.get("requester") or "").strip()
    worker = str(value.get("worker") or value.get("peer_agent_id") or "").strip()
    agents = [str(item).strip() for item in value.get("agents", []) if str(item).strip()] \
        if isinstance(value.get("agents"), list) else []

    local = str(local_identity or "").strip()
    if local and requester and worker:
        if local == requester:
            return requester, worker
        if local == worker:
            return worker, requester
    if worker and requester:
        return worker, requester
    if local and agents:
        for agent_id in agents:
            if agent_id != local:
                return local, agent_id
    if len(agents) >= 2:
        return agents[0], agents[1]
    return None


def _failure_refs(relation: dict[str, Any], *, relation_id: str) -> list[str]:
    refs = [
        ref for ref in _memory_refs(relation)
        if ref.startswith("failure:") and _ref_matches_relation(ref, relation_id)
    ]
    for event in _event_list(relation.get("recent_failures")):
        refs.append(_event_ref("failure", event, relation_id=relation_id))
    return _dedupe(ref for ref in refs if ref)


def _repair_refs(relation: dict[str, Any], *, relation_id: str) -> list[str]:
    refs = [
        ref for ref in _memory_refs(relation)
        if ref.startswith("repair:") and _ref_matches_relation(ref, relation_id)
    ]
    for event in _event_list(relation.get("recent_repairs")):
        refs.append(_event_ref("repair", event, relation_id=relation_id))
    return _dedupe(ref for ref in refs if ref)


def _prior_vector(
    *, relation: dict[str, Any], key: str, recall: RecallFn | None,
) -> RelationExpectationVector:
    inline = relation.get("relation_expectation") or relation.get("expectation")
    if isinstance(inline, dict):
        return _vector_from_mapping(inline)
    if recall is not None:
        try:
            stored = recall(f"relation_expectation:{key}")
        except Exception:
            stored = None
        if isinstance(stored, dict):
            if isinstance(stored.get("expectation"), dict):
                return _vector_from_mapping(stored["expectation"])
            return _vector_from_mapping(stored)
    return RelationExpectationVector()


def _updated_vector(
    before: RelationExpectationVector, *, failures: int, repairs: int,
) -> RelationExpectationVector:
    failure_weight = min(float(failures), 5.0)
    repair_weight = min(float(repairs), 5.0)
    return RelationExpectationVector(
        expected_trust=_clamp(before.expected_trust - 0.12 * failure_weight + 0.04 * repair_weight),
        expected_delivery_quality=_clamp(
            before.expected_delivery_quality - 0.18 * failure_weight + 0.05 * repair_weight
        ),
        expected_cooperation=_clamp(before.expected_cooperation - 0.10 * failure_weight + 0.04 * repair_weight),
        expected_betrayal_risk=_clamp(
            before.expected_betrayal_risk + 0.20 * failure_weight - 0.05 * repair_weight
        ),
        expected_repair_probability=_clamp(
            before.expected_repair_probability - 0.04 * failure_weight + 0.14 * repair_weight
        ),
        precision=_clamp(before.precision + 0.08 * (failure_weight + repair_weight), 0.05, 0.95),
    )


def _action_bias(
    *,
    relation_id: str,
    from_identity: str,
    to_identity: str,
    vector: RelationExpectationVector,
    source_event_ids: list[str],
) -> dict[str, Any]:
    verification_level = "baseline"
    if vector.expected_betrayal_risk >= 0.40 or vector.expected_trust < 0.55:
        verification_level = "strict"
    elif vector.expected_betrayal_risk >= 0.25 or vector.expected_trust < 0.65:
        verification_level = "elevated"

    trust_penalty = max(0.0, 0.70 - vector.expected_trust)
    return {
        "relation_id": relation_id,
        "from": from_identity,
        "to": to_identity,
        "verification_level": verification_level,
        "required_stake_multiplier": round(
            1.0 + vector.expected_betrayal_risk * 2.0 + trust_penalty,
            3,
        ),
        "claim_priority_delta": round(-0.5 * vector.expected_betrayal_risk, 3),
        "trust_hint": round(vector.expected_trust, 3),
        "direct_match_allowed": vector.expected_betrayal_risk < 0.75,
        "reason_event_ids": source_event_ids,
    }


def _training_invariants(
    *,
    relation_id: str,
    from_identity: str,
    to_identity: str,
    before: RelationExpectationVector,
    after: RelationExpectationVector,
    failure_refs: list[str],
    repair_refs: list[str],
    source_event_ids: list[str],
) -> dict[str, Any]:
    before_payload = asdict(before)
    after_payload = asdict(after)
    deltas = {
        key: round(float(after_payload[key]) - float(before_payload[key]), 4)
        for key in before_payload
        if key in after_payload
    }
    component_deltas = _learning_component_deltas(
        failures=len(failure_refs),
        repairs=len(repair_refs),
    )
    learning_rates = {
        "expected_trust": {"negative": 0.12, "repair": 0.04},
        "expected_delivery_quality": {"negative": 0.18, "repair": 0.05},
        "expected_cooperation": {"negative": 0.10, "repair": 0.04},
        "expected_betrayal_risk": {"negative": 0.20, "repair": 0.05},
        "expected_repair_probability": {"negative": 0.04, "repair": 0.14},
    }
    negative_sample = bool(failure_refs)
    repair_sample = bool(repair_refs)
    history_preserved = bool(source_event_ids) and set(source_event_ids) == set(
        _dedupe([*failure_refs, *repair_refs])
    )
    return {
        "schema_version": "h0g_relation_training.v1",
        "relation_id": relation_id,
        "from": from_identity,
        "to": to_identity,
        "state_kind": ExpectationStateKind.PREDICTED.value,
        "training_sample_present": bool(source_event_ids),
        "negative_sample_present": negative_sample,
        "negative_fast_learning_present": negative_sample
        and _negative_fast_learning(component_deltas["failure"]),
        "repair_sample_present": repair_sample,
        "repair_slow_recovery_present": repair_sample
        and _repair_slow_recovery(component_deltas["repair"], learning_rates),
        "history_preserved_present": history_preserved,
        "normative_relation_guard_present": True,
        "failure_refs": failure_refs,
        "repair_refs": repair_refs,
        "source_event_ids": source_event_ids,
        "deltas": deltas,
        "learning_component_deltas": component_deltas,
        "learning_rates": learning_rates,
    }


def _learning_component_deltas(*, failures: int, repairs: int) -> dict[str, dict[str, float]]:
    failure_weight = min(float(failures), 5.0)
    repair_weight = min(float(repairs), 5.0)
    failure = {
        "expected_trust": -0.12 * failure_weight,
        "expected_delivery_quality": -0.18 * failure_weight,
        "expected_cooperation": -0.10 * failure_weight,
        "expected_betrayal_risk": 0.20 * failure_weight,
        "expected_repair_probability": -0.04 * failure_weight,
        "precision": 0.08 * failure_weight,
    }
    repair = {
        "expected_trust": 0.04 * repair_weight,
        "expected_delivery_quality": 0.05 * repair_weight,
        "expected_cooperation": 0.04 * repair_weight,
        "expected_betrayal_risk": -0.05 * repair_weight,
        "expected_repair_probability": 0.14 * repair_weight,
        "precision": 0.08 * repair_weight,
    }
    net = {
        key: failure.get(key, 0.0) + repair.get(key, 0.0)
        for key in failure
    }
    return {
        "failure": {key: round(value, 4) for key, value in failure.items()},
        "repair": {key: round(value, 4) for key, value in repair.items()},
        "net_unclamped": {key: round(value, 4) for key, value in net.items()},
    }


def _negative_fast_learning(deltas: dict[str, float]) -> bool:
    return (
        deltas.get("expected_trust", 0.0) < 0.0
        and deltas.get("expected_delivery_quality", 0.0) < 0.0
        and deltas.get("expected_betrayal_risk", 0.0) > 0.0
        and deltas.get("precision", 0.0) > 0.0
    )


def _repair_slow_recovery(
    deltas: dict[str, float],
    learning_rates: dict[str, dict[str, float]],
) -> bool:
    trust_rates = learning_rates["expected_trust"]
    betrayal_rates = learning_rates["expected_betrayal_risk"]
    return (
        trust_rates["repair"] < trust_rates["negative"]
        and betrayal_rates["repair"] < betrayal_rates["negative"]
        and deltas.get("expected_repair_probability", 0.0) > 0.0
        and deltas.get("precision", 0.0) > 0.0
    )


def _update_log(
    *,
    key: str,
    before: RelationExpectationVector,
    after: RelationExpectationVector,
    relation_id: str,
    from_identity: str,
    to_identity: str,
    source_event_ids: list[str],
) -> list[ExpectationUpdate]:
    before_payload = asdict(before)
    after_payload = asdict(after)
    updates: list[ExpectationUpdate] = []
    for parameter, old_value in before_payload.items():
        new_value = after_payload[parameter]
        if old_value == new_value:
            continue
        updates.append(
            ExpectationUpdate(
                target=f"relation_expectation:{key}",
                parameter_name=parameter,
                old_value=old_value,
                new_value=new_value,
                rule=ExpectationUpdateRule.RELATION_MATRIX_UPDATE,
                reason_event=";".join(source_event_ids),
                update_params={
                    "relation_id": relation_id,
                    "from": from_identity,
                    "to": to_identity,
                    "state_kind": ExpectationStateKind.PREDICTED.value,
                    "before": before_payload,
                    "after": after_payload,
                },
                constitution_verdict="allowed: predicted relation update",
            )
        )
    return updates


def _normative_guard_update(*, key: str, source_event_ids: list[str]) -> ExpectationUpdate:
    return ExpectationUpdate(
        target=f"relation_expectation:{key}:normative_guard",
        parameter_name="normative_relation",
        old_value="governance_owned",
        new_value="local_update_rejected",
        rule=ExpectationUpdateRule.GOVERNANCE_TRIGGER,
        reason_event=";".join(source_event_ids),
        constitution_verdict="blocked: normative relation updates require governance",
        local_update_blocked=True,
    )


def _trace_payload(trace: ExpectationTrace) -> dict[str, Any]:
    payload = asdict(trace)
    payload["domain"] = trace.domain.value
    payload["state_kind"] = trace.state_kind.value
    payload["lifecycle_state"] = trace.lifecycle_state.value
    return payload


def _vector_from_mapping(value: dict[str, Any]) -> RelationExpectationVector:
    return RelationExpectationVector(
        expected_trust=_float(value.get("expected_trust"), 0.72),
        expected_delivery_quality=_float(value.get("expected_delivery_quality"), 0.72),
        expected_cooperation=_float(value.get("expected_cooperation"), 0.70),
        expected_betrayal_risk=_float(value.get("expected_betrayal_risk"), 0.12),
        expected_repair_probability=_float(value.get("expected_repair_probability"), 0.55),
        precision=_float(value.get("precision"), 0.35),
    )


def _memory_refs(relation: dict[str, Any]) -> list[str]:
    refs = relation.get("memory_refs")
    if refs is None:
        refs = relation.get("relation_memory_refs")
    if isinstance(refs, str):
        parts = refs.replace(",", ";").split(";")
        return [part.strip() for part in parts if part.strip()]
    if isinstance(refs, (list, tuple, set)):
        return [str(ref).strip() for ref in refs if str(ref).strip()]
    return []


def _event_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, dict)]


def _event_ref(prefix: str, event: dict[str, Any], *, relation_id: str) -> str:
    task_id = _first_text(event, "task_id", "id")
    event_relation = _first_text(event, "r2r_relation_id", "relation_id") or relation_id
    timestamp = _first_text(event, "failed_at", "repaired_at", "timestamp")
    if not task_id:
        return ""
    suffix = f":{_ref_token(timestamp)}" if timestamp else ""
    return f"{prefix}:{event_relation}:{task_id}{suffix}"


def _ref_matches_relation(ref: str, relation_id: str) -> bool:
    if not relation_id:
        return True
    return ref.startswith(f"failure:{relation_id}:") or ref.startswith(f"repair:{relation_id}:")


def _relation_key(
    *, relation_id: str, from_identity: str, to_identity: str, state_kind: ExpectationStateKind,
) -> str:
    return f"{relation_id}|{from_identity}->{to_identity}|{state_kind.value}"


def _first_text(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(max(low, min(high, value)), 4)


def _ref_token(value: str) -> str:
    token = str(value or "").strip()
    for char in (" ", "/", "\\", ":", "+"):
        token = token.replace(char, "_")
    return token or "unknown"


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out