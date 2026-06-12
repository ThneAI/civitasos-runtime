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
from .relation_learning import (
    RelationEvidence,
    calculate_relation_update,
    learning_provenance,
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
    key = _relation_key(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        state_kind=ExpectationStateKind.PREDICTED,
    )
    prior, prior_source_event_ids, prior_sample_count = _prior_state(
        relation=relation,
        key=key,
        recall=recall,
    )
    evidence = _relation_evidence(relation, relation_id=relation_id)
    learning = calculate_relation_update(
        prior,
        evidence,
        prior_source_event_ids=prior_source_event_ids,
        prior_sample_count=prior_sample_count,
    )
    if not learning.novel_evidence:
        return False
    after = learning.after
    source_event_ids = _dedupe(
        [*prior_source_event_ids, *(item.ref for item in learning.novel_evidence)]
    )
    provenance = learning_provenance(learning)

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
    expectation_payload["sample_count"] = learning.sample_count
    expectation_payload["learning_provenance"] = provenance

    relation_expectations = ctx.expectations.setdefault("relation", {})
    if isinstance(relation_expectations, dict):
        relation_expectations[key] = expectation_payload

    actual_delivery = learning.weighted_actual_delivery
    surprise_score = round(abs(actual_delivery - prior.expected_delivery_quality) * after.precision, 4)
    valence = "negative" if learning.applied_deltas["expected_trust"] < 0 else "positive"
    failure_refs = [
        item.ref
        for item in learning.novel_evidence
        if item.outcome_kind != "settlement_confirmed"
    ]
    repair_refs = [
        item.ref
        for item in learning.novel_evidence
        if item.outcome_kind == "settlement_confirmed"
    ]
    training_invariants = _training_invariants(
        relation_id=relation_id,
        from_identity=from_identity,
        to_identity=to_identity,
        before=prior,
        after=after,
        failure_refs=failure_refs,
        repair_refs=repair_refs,
        source_event_ids=source_event_ids,
        provenance=provenance,
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
        outcome_kinds=[
            str(component.get("outcome_kind") or "")
            for component in provenance["components"]
            if isinstance(component, dict)
        ],
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
            provenance=provenance,
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


def _prior_state(
    *, relation: dict[str, Any], key: str, recall: RecallFn | None,
) -> tuple[RelationExpectationVector, list[str], int]:
    inline = relation.get("relation_expectation") or relation.get("expectation")
    if isinstance(inline, dict):
        return _prior_state_from_mapping(inline)
    if recall is not None:
        try:
            stored = recall(f"relation_expectation:{key}")
        except Exception:
            stored = None
        if isinstance(stored, dict):
            return _prior_state_from_mapping(stored)
    return RelationExpectationVector(), [], 0


def _action_bias(
    *,
    relation_id: str,
    from_identity: str,
    to_identity: str,
    vector: RelationExpectationVector,
    source_event_ids: list[str],
    outcome_kinds: list[str],
) -> dict[str, Any]:
    verification_level = "baseline"
    if vector.expected_betrayal_risk >= 0.40 or vector.expected_trust < 0.55:
        verification_level = "strict"
    elif vector.expected_betrayal_risk >= 0.25 or vector.expected_trust < 0.65:
        verification_level = "elevated"
    if any(kind == "relation_repair_relapse" for kind in outcome_kinds):
        verification_level = "strict"
    elif any(
        kind in {"post_delivery_dispute", "post_delivery_failure"}
        for kind in outcome_kinds
    ) and verification_level == "baseline":
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
        "evidence_outcome_kinds": _dedupe(outcome_kinds),
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
    provenance: dict[str, Any],
) -> dict[str, Any]:
    before_payload = asdict(before)
    after_payload = asdict(after)
    deltas = {
        key: round(float(after_payload[key]) - float(before_payload[key]), 4)
        for key in before_payload
        if key in after_payload
    }
    components = provenance["components"]
    component_deltas = {
        "failure": _component_total(components, negative=True),
        "repair": _component_total(components, negative=False),
        "net_unclamped": dict(provenance["raw_deltas"]),
    }
    negative_sample = bool(failure_refs)
    repair_sample = bool(repair_refs)
    current_refs = set(_dedupe([*failure_refs, *repair_refs]))
    history_preserved = bool(source_event_ids) and current_refs.issubset(
        set(source_event_ids)
    )
    return {
        "schema_version": "h0g_relation_training.v2",
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
        and _repair_slow_recovery(component_deltas["repair"]),
        "history_preserved_present": history_preserved,
        "normative_relation_guard_present": True,
        "failure_refs": failure_refs,
        "repair_refs": repair_refs,
        "source_event_ids": source_event_ids,
        "deltas": deltas,
        "learning_component_deltas": component_deltas,
        "learning_provenance": provenance,
    }


def _component_total(components: Any, *, negative: bool) -> dict[str, float]:
    total: dict[str, float] = {}
    if not isinstance(components, list):
        return total
    for component in components:
        if not isinstance(component, dict):
            continue
        is_negative = component.get("outcome_kind") != "settlement_confirmed"
        if is_negative != negative:
            continue
        for key, value in (component.get("component_deltas") or {}).items():
            total[str(key)] = total.get(str(key), 0.0) + _float(value, 0.0)
    return {key: round(value, 6) for key, value in total.items()}


def _negative_fast_learning(deltas: dict[str, float]) -> bool:
    return (
        deltas.get("expected_trust", 0.0) < 0.0
        and deltas.get("expected_delivery_quality", 0.0) < 0.0
        and deltas.get("expected_betrayal_risk", 0.0) > 0.0
        and deltas.get("precision", 0.0) > 0.0
    )


def _repair_slow_recovery(
    deltas: dict[str, float],
) -> bool:
    return (
        deltas.get("expected_trust", 0.0) > 0.0
        and deltas.get("expected_betrayal_risk", 0.0) < 0.0
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
    provenance: dict[str, Any],
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
                    "delta_provenance": provenance,
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


def _prior_state_from_mapping(
    value: dict[str, Any],
) -> tuple[RelationExpectationVector, list[str], int]:
    vector_payload = value.get("expectation")
    if not isinstance(vector_payload, dict):
        vector_payload = value
    refs = value.get("source_event_ids")
    if not isinstance(refs, list):
        refs = []
    sample_count = value.get("sample_count")
    try:
        parsed_sample_count = int(sample_count)
    except (TypeError, ValueError):
        parsed_sample_count = len(refs)
    return (
        _vector_from_mapping(vector_payload),
        _dedupe(str(ref) for ref in refs if str(ref)),
        max(parsed_sample_count, len(refs)),
    )


def _relation_evidence(
    relation: dict[str, Any],
    *,
    relation_id: str,
) -> list[RelationEvidence]:
    evidence_by_ref: dict[str, RelationEvidence] = {}
    for ref in _memory_refs(relation):
        if ref.startswith("failure:") and _ref_matches_relation(ref, relation_id):
            evidence_by_ref[ref] = RelationEvidence(
                ref=ref,
                outcome_kind="post_delivery_failure",
            )
        elif ref.startswith("repair:") and _ref_matches_relation(ref, relation_id):
            evidence_by_ref[ref] = RelationEvidence(
                ref=ref,
                outcome_kind="settlement_confirmed",
            )
    for event in _event_list(relation.get("recent_failures")):
        item = _evidence_from_event(event, relation_id=relation_id, negative=True)
        if item.ref:
            evidence_by_ref[item.ref] = item
    for event in _event_list(relation.get("recent_repairs")):
        item = _evidence_from_event(event, relation_id=relation_id, negative=False)
        if item.ref:
            evidence_by_ref[item.ref] = item
    return list(evidence_by_ref.values())


def _evidence_from_event(
    event: dict[str, Any],
    *,
    relation_id: str,
    negative: bool,
) -> RelationEvidence:
    prefix = "failure" if negative else "repair"
    default_outcome = "post_delivery_failure" if negative else "settlement_confirmed"
    return RelationEvidence(
        ref=_event_ref(prefix, event, relation_id=relation_id),
        outcome_kind=_first_text(event, "event_kind", "outcome_status") or default_outcome,
        task_kind=_first_text(event, "task_kind", "kind"),
        required_capability=_first_text(event, "required_capability", "capability"),
        provider=_first_text(event, "agent_provider", "provider"),
        owner_id=_first_text(event, "task_owner_id", "owner_id"),
        risk_class=_first_text(event, "risk_class") or "normal",
        confidence=_float(event.get("evidence_confidence"), 1.0),
        upstream_event_id=_first_text(event, "event_id"),
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
    timestamp = _first_text(
        event,
        "failed_at",
        "repaired_at",
        "observed_at",
        "timestamp",
    )
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
