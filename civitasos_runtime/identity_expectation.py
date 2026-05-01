"""H.0 identity-level expectation traces and minimal evolution rules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import (
    EnergyState,
    ExpectationDomain,
    ExpectationLifecycleState,
    ExpectationStateKind,
    ExpectationTrace,
    ExpectationUpdate,
    ExpectationUpdateRule,
    TickContext,
)


def apply_identity_expectation_traces(
    ctx: TickContext,
    *,
    energy_state: EnergyState,
    iem_state: dict[str, Any],
) -> bool:
    """Populate identity-level expectation traces.

    This path makes H.0 useful beyond relation-only evidence while keeping
    Drive as action bias and keeping Normative state governance-owned.
    """
    emitted = False
    emitted |= _apply_predicted_scalar(
        ctx,
        iem_state=iem_state,
        domain=ExpectationDomain.SURVIVAL,
        parameter="survival_probability",
        expected_default=0.85,
        actual_value=_survival_actual(ctx.briefing, energy_state),
        stake=_survival_stake(ctx.briefing, energy_state),
        negative_bias="reduce_action_intensity",
        positive_bias="maintain_survival_posture",
    )
    emitted |= _apply_predicted_scalar(
        ctx,
        iem_state=iem_state,
        domain=ExpectationDomain.ECONOMIC,
        parameter="economic_balance_ratio",
        expected_default=0.50,
        actual_value=_economic_actual(energy_state),
        stake=1.0,
        negative_bias="conserve_energy",
        positive_bias="maintain_economic_posture",
    )
    emitted |= _apply_predicted_scalar(
        ctx,
        iem_state=iem_state,
        domain=ExpectationDomain.REPUTATION,
        parameter="reputation_score",
        expected_default=0.65,
        actual_value=_reputation_actual(ctx.briefing, energy_state),
        stake=_reputation_stake(ctx.briefing),
        negative_bias="repair_reputation",
        positive_bias="maintain_reputation_posture",
    )
    emitted |= _apply_predicted_scalar(
        ctx,
        iem_state=iem_state,
        domain=ExpectationDomain.TASK,
        parameter="task_success_probability",
        expected_default=0.75,
        actual_value=_task_actual(ctx.briefing),
        stake=_task_stake(ctx.briefing),
        negative_bias="reduce_task_risk",
        positive_bias="maintain_task_posture",
    )
    emitted |= _apply_predicted_scalar(
        ctx,
        iem_state=iem_state,
        domain=ExpectationDomain.GOVERNANCE,
        parameter="governance_participation",
        expected_default=0.60,
        actual_value=_governance_actual(ctx.briefing),
        stake=_governance_stake(ctx.briefing),
        negative_bias="increase_governance_attention",
        positive_bias="maintain_governance_posture",
    )
    emitted |= _apply_normative_guard(ctx)
    emitted |= _apply_governed_revision(ctx)
    _apply_desired_slow_drift(ctx, iem_state=iem_state)
    return emitted


def apply_iem_updates_to_state(
    state: dict[str, Any],
    update_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply auditable non-normative IEM updates to a persisted state copy."""
    next_state = dict(state)
    for key in (
        "expectation_vector",
        "precision_vector",
        "desire_vector",
        "domain_weight_matrix",
        "drift_parameters",
        "normative_state",
    ):
        if not isinstance(next_state.get(key), dict):
            next_state[key] = {}

    for entry in update_entries:
        if not isinstance(entry, dict) or entry.get("local_update_blocked"):
            continue
        target = str(entry.get("target") or "")
        parameter = str(entry.get("parameter_name") or "")
        if not parameter:
            continue
        if target == "identity_expectation_vector":
            next_state["expectation_vector"][parameter] = entry.get("new_value")
        elif target == "identity_precision_vector":
            next_state["precision_vector"][parameter] = entry.get("new_value")
        elif target == "identity_desire_vector":
            next_state["desire_vector"][parameter] = entry.get("new_value")
        elif target == "identity_drift_parameters":
            next_state["drift_parameters"][parameter] = entry.get("new_value")
        elif target == "normative_state" and _entry_rule(entry) == ExpectationUpdateRule.GOVERNED_REVISION.value:
            next_state["normative_state"][parameter] = entry.get("new_value")
    return next_state


def _apply_predicted_scalar(
    ctx: TickContext,
    *,
    iem_state: dict[str, Any],
    domain: ExpectationDomain,
    parameter: str,
    expected_default: float,
    actual_value: float,
    stake: float,
    negative_bias: str,
    positive_bias: str,
) -> bool:
    expected = _vector_value(iem_state, "expectation_vector", parameter, expected_default)
    precision = _vector_value(iem_state, "precision_vector", parameter, 0.40)
    delta = actual_value - expected
    surprise_score = round(abs(delta) * precision, 4)
    valence = "negative" if delta < -0.001 else "positive"
    lifecycle = (
        ExpectationLifecycleState.CONFIRMED
        if abs(delta) <= 0.05
        else ExpectationLifecycleState.VIOLATED
    )

    trace = ExpectationTrace(
        domain=domain,
        state_kind=ExpectationStateKind.PREDICTED,
        expected_value=round(expected, 4),
        actual_value=round(actual_value, 4),
        surprise_score=surprise_score,
        precision=round(precision, 4),
        valence=valence,
        stake=round(stake, 4),
        source_event_id=f"h0:{domain.value}:{ctx.tick_id}",
        lifecycle_state=lifecycle,
    )
    domain_key = domain.value
    ctx.expectations.setdefault("identity", {})[parameter] = {
        "state_kind": ExpectationStateKind.PREDICTED.value,
        "expected_value": round(expected, 4),
        "precision": round(precision, 4),
    }
    ctx.surprise.setdefault(domain_key, {})[parameter] = _trace_payload(trace)

    action_bias = negative_bias if valence == "negative" and surprise_score > 0 else positive_bias
    ctx.action_bias.setdefault(domain_key, {})[parameter] = action_bias
    ctx.drive.setdefault(domain_key, {})[parameter] = {
        "drive_score": round(surprise_score * max(stake, 0.0), 4),
        "actionability": 0.8 if valence == "negative" else 0.4,
        "action_bias": action_bias,
        "constitution_verdict": "allowed: predicted state action bias only",
    }

    learning_rate = _learning_rate(iem_state, "predicted_learning_rate", 0.20)
    next_expected = _clamp(expected + learning_rate * precision * delta)
    next_precision = _next_precision(precision, delta)
    ctx.expectation_updates.append(
        ExpectationUpdate(
            target="identity_expectation_vector",
            parameter_name=parameter,
            old_value=round(expected, 4),
            new_value=next_expected,
            rule=ExpectationUpdateRule.PRECISION_WEIGHTED_DELTA,
            reason_event=trace.source_event_id,
            update_params={
                "domain": domain.value,
                "actual_value": round(actual_value, 4),
                "surprise_score": surprise_score,
                "learning_rate": learning_rate,
                "precision_before": round(precision, 4),
                "stake": round(stake, 4),
            },
            constitution_verdict="allowed: predicted state update",
        )
    )
    ctx.expectation_updates.append(
        ExpectationUpdate(
            target="identity_precision_vector",
            parameter_name=parameter,
            old_value=round(precision, 4),
            new_value=next_precision,
            rule=ExpectationUpdateRule.PRECISION_WEIGHTED_DELTA,
            reason_event=trace.source_event_id,
            update_params={"domain": domain.value, "delta": round(delta, 4)},
            constitution_verdict="allowed: precision calibration",
        )
    )
    return True


def _apply_normative_guard(ctx: TickContext) -> bool:
    normative = ctx.briefing.get("normative_context")
    if not isinstance(normative, dict):
        normative = ctx.briefing.get("constitutional_context")
    if not isinstance(normative, dict) or not bool(normative.get("breach")):
        return False

    rule_id = str(normative.get("rule_id") or normative.get("id") or "unknown_rule")
    expected = _float(normative.get("expected_value"), 1.0)
    actual = _float(normative.get("actual_value"), 0.0)
    precision = _float(normative.get("precision"), 1.0)
    surprise_score = round(abs(actual - expected) * precision, 4)
    trace = ExpectationTrace(
        domain=ExpectationDomain.CONSTITUTIONAL,
        state_kind=ExpectationStateKind.NORMATIVE,
        expected_value=expected,
        actual_value=actual,
        surprise_score=surprise_score,
        precision=precision,
        valence="negative",
        stake=_float(normative.get("stake"), 1.0),
        source_event_id=f"normative:{rule_id}:{ctx.tick_id}",
        lifecycle_state=ExpectationLifecycleState.VIOLATED,
    )
    ctx.surprise.setdefault("constitutional", {})[rule_id] = _trace_payload(trace)
    ctx.action_bias.setdefault("normative", {})[rule_id] = "governance_trigger"
    ctx.drive.setdefault("constitutional", {})[rule_id] = {
        "drive_score": surprise_score,
        "actionability": 1.0,
        "action_bias": "governance_trigger",
        "constitution_verdict": "blocked: normative state requires governance",
    }
    ctx.expectation_updates.append(
        ExpectationUpdate(
            target="normative_state",
            parameter_name=rule_id,
            old_value="governance_owned",
            new_value="governance_triggered",
            rule=ExpectationUpdateRule.GOVERNANCE_TRIGGER,
            reason_event=trace.source_event_id,
            update_params={"domain": "constitutional", "rule_id": rule_id},
            constitution_verdict="blocked: normative state updates require governance",
            local_update_blocked=True,
        )
    )
    return True


def _apply_governed_revision(ctx: TickContext) -> bool:
    revisions = _governed_revision_entries(ctx.briefing)
    emitted = False
    for revision in revisions:
        if not _revision_is_approved(revision):
            continue
        rule_id = str(revision.get("rule_id") or revision.get("id") or "unknown_rule")
        new_value = revision.get("new_value")
        if new_value is None:
            new_value = revision.get("value")
        if new_value is None:
            continue
        old_value = revision.get("old_value", "governance_owned")
        revision_id = str(
            revision.get("revision_id")
            or revision.get("decision_id")
            or revision.get("proposal_id")
            or f"governed_revision:{rule_id}:{ctx.tick_id}"
        )
        authority = str(revision.get("authority") or revision.get("source") or "governance")
        ctx.expectations.setdefault("normative_revision", {})[rule_id] = {
            "state_kind": ExpectationStateKind.NORMATIVE.value,
            "lifecycle_state": ExpectationLifecycleState.REVISED.value,
            "revision_id": revision_id,
            "authority": authority,
            "source": revision.get("source", "governed_revision_read_model"),
        }
        ctx.action_bias.setdefault("governance", {})[rule_id] = "apply_governed_revision"
        ctx.drive.setdefault("governance", {})[rule_id] = {
            "drive_score": 0.0,
            "actionability": 1.0,
            "action_bias": "apply_governed_revision",
            "constitution_verdict": "approved: normative state revision is governance-owned",
        }
        ctx.expectation_updates.append(
            ExpectationUpdate(
                target="normative_state",
                parameter_name=rule_id,
                old_value=old_value,
                new_value=new_value,
                rule=ExpectationUpdateRule.GOVERNED_REVISION,
                reason_event=revision_id,
                update_params={
                    "domain": "constitutional",
                    "rule_id": rule_id,
                    "authority": authority,
                    "status": revision.get("status", "approved"),
                },
                constitution_verdict="approved: governed revision read model",
                local_update_blocked=False,
            )
        )
        emitted = True
    return emitted


def _apply_desired_slow_drift(ctx: TickContext, *, iem_state: dict[str, Any]) -> None:
    negative_pressure = _has_negative_pressure(ctx)
    prior_streak = int(_vector_value(iem_state, "drift_parameters", "negative_pressure_streak", 0.0))
    next_streak = min(prior_streak + 1, 12) if negative_pressure else 0
    ctx.expectation_updates.append(
        ExpectationUpdate(
            target="identity_drift_parameters",
            parameter_name="negative_pressure_streak",
            old_value=float(prior_streak),
            new_value=float(next_streak),
            rule=ExpectationUpdateRule.DECAY,
            reason_event=f"h0:drift:{ctx.tick_id}",
            update_params={"negative_pressure": negative_pressure},
            constitution_verdict="allowed: drift counter update",
        )
    )
    if not negative_pressure or prior_streak < 2:
        return

    current = _vector_value(iem_state, "desire_vector", "risk_aversion", 0.50)
    learning_rate = _learning_rate(iem_state, "desired_learning_rate", 0.02)
    next_value = _clamp(current + learning_rate)
    ctx.expectation_updates.append(
        ExpectationUpdate(
            target="identity_desire_vector",
            parameter_name="risk_aversion",
            old_value=round(current, 4),
            new_value=next_value,
            rule=ExpectationUpdateRule.SLOW_TRAIT_DRIFT,
            reason_event=f"h0:repeated_negative_pressure:{ctx.tick_id}",
            update_params={
                "prior_negative_pressure_streak": prior_streak,
                "learning_rate": learning_rate,
            },
            constitution_verdict="allowed: desired state slow drift",
        )
    )


def _survival_actual(briefing: dict[str, Any], energy_state: EnergyState) -> float:
    identity = briefing.get("identity")
    if isinstance(identity, dict):
        if bool(identity.get("at_risk")):
            return 0.25
        state = str(identity.get("state") or "").upper()
        if state == "ACTIVE":
            return 0.95
        if state == "PROVISIONAL":
            remaining = _float(identity.get("remaining_epochs"), 2.0)
            return _clamp(0.45 + min(remaining, 5.0) * 0.08)
    if energy_state.balance <= 0 and energy_state.staked <= 0:
        return 0.0
    if energy_state.balance < 5.0:
        return 0.35
    return 0.90


def _survival_stake(briefing: dict[str, Any], energy_state: EnergyState) -> float:
    identity = briefing.get("identity")
    if isinstance(identity, dict) and bool(identity.get("at_risk")):
        return 1.0
    if energy_state.balance < 5.0:
        return 0.9
    return 0.6


def _economic_actual(energy_state: EnergyState) -> float:
    cap = max(float(energy_state.balance_cap or 0.0), 1.0)
    return _clamp(float(energy_state.balance) / cap)


def _reputation_actual(briefing: dict[str, Any], energy_state: EnergyState) -> float:
    context = briefing.get("reputation_context")
    if not isinstance(context, dict):
        context = briefing.get("reputation")
    if isinstance(context, dict):
        for key in ("actual_score", "score", "current_score", "trust_score"):
            if key in context:
                return _clamp(_float(context.get(key), 0.5))
        recent_failures = _float(context.get("recent_failures"), 0.0)
        unresolved_challenges = _float(context.get("unresolved_challenges"), 0.0)
        repair_signal = _float(context.get("repair_signal"), 0.0)
        if recent_failures or unresolved_challenges or repair_signal:
            return _clamp(
                0.65 - 0.20 * recent_failures - 0.15 * unresolved_challenges + 0.10 * repair_signal
            )
    return _clamp(float(getattr(energy_state, "reputation", 0.5)))


def _reputation_stake(briefing: dict[str, Any]) -> float:
    context = briefing.get("reputation_context")
    if isinstance(context, dict):
        return _clamp(_float(context.get("stake"), 0.8))
    return 0.7


def _task_actual(briefing: dict[str, Any]) -> float:
    context = briefing.get("task_context")
    if not isinstance(context, dict):
        context = briefing.get("task")
    if isinstance(context, dict):
        for key in (
            "actual_success_probability",
            "success_probability",
            "completion_probability",
            "quality_score",
        ):
            if key in context:
                return _clamp(_float(context.get(key), 0.5))
        if bool(context.get("failed")):
            return 0.10
        if bool(context.get("blocked")):
            return 0.25
        if bool(context.get("delivered")) or bool(context.get("completed")):
            return 0.90
        overdue = _float(context.get("overdue"), 0.0)
        missing_inputs = _float(context.get("missing_inputs"), 0.0)
        if overdue or missing_inputs:
            return _clamp(0.70 - 0.25 * overdue - 0.15 * missing_inputs)
    return 0.75


def _task_stake(briefing: dict[str, Any]) -> float:
    context = briefing.get("task_context")
    if isinstance(context, dict):
        return _clamp(_float(context.get("stake"), 0.8))
    return 0.7


def _governance_actual(briefing: dict[str, Any]) -> float:
    context = briefing.get("governance_context")
    if not isinstance(context, dict):
        context = briefing.get("governance")
    if isinstance(context, dict):
        for key in ("actual_participation", "participation", "participation_rate", "responsiveness"):
            if key in context:
                return _clamp(_float(context.get(key), 0.5))
        pending_votes = _float(context.get("pending_votes"), 0.0)
        missed_reviews = _float(context.get("missed_reviews"), 0.0)
        if pending_votes or missed_reviews:
            return _clamp(0.65 - 0.15 * pending_votes - 0.20 * missed_reviews)
    return 0.60


def _governance_stake(briefing: dict[str, Any]) -> float:
    context = briefing.get("governance_context")
    if isinstance(context, dict):
        return _clamp(_float(context.get("stake"), 0.9))
    return 0.8


def _has_negative_pressure(ctx: TickContext) -> bool:
    for domain in ("survival", "economic", "reputation", "task", "governance", "constitutional"):
        payload = ctx.surprise.get(domain)
        if not isinstance(payload, dict):
            continue
        for item in payload.values():
            if isinstance(item, dict) and item.get("valence") == "negative":
                if _float(item.get("surprise_score"), 0.0) > 0.05:
                    return True
    return False


def _governed_revision_entries(briefing: dict[str, Any]) -> list[dict[str, Any]]:
    raw = briefing.get("governed_revision_context")
    if raw is None:
        raw = briefing.get("normative_revision_context")
    if raw is None:
        raw = briefing.get("governed_revisions")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _revision_is_approved(revision: dict[str, Any]) -> bool:
    status = str(revision.get("status") or "").strip().lower()
    approved = bool(revision.get("approved")) or status in {"approved", "ratified", "enacted"}
    authority = str(revision.get("authority") or revision.get("source") or "").strip().lower()
    governed_authority = any(
        token in authority
        for token in ("governance", "arbitration", "constitution", "council")
    )
    return approved and governed_authority


def _entry_rule(entry: dict[str, Any]) -> str:
    rule = entry.get("rule")
    return str(getattr(rule, "value", rule) or "")


def _vector_value(state: dict[str, Any], section: str, parameter: str, default: float) -> float:
    values = state.get(section)
    if isinstance(values, dict):
        return _float(values.get(parameter), default)
    return default


def _learning_rate(state: dict[str, Any], parameter: str, default: float) -> float:
    value = _vector_value(state, "drift_parameters", parameter, default)
    return _clamp(value, 0.0, 1.0)


def _next_precision(precision: float, delta: float) -> float:
    if abs(delta) <= 0.05:
        return _clamp(precision + 0.03, 0.05, 0.95)
    if abs(delta) >= 0.25:
        return _clamp(precision - 0.05, 0.05, 0.95)
    return _clamp(precision + 0.01, 0.05, 0.95)


def _trace_payload(trace: ExpectationTrace) -> dict[str, Any]:
    payload = asdict(trace)
    payload["domain"] = trace.domain.value
    payload["state_kind"] = trace.state_kind.value
    payload["lifecycle_state"] = trace.lifecycle_state.value
    return payload


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(max(low, min(high, value)), 4)