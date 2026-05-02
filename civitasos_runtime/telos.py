"""H.1 Telos alignment and verifier planning.

This module is deliberately pure and opt-in. It derives an auditable intent
stack from briefing/read-model state, but it does not grant permission and it
does not mutate normative state.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import IntentFrame, IntentLayer, TelosAlignmentTrace, VerificationPlan
from .telos_inputs import (
    extract_explicit_telos,
    find_verification_level,
    first_task,
    has_hint,
    list_from_mapping,
    looks_risky,
    mapping,
    task_list,
    task_text,
    unique_strings,
)

SCHEMA_VERSION = "h1_telos_alignment.v1"

_NORTH_STAR_TELOS = (
    "advance verifiable, accountable, evolvable human-sovereign AI social operation"
)
_VERIFICATION_HINTS = (
    "verify",
    "verifier",
    "validate",
    "check",
    "test",
    "audit",
    "analyze",
    "analyse",
    "compare",
    "diff",
    "review",
    "confirm",
)
_TASK_ACTIONS = frozenset({"pool_claim", "pool_get_task", "task_execute"})
_RELATION_ACTION_HINTS = ("r2r", "relation", "reputation", "repair", "trust")
_LONG_ACTION_HINTS = ("governance", "proposal", "constitution", "evolve")


def build_telos_alignment(
    briefing: dict[str, Any],
    memories: dict[str, Any] | None = None,
    action_bias: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a five-layer H.1 intent stack from current read models."""
    del memories
    task = first_task(briefing)
    task_id = task_text(task, "task_id", "id") or str(briefing.get("task_id") or "")
    explicit_telos, telos_source = extract_explicit_telos(briefing, task)
    verifier_plan = _build_verification_plan(briefing, task, action_bias or {})

    if task_id:
        immediate = f"deliver or safely progress task {task_id}"
    elif briefing.get("briefing"):
        immediate = "interpret the current briefing without inventing success evidence"
    else:
        immediate = "observe current state and avoid empty action"

    short = explicit_telos or "preserve result quality and verification evidence"
    mid = "preserve R2R trust, relation accountability, and repair obligations"
    long = "increase identity reliability without local normative mutation"
    telos = _NORTH_STAR_TELOS
    active_layer = IntentLayer.SHORT if verifier_plan.required else (
        IntentLayer.IMMEDIATE if task_id or briefing.get("briefing") else IntentLayer.LONG
    )

    trace = TelosAlignmentTrace(
        schema_version=SCHEMA_VERSION,
        active_layer=active_layer,
        intent_stack=[
            IntentFrame(IntentLayer.IMMEDIATE, immediate, "briefing.task", 0.85),
            IntentFrame(IntentLayer.SHORT, short, telos_source or "quality_guard", 0.80),
            IntentFrame(IntentLayer.MID, mid, "r2r_expectation", 0.70),
            IntentFrame(IntentLayer.LONG, long, "identity_iem", 0.70),
            IntentFrame(IntentLayer.TELOS, telos, "vmv", 1.0),
        ],
        verification_plan=verifier_plan,
        source_task_id=task_id,
        telos_source=telos_source,
    )
    return _trace_to_dict(trace)


def served_intent_layer_for_action(
    action: str,
    params: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
) -> str:
    """Map a decision action to the intent layer it primarily serves."""
    action_l = (action or "").strip().lower()
    if not action_l or not alignment:
        return ""
    verifier = mapping(alignment.get("verification_plan"))
    verifier_tools = {str(tool).strip().lower() for tool in verifier.get("tools") or []}
    if action_l in verifier_tools or has_hint(action_l, _VERIFICATION_HINTS):
        return IntentLayer.SHORT.value
    if action_l == "wait":
        active = str(alignment.get("active_layer") or "")
        if verifier.get("required"):
            return IntentLayer.SHORT.value
        return active if active in {IntentLayer.LONG.value, IntentLayer.TELOS.value} else ""
    if action_l in _TASK_ACTIONS:
        return IntentLayer.IMMEDIATE.value
    if has_hint(action_l, _RELATION_ACTION_HINTS):
        return IntentLayer.MID.value
    if has_hint(action_l, _LONG_ACTION_HINTS):
        return IntentLayer.LONG.value
    if params and has_hint(" ".join(str(k) for k in params.keys()), _VERIFICATION_HINTS):
        return IntentLayer.SHORT.value
    active = str(alignment.get("active_layer") or "")
    return active if active in {layer.value for layer in IntentLayer} else IntentLayer.IMMEDIATE.value


def _build_verification_plan(
    briefing: dict[str, Any],
    task: dict[str, Any],
    action_bias: dict[str, Any],
) -> VerificationPlan:
    tools = unique_strings(
        task_list(task, "verifier_tools")
        + list_from_mapping(briefing, "verifier_tools")
        + list_from_mapping(briefing, "required_verifier_tools")
    )
    reasons: list[str] = []
    relation_level = find_verification_level(action_bias) or find_verification_level(briefing)
    level = relation_level or "baseline"
    if tools:
        reasons.append("task declares verifier tools")
        if level == "baseline":
            level = "elevated"
    if relation_level in {"elevated", "strict"}:
        reasons.append(f"relation expectation requested {relation_level} verification")
    if looks_risky(briefing, task):
        reasons.append("briefing contains high-risk execution hints")
        if level != "strict":
            level = "elevated"
    required = bool(tools or relation_level in {"elevated", "strict"} or reasons)
    return VerificationPlan(required=required, level=level, tools=tools, reasons=reasons)


def _trace_to_dict(trace: TelosAlignmentTrace) -> dict[str, Any]:
    payload = asdict(trace)
    payload["active_layer"] = trace.active_layer.value
    payload["intent_stack"] = [
        {**asdict(frame), "layer": frame.layer.value}
        for frame in trace.intent_stack
    ]
    return payload
