"""Non-secret milestone observation for checkpoint fault gates."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Mapping


class CheckpointRestoreMilestone(str, Enum):
    """Durable restore boundaries visible to an external fault controller."""

    RUNTIME_INTENT_DURABLE = "runtime_intent_durable"
    BACKEND_PREFLIGHT_DURABLE = "backend_preflight_durable"
    RUNTIME_RESTORE_JOURNAL_DURABLE = "runtime_restore_journal_durable"
    RUNTIME_SQLITE_COMMITTED = "runtime_sqlite_committed"
    RUNTIME_APPLIED_JOURNAL_DURABLE = "runtime_applied_journal_durable"
    BACKEND_ACTIVATION_DURABLE = "backend_activation_durable"
    RUNTIME_ACTIVATION_JOURNAL_DURABLE = "runtime_activation_journal_durable"
    RUNTIME_INTENT_ACTIVATED_DURABLE = "runtime_intent_activated_durable"
    RUNTIME_TICK_LATCH_RELEASED = "runtime_tick_latch_released"


CheckpointRestoreObserver = Callable[[str, dict[str, Any]], None]

_PUBLIC_BINDING_FIELDS = (
    "checkpoint_id",
    "manifest_hash",
    "sequence",
    "identity_id",
    "node_id",
    "status",
    "backend_status",
    "activation_fact_id",
)


def emit_checkpoint_restore_milestone(
    observer: CheckpointRestoreObserver | None,
    milestone: CheckpointRestoreMilestone,
    record: Mapping[str, Any],
) -> None:
    """Emit a field-whitelisted event after a durable boundary is crossed."""
    if observer is None:
        return
    event = {
        "schema_version": "civitasos-checkpoint-restore-milestone:v1",
        "milestone": milestone.value,
    }
    event.update(
        {
            field: record[field]
            for field in _PUBLIC_BINDING_FIELDS
            if field in record
        }
    )
    observer(milestone.value, event)
