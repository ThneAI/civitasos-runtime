"""H.0-C Identity Expectation Model version anchors.

The IEM anchor is intentionally small: DID/Identity surfaces can carry it
without embedding the full expectation vectors or relation matrix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from .models import IEMVersionAnchor

IEM_SCHEMA_VERSION = "iem:v1"


def genesis_iem_state(identity_id: str) -> dict[str, Any]:
    """Return the empty, replayable IEM state for a new identity."""
    identity = str(identity_id or "unknown")
    return {
        "schema_version": IEM_SCHEMA_VERSION,
        "identity_id": identity,
        "expectation_vector": {},
        "precision_vector": {},
        "desire_vector": {},
        "domain_weight_matrix": {},
        "drift_parameters": {},
        "relation_expectation_matrix": {},
    }


def iem_state_hash(state: Any) -> str:
    """Stable sha256 hash for an IEM state payload."""
    return _hash_payload(state)


def iem_update_log_hash(update_log: Any) -> str:
    """Stable sha256 hash for an expectation update log payload."""
    return _hash_payload(update_log if update_log is not None else [])


def build_iem_anchor(
    *,
    identity_id: str,
    state: Any,
    update_log: Any = None,
) -> IEMVersionAnchor:
    """Build a deterministic version anchor for the latest IEM state."""
    state_hash = iem_state_hash(state)
    hash_body = state_hash.removeprefix("sha256:")
    identity = str(identity_id or "unknown")
    return IEMVersionAnchor(
        version_id=f"{IEM_SCHEMA_VERSION}:{hash_body[:12]}",
        state_hash=state_hash,
        latest_update_log_hash=iem_update_log_hash(update_log),
        storage_hint=f"civitasos://identity/{identity}/iem/latest",
    )


def normalize_iem_payload(value: Any) -> Any:
    """Convert dataclasses/enums into a stable JSON-compatible shape."""
    if is_dataclass(value):
        return normalize_iem_payload(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): normalize_iem_payload(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_iem_payload(item) for item in value]
    return value


def _hash_payload(value: Any) -> str:
    normalized = normalize_iem_payload(value)
    blob = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"