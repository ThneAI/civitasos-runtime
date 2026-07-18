"""Adapters between Runtime-owned state and atomic checkpoint domains."""

from __future__ import annotations

from typing import Any

from .checkpoint_models import DomainSnapshot, sha256_payload
from .memory import LocalMemory


IEM_KEYS = frozenset(
    {"identity_iem_state", "identity_iem_anchor", "expectation_update_log"}
)


def capture_runtime_snapshots(
    memory: LocalMemory,
    *,
    identity_id: str,
    source_ref: str,
) -> tuple[DomainSnapshot, DomainSnapshot]:
    """Capture IEM and remaining memory from one SQLite read snapshot."""
    values = memory.snapshot()
    missing = IEM_KEYS - values.keys()
    if missing:
        raise ValueError(f"runtime checkpoint is missing IEM keys: {sorted(missing)}")
    iem_payload = {key: values[key] for key in sorted(IEM_KEYS)}
    memory_payload = {
        key: value for key, value in values.items() if key not in IEM_KEYS
    }
    return (
        _snapshot("iem", identity_id, source_ref, iem_payload),
        _snapshot("memory", identity_id, source_ref, memory_payload),
    )


def restore_runtime_snapshots(
    memory: LocalMemory,
    *,
    identity_id: str,
    iem: DomainSnapshot,
    memory_snapshot: DomainSnapshot,
) -> None:
    """Restore both Runtime domains in one SQLite transaction."""
    for expected, snapshot in (("iem", iem), ("memory", memory_snapshot)):
        snapshot.validate()
        if snapshot.domain != expected or snapshot.identity_id != identity_id:
            raise ValueError(f"invalid {expected} checkpoint binding")
    if set(iem.payload) != IEM_KEYS:
        raise ValueError("IEM checkpoint key set is incomplete")
    overlap = set(iem.payload) & set(memory_snapshot.payload)
    if overlap:
        raise ValueError(f"runtime checkpoint domains overlap: {sorted(overlap)}")
    memory.restore_snapshot({**memory_snapshot.payload, **iem.payload})


def identity_snapshot(
    *,
    identity_id: str,
    public_key_hex: str,
    signer_ref: str,
) -> DomainSnapshot:
    """Bind public identity metadata without exporting signer secret material."""
    if len(public_key_hex) != 64:
        raise ValueError("identity checkpoint requires a 32-byte public key")
    try:
        bytes.fromhex(public_key_hex)
    except ValueError as error:
        raise ValueError("identity checkpoint public key must be hex") from error
    payload = {
        "identity_id": identity_id,
        "public_key_hex": public_key_hex,
        "signer_ref": signer_ref,
        "private_material_exported": False,
    }
    return _snapshot("identity", identity_id, signer_ref, payload)


def _snapshot(
    domain: str,
    identity_id: str,
    source_ref: str,
    payload: dict[str, Any],
) -> DomainSnapshot:
    revision = f"{domain}:v1:{sha256_payload(payload).removeprefix('sha256:')[:16]}"
    return DomainSnapshot.create(
        domain=domain,
        identity_id=identity_id,
        revision=revision,
        source_ref=source_ref,
        payload=payload,
    )
