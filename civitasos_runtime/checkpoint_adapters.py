"""Adapters between Runtime-owned state and atomic checkpoint domains."""

from __future__ import annotations

import hashlib
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


def backend_projection_snapshots(
    projection: dict[str, Any],
    *,
    identity_id: str,
    public_key_hex: str,
    signer_ref: str,
    source_ref: str,
) -> tuple[DomainSnapshot, DomainSnapshot, DomainSnapshot, DomainSnapshot]:
    """Bind one backend barrier projection to the Runtime signer identity."""
    if projection.get("schema_version") != (
        "civitasos-backend-sustainable-identity-checkpoint:v1"
    ):
        raise ValueError("unsupported backend checkpoint projection schema")
    if projection.get("identity_id") != identity_id:
        raise ValueError("backend checkpoint identity binding mismatch")
    node_id = projection.get("node_id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("backend checkpoint node identity is missing")
    epoch = projection.get("barrier_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("backend checkpoint barrier epoch is invalid")

    components: dict[str, dict[str, Any]] = {}
    for domain in ("identity", "credential", "r2r", "economy"):
        value = projection.get(domain)
        if not isinstance(value, dict):
            raise ValueError(f"backend checkpoint is missing {domain} projection")
        if value.get("identity_id") != identity_id:
            raise ValueError(f"backend {domain} identity binding mismatch")
        if value.get("barrier_epoch") != epoch:
            raise ValueError(f"backend {domain} barrier epoch mismatch")
        components[domain] = value

    identity_component = components["identity"]
    card = identity_component.get("card")
    if not isinstance(card, dict) or card.get("did") != identity_id:
        raise ValueError("backend identity card binding mismatch")
    social_graph = components["r2r"].get("social_graph")
    if not isinstance(social_graph, dict) or social_graph.get("agent_id") != identity_id:
        raise ValueError("backend R2R social graph binding mismatch")
    account = components["economy"].get("account")
    if not isinstance(account, dict) or account.get("id") != identity_id:
        raise ValueError("backend economy account binding mismatch")

    base_identity = identity_snapshot(
        identity_id=identity_id,
        public_key_hex=public_key_hex,
        signer_ref=signer_ref,
    )
    expected_fingerprint = hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()
    if components["credential"].get("public_key_fingerprint") != expected_fingerprint:
        raise ValueError("backend credential does not bind the checkpoint signer public key")

    identity_payload = {
        **base_identity.payload,
        "backend_identity": identity_component,
    }
    revision_prefix = f"backend-epoch:{epoch}"
    return (
        _snapshot("identity", identity_id, source_ref, identity_payload),
        DomainSnapshot.create(
            domain="credential",
            identity_id=identity_id,
            revision=f"{revision_prefix}:credential",
            source_ref=source_ref,
            payload=components["credential"],
        ),
        DomainSnapshot.create(
            domain="r2r",
            identity_id=identity_id,
            revision=f"{revision_prefix}:r2r",
            source_ref=source_ref,
            payload=components["r2r"],
        ),
        DomainSnapshot.create(
            domain="economy",
            identity_id=identity_id,
            revision=f"{revision_prefix}:economy",
            source_ref=source_ref,
            payload=components["economy"],
        ),
    )


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
