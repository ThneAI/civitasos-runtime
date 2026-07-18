from pathlib import Path

from civitasos_runtime.checkpoint_adapters import (
    backend_projection_snapshots,
    capture_runtime_snapshots,
    identity_snapshot,
    restore_runtime_snapshots,
)
from civitasos_runtime.memory import LocalMemory


IDENTITY_ID = "did:civ:testnet:checkpoint-agent"


def _backend_projection(public_key_hex: str) -> dict:
    import hashlib

    epoch = 7
    return {
        "schema_version": "civitasos-backend-sustainable-identity-checkpoint:v1",
        "identity_id": IDENTITY_ID,
        "barrier_epoch": epoch,
        "identity": {
            "schema_version": "civitasos-backend-identity-projection:v1",
            "identity_id": IDENTITY_ID,
            "barrier_epoch": epoch,
            "card": {"did": IDENTITY_ID, "capabilities": []},
            "institutional_identity": {"state": "CIVITAS_IDENTITY"},
        },
        "credential": {
            "schema_version": "civitasos-backend-credential-projection:v1",
            "identity_id": IDENTITY_ID,
            "barrier_epoch": epoch,
            "public_key_fingerprint": hashlib.sha256(
                bytes.fromhex(public_key_hex)
            ).hexdigest(),
            "registered_at": 1,
            "version": 3,
        },
        "r2r": {
            "schema_version": "civitasos-backend-r2r-projection:v1",
            "identity_id": IDENTITY_ID,
            "barrier_epoch": epoch,
            "social_graph": {"agent_id": IDENTITY_ID, "relations": []},
        },
        "economy": {
            "schema_version": "civitasos-backend-economy-projection:v1",
            "identity_id": IDENTITY_ID,
            "barrier_epoch": epoch,
            "account": {"id": IDENTITY_ID, "balance": 10},
        },
    }


def test_backend_projection_binds_four_domains_to_signer_public_key() -> None:
    public_key_hex = "11" * 32
    snapshots = backend_projection_snapshots(
        _backend_projection(public_key_hex),
        identity_id=IDENTITY_ID,
        public_key_hex=public_key_hex,
        signer_ref="pkcs11:token=civitas;id=01",
        source_ref="backend:node-1:epoch-7",
    )

    assert [snapshot.domain for snapshot in snapshots] == [
        "identity",
        "credential",
        "r2r",
        "economy",
    ]
    assert {snapshot.revision for snapshot in snapshots[1:]} == {
        "backend-epoch:7:credential",
        "backend-epoch:7:r2r",
        "backend-epoch:7:economy",
    }
    assert snapshots[0].payload["private_material_exported"] is False


def test_backend_projection_rejects_cross_epoch_or_key_binding() -> None:
    import pytest

    public_key_hex = "22" * 32
    projection = _backend_projection(public_key_hex)
    projection["r2r"]["barrier_epoch"] = 8
    with pytest.raises(ValueError, match="r2r barrier epoch mismatch"):
        backend_projection_snapshots(
            projection,
            identity_id=IDENTITY_ID,
            public_key_hex=public_key_hex,
            signer_ref="signer:test",
            source_ref="backend:test",
        )

    projection = _backend_projection(public_key_hex)
    projection["credential"]["public_key_fingerprint"] = "00" * 32
    with pytest.raises(ValueError, match="does not bind"):
        backend_projection_snapshots(
            projection,
            identity_id=IDENTITY_ID,
            public_key_hex=public_key_hex,
            signer_ref="signer:test",
            source_ref="backend:test",
        )


def test_local_memory_snapshot_restore_is_atomic(tmp_path: Path) -> None:
    memory = LocalMemory(tmp_path / "memory")
    memory.put("identity_iem_state", {"version": 1})
    memory.put("relation_expectation:peer", {"trust": 0.7})
    captured = memory.snapshot()

    memory.put("identity_iem_state", {"version": 2})
    memory.put("temporary", True)
    memory.restore_snapshot(captured)

    assert memory.snapshot() == captured
    assert memory.get("temporary") is None
    memory.close()


def test_runtime_adapter_partitions_and_restores_one_sqlite_snapshot(
    tmp_path: Path,
) -> None:
    memory = LocalMemory(tmp_path / "memory")
    memory.put("identity_iem_state", {"identity_id": IDENTITY_ID})
    memory.put("identity_iem_anchor", {"version_id": "iem:v1:abc"})
    memory.put("expectation_update_log", [{"update": 1}])
    memory.put("lessons_learned", [{"lesson": "retain"}])

    iem, local = capture_runtime_snapshots(
        memory,
        identity_id=IDENTITY_ID,
        source_ref="sqlite://memory.db",
    )
    memory.put("lessons_learned", [{"lesson": "changed"}])
    restore_runtime_snapshots(
        memory,
        identity_id=IDENTITY_ID,
        iem=iem,
        memory_snapshot=local,
    )

    assert memory.get("lessons_learned") == [{"lesson": "retain"}]
    assert memory.get("identity_iem_state") == {"identity_id": IDENTITY_ID}
    memory.close()


def test_identity_adapter_never_exports_private_material() -> None:
    snapshot = identity_snapshot(
        identity_id=IDENTITY_ID,
        public_key_hex="ab" * 32,
        signer_ref="pkcs11:token=civitas;id=01",
    )

    assert snapshot.payload["private_material_exported"] is False
    assert set(snapshot.payload) == {
        "identity_id",
        "public_key_hex",
        "signer_ref",
        "private_material_exported",
    }
