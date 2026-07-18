from pathlib import Path

from civitasos_runtime.checkpoint_adapters import (
    capture_runtime_snapshots,
    identity_snapshot,
    restore_runtime_snapshots,
)
from civitasos_runtime.memory import LocalMemory


IDENTITY_ID = "did:civ:testnet:checkpoint-agent"


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
