import json
import os
from pathlib import Path

import pytest

from civitasos_runtime.atomic_checkpoint import AtomicCheckpointStore
from civitasos_runtime.checkpoint_models import REQUIRED_DOMAINS, DomainSnapshot


IDENTITY_ID = "did:civ:testnet:checkpoint-agent"


def snapshots(revision: str = "v1") -> list[DomainSnapshot]:
    return [
        DomainSnapshot.create(
            domain=domain,
            identity_id=IDENTITY_ID,
            revision=revision,
            source_ref=f"test://{domain}/{revision}",
            payload={"domain": domain, "value": f"{domain}-{revision}"},
        )
        for domain in sorted(REQUIRED_DOMAINS)
    ]


def test_prepare_is_invisible_until_commit(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    prepared = store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots())

    assert store.load_latest() is None
    committed = store.commit(prepared["checkpoint_id"])

    assert committed == prepared
    assert store.load_latest() == prepared
    assert AtomicCheckpointStore.snapshots(prepared).keys() == REQUIRED_DOMAINS


def test_checkpoint_requires_exact_domain_set_and_identity(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    with pytest.raises(ValueError, match="six identity domains"):
        store.prepare(
            identity_id=IDENTITY_ID,
            sequence=1,
            snapshots=snapshots()[:-1],
        )

    mixed = snapshots()
    mixed[-1] = DomainSnapshot.create(
        domain=mixed[-1].domain,
        identity_id="did:civ:testnet:other",
        revision="v1",
        source_ref="test://mixed",
        payload={"value": "mixed"},
    )
    with pytest.raises(ValueError, match="mix identity"):
        store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=mixed)


def test_checkpoint_rejects_secret_material() -> None:
    with pytest.raises(ValueError, match="forbidden secret field"):
        DomainSnapshot.create(
            domain="identity",
            identity_id=IDENTITY_ID,
            revision="v1",
            source_ref="test://identity",
            payload={"seed_hex": "not-allowed"},
        )

    allowed = DomainSnapshot.create(
        domain="memory",
        identity_id=IDENTITY_ID,
        revision="v1",
        source_ref="test://memory",
        payload={"opinion": "benign memory field"},
    )
    assert allowed.payload["opinion"] == "benign memory field"


def test_stale_or_forked_checkpoint_cannot_replace_latest(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    first = store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots())
    store.commit(first["checkpoint_id"])

    with pytest.raises(ValueError, match="sequence must be 2"):
        store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots("replay"))
    with pytest.raises(ValueError, match="previous id"):
        store.prepare(
            identity_id=IDENTITY_ID,
            sequence=2,
            previous_checkpoint_id="aic:v1:000000000000000000000000",
            snapshots=snapshots("fork"),
        )


def test_orphan_manifest_does_not_activate_after_interrupted_commit(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    first = store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots())
    store.commit(first["checkpoint_id"])
    second = store.prepare(
        identity_id=IDENTITY_ID,
        sequence=2,
        previous_checkpoint_id=first["checkpoint_id"],
        snapshots=snapshots("v2"),
    )
    source = store.prepared / f"{second['checkpoint_id']}.json"
    orphan = store.committed / f"{second['checkpoint_id']}.json"
    orphan.write_bytes(source.read_bytes())
    os.chmod(orphan, 0o600)

    assert store.load_latest()["checkpoint_id"] == first["checkpoint_id"]
    store.commit(second["checkpoint_id"])
    assert store.load_latest()["checkpoint_id"] == second["checkpoint_id"]


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    manifest = store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots())
    store.commit(manifest["checkpoint_id"])
    path = store.committed / f"{manifest['checkpoint_id']}.json"
    changed = json.loads(path.read_text())
    changed["domains"][0]["payload"]["value"] = "tampered"
    path.write_text(json.dumps(changed))
    os.chmod(path, 0o600)

    with pytest.raises(ValueError, match="payload hash mismatch"):
        store.load_latest()


def test_commit_marker_tampering_fails_closed(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path)
    manifest = store.prepare(identity_id=IDENTITY_ID, sequence=1, snapshots=snapshots())
    store.commit(manifest["checkpoint_id"])
    marker = json.loads(store.latest_path.read_text())
    marker["sequence"] = 2
    store.latest_path.write_text(json.dumps(marker))
    os.chmod(store.latest_path, 0o600)

    with pytest.raises(ValueError, match="marker sequence mismatch"):
        store.load_latest()


def test_checkpoint_root_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="directory must be real"):
        AtomicCheckpointStore(linked)
