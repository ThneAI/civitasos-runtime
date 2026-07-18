import json
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from civitasos_runtime.atomic_checkpoint import AtomicCheckpointStore
from civitasos_runtime.checkpoint_capture import (
    AtomicCheckpointCapture,
    BackendCheckpointClient,
)
from civitasos_runtime.checkpoint_files import read_private_json
from civitasos_runtime.checkpoint_restore import (
    AtomicCheckpointRestore,
    CheckpointRestoreIncomplete,
)
from civitasos_runtime.memory import LocalMemory

from test_checkpoint_adapters import IDENTITY_ID, _backend_projection
from test_checkpoint_capture import _Signer, _memory


class _BackendTransport:
    def __init__(
        self,
        signer: _Signer,
        *,
        fail_activations: int = 0,
        lose_activation_responses: int = 0,
    ) -> None:
        self.signer = signer
        self.projection = _backend_projection(signer.public_key_hex)
        self.fail_activations = fail_activations
        self.lose_activation_responses = lose_activation_responses
        self.requests = []
        self.record = None

    def __call__(self, request, timeout):  # noqa: ANN001
        body = json.loads(request.data) if request.data is not None else None
        self.requests.append(
            {
                "method": request.method,
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "timeout": timeout,
                "body": body,
            }
        )
        if request.method == "GET":
            return self.projection
        if request.full_url.endswith("/restore/preflight"):
            if self.record is None:
                self.record = self._record(body, "validated")
            return dict(self.record)
        if request.full_url.endswith("/restore/activate"):
            if self.fail_activations > 0:
                self.fail_activations -= 1
                raise RuntimeError("injected activation failure")
            assert self.record is not None
            self.record = {
                **self.record,
                "status": "activated",
                "activation_fact_id": "fact:checkpoint-activation",
            }
            if self.lose_activation_responses > 0:
                self.lose_activation_responses -= 1
                raise RuntimeError("injected lost activation response")
            return dict(self.record)
        raise AssertionError(f"unexpected request: {request.full_url}")

    @staticmethod
    def _record(request, status):  # noqa: ANN001
        return {
            "schema_version": "civitasos-checkpoint-restore-journal:v1",
            "checkpoint_id": request["checkpoint_id"],
            "manifest_hash": request["manifest_hash"],
            "sequence": request["sequence"],
            "identity_id": IDENTITY_ID,
            "node_id": request["node_id"],
            "status": status,
            "validated_epoch": 7,
            "materialization": request["materialization"],
            "updated_at": 1,
            "activation_fact_id": None,
        }


def _active_checkpoint(tmp_path: Path, transport: _BackendTransport):
    store = AtomicCheckpointStore(tmp_path / "checkpoints")
    memory = _memory(tmp_path / "memory")
    client = BackendCheckpointClient(
        "https://backend.internal:8443",
        "owner-jwt",
        transport=transport,
    )

    @contextmanager
    def paused():
        yield

    manifest = AtomicCheckpointCapture(store, client, paused).prepare(
        identity_id=IDENTITY_ID,
        memory=memory,
        signer=transport.signer,
        signer_ref="pkcs11:token=civitas;id=01",
    )
    store.commit(manifest["checkpoint_id"])
    return store, memory, client, manifest

def test_restore_latest_is_atomic_authenticated_and_private(tmp_path: Path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer)
    store, memory, client, manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])
    memory.put("temporary", True)
    pauses = []

    @contextmanager
    def pause_runtime():
        pauses.append("entered")
        try:
            yield
        finally:
            pauses.append("exited")

    record = AtomicCheckpointRestore(store, client, pause_runtime).restore_latest(
        identity_id=IDENTITY_ID,
        memory=memory,
        signer=signer,
    )

    assert record["status"] == "activated"
    assert memory.snapshot() == expected
    assert pauses == ["entered", "exited"]
    posts = [request for request in transport.requests if request["method"] == "POST"]
    assert [request["url"].rsplit("/", 1)[-1] for request in posts] == [
        "preflight",
        "activate",
    ]
    assert all(request["authorization"] == "Bearer owner-jwt" for request in posts)
    assert posts[0]["body"]["manifest_hash"] == manifest["manifest_hash"]
    journal_path = store.root / "restore_journal" / f"{manifest['checkpoint_id']}.json"
    assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
    assert "lessons" not in read_private_json(journal_path)
    memory.close()


def test_activation_failure_replays_after_local_materialization(tmp_path: Path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer, fail_activations=1)
    store, memory, client, manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])

    @contextmanager
    def paused():
        yield

    restore = AtomicCheckpointRestore(store, client, paused)
    with pytest.raises(CheckpointRestoreIncomplete, match="replay before resuming"):
        restore.restore_latest(identity_id=IDENTITY_ID, memory=memory, signer=signer)

    journal_path = store.root / "restore_journal" / f"{manifest['checkpoint_id']}.json"
    assert read_private_json(journal_path)["status"] == "runtime_applied"
    assert memory.snapshot() == expected

    completed = restore.restore_latest(
        identity_id=IDENTITY_ID,
        memory=memory,
        signer=signer,
    )
    assert completed["status"] == "activated"
    assert memory.snapshot() == expected
    memory.close()


def test_lost_activation_response_replays_already_activated_backend(tmp_path: Path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer, lose_activation_responses=1)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])

    @contextmanager
    def paused():
        yield

    restore = AtomicCheckpointRestore(store, client, paused)
    with pytest.raises(CheckpointRestoreIncomplete):
        restore.restore_latest(identity_id=IDENTITY_ID, memory=memory, signer=signer)
    assert transport.record["status"] == "activated"

    completed = restore.restore_latest(
        identity_id=IDENTITY_ID,
        memory=memory,
        signer=signer,
    )
    assert completed["status"] == "activated"
    assert memory.snapshot() == expected
    memory.close()


def test_restore_rejects_wrong_signer_before_mutating_memory(tmp_path: Path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    memory.put("lessons", ["changed"])
    before = memory.snapshot()

    @contextmanager
    def paused():
        yield

    with pytest.raises(ValueError, match="does not match"):
        AtomicCheckpointRestore(store, client, paused).restore_latest(
            identity_id=IDENTITY_ID,
            memory=memory,
            signer=_Signer(),
        )
    assert memory.snapshot() == before
    assert len([item for item in transport.requests if item["method"] == "POST"]) == 0
    memory.close()


def test_restore_rejects_broad_journal_permissions(tmp_path: Path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer, fail_activations=1)
    store, memory, client, manifest = _active_checkpoint(tmp_path, transport)

    @contextmanager
    def paused():
        yield

    restore = AtomicCheckpointRestore(store, client, paused)
    with pytest.raises(CheckpointRestoreIncomplete):
        restore.restore_latest(identity_id=IDENTITY_ID, memory=memory, signer=signer)
    journal_path = store.root / "restore_journal" / f"{manifest['checkpoint_id']}.json"
    journal_path.chmod(0o644)

    with pytest.raises(ValueError, match="permissions are too broad"):
        restore.restore_latest(identity_id=IDENTITY_ID, memory=memory, signer=signer)
    memory.close()
