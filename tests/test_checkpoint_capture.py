from contextlib import contextmanager
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from civitasos_runtime.atomic_checkpoint import AtomicCheckpointStore
from civitasos_runtime.checkpoint_capture import (
    AtomicCheckpointCapture,
    BackendCheckpointClient,
    verify_signer_possession,
)
from civitasos_runtime.memory import LocalMemory

from test_checkpoint_adapters import IDENTITY_ID, _backend_projection


class _Signer:
    def __init__(self) -> None:
        self.key = SigningKey.generate()

    @property
    def public_key_hex(self) -> str:
        return self.key.verify_key.encode().hex()

    def sign(self, message: bytes) -> bytes:
        return self.key.sign(message).signature


def _memory(path: Path) -> LocalMemory:
    memory = LocalMemory(path)
    memory.put("identity_iem_state", {"identity_id": IDENTITY_ID})
    memory.put("identity_iem_anchor", {"version_id": "iem:v1:test"})
    memory.put("expectation_update_log", [])
    memory.put("lessons", ["retain"])
    return memory


def test_authenticated_capture_prepares_six_domains_without_committing(tmp_path: Path) -> None:
    signer = _Signer()
    observed = {}

    def transport(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return _backend_projection(signer.public_key_hex)

    pauses = []

    @contextmanager
    def pause_runtime():
        pauses.append("entered")
        try:
            yield
        finally:
            pauses.append("exited")

    store = AtomicCheckpointStore(tmp_path / "checkpoints")
    client = BackendCheckpointClient(
        "https://backend.internal:8443",
        "owner-jwt",
        transport=transport,
    )
    memory = _memory(tmp_path / "memory")
    manifest = AtomicCheckpointCapture(store, client, pause_runtime).prepare(
        identity_id=IDENTITY_ID,
        memory=memory,
        signer=signer,
        signer_ref="pkcs11:token=civitas;id=01",
    )

    assert manifest["sequence"] == 1
    assert {item["domain"] for item in manifest["domains"]} == {
        "identity", "credential", "iem", "memory", "r2r", "economy"
    }
    assert store.load_latest() is None
    assert pauses == ["entered", "exited"]
    assert observed["authorization"] == "Bearer owner-jwt"
    assert "owner-jwt" not in observed["url"]
    assert "%3A" in observed["url"]
    memory.close()


def test_capture_failure_does_not_prepare_or_leak_pause(tmp_path: Path) -> None:
    signer = _Signer()
    projection = _backend_projection(signer.public_key_hex)
    projection["credential"]["public_key_fingerprint"] = "00" * 32
    active = False

    @contextmanager
    def pause_runtime():
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    store = AtomicCheckpointStore(tmp_path / "checkpoints")
    client = BackendCheckpointClient(
        "https://backend.internal",
        "owner-jwt",
        transport=lambda _request, _timeout: projection,
    )
    memory = _memory(tmp_path / "memory")
    with pytest.raises(ValueError, match="does not bind"):
        AtomicCheckpointCapture(store, client, pause_runtime).prepare(
            identity_id=IDENTITY_ID,
            memory=memory,
            signer=signer,
            signer_ref="callback:agent",
        )
    assert active is False
    assert list(store.prepared.iterdir()) == []
    assert store.load_latest() is None
    memory.close()


def test_signer_possession_rejects_mismatched_signature() -> None:
    signer = _Signer()
    other = _Signer()
    signer.sign = other.sign  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="possession verification failed"):
        verify_signer_possession(IDENTITY_ID, signer)


@pytest.mark.parametrize(
    "url,token",
    [
        ("http://backend.internal", "jwt"),
        ("https://user:pass@backend.internal", "jwt"),
        ("https://backend.internal?token=query", "jwt"),
        ("https://backend.internal", "bad token"),
    ],
)
def test_backend_client_rejects_unsafe_transport_configuration(url: str, token: str) -> None:
    with pytest.raises(ValueError):
        BackendCheckpointClient(url, token)
