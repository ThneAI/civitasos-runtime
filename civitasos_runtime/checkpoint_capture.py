"""Authenticated backend capture orchestration for atomic identity checkpoints."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from nacl.signing import VerifyKey

from .atomic_checkpoint import AtomicCheckpointStore
from .checkpoint_adapters import (
    backend_projection_snapshots,
    capture_runtime_snapshots,
)
from .memory import LocalMemory


class CheckpointSigner(Protocol):
    @property
    def public_key_hex(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


Transport = Callable[[urllib.request.Request, float], dict[str, Any]]
PauseFactory = Callable[[], AbstractContextManager[Any]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _default_transport(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"backend checkpoint HTTP status {response.status}")
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"backend checkpoint HTTP status {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("backend checkpoint request failed") from error
    if not isinstance(value, dict):
        raise RuntimeError("backend checkpoint response must be a JSON object")
    return value


@dataclass(frozen=True)
class BackendCheckpointClient:
    base_url: str
    bearer_token: str
    timeout: float = 10.0
    transport: Transport = _default_transport

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("backend checkpoint URL must use HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("backend checkpoint URL must not contain credentials or query data")
        if not self.bearer_token or any(char.isspace() for char in self.bearer_token):
            raise ValueError("backend checkpoint bearer token is invalid")
        if self.timeout <= 0:
            raise ValueError("backend checkpoint timeout must be positive")

    def fetch(self, identity_id: str) -> dict[str, Any]:
        if not identity_id.strip():
            raise ValueError("checkpoint identity_id is required")
        path_id = urllib.parse.quote(identity_id, safe="")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/v1/checkpoints/identity/{path_id}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            },
            method="GET",
        )
        return self.transport(request, self.timeout)


def verify_signer_possession(identity_id: str, signer: CheckpointSigner) -> str:
    """Sign and locally verify a fresh domain-bound possession challenge."""
    public_key_hex = signer.public_key_hex
    try:
        public_key = bytes.fromhex(public_key_hex)
    except ValueError as error:
        raise ValueError("checkpoint signer public key must be hex") from error
    if len(public_key) != 32:
        raise ValueError("checkpoint signer public key must be 32 bytes")
    challenge = os.urandom(32)
    message = b"civitasos-checkpoint-possession:v1\x00" + identity_id.encode() + b"\x00" + challenge
    signature = signer.sign(message)
    if not isinstance(signature, bytes):
        raise ValueError("checkpoint signer must return signature bytes")
    try:
        VerifyKey(public_key).verify(message, signature)
    except Exception as error:  # PyNaCl exposes multiple signature/backend errors.
        raise ValueError("checkpoint signer possession verification failed") from error
    return public_key_hex.lower()


class AtomicCheckpointCapture:
    def __init__(
        self,
        store: AtomicCheckpointStore,
        backend: BackendCheckpointClient,
        pause_runtime: PauseFactory,
    ) -> None:
        self.store = store
        self.backend = backend
        self.pause_runtime = pause_runtime

    def prepare(
        self,
        *,
        identity_id: str,
        memory: LocalMemory,
        signer: CheckpointSigner,
        signer_ref: str,
    ) -> dict[str, Any]:
        if not signer_ref.strip():
            raise ValueError("checkpoint signer_ref is required")
        with self.pause_runtime():
            public_key_hex = verify_signer_possession(identity_id, signer)
            projection = self.backend.fetch(identity_id)
            source_ref = (
                f"backend:{projection.get('node_id')}:"
                f"epoch-{projection.get('barrier_epoch')}"
            )
            backend_snapshots = backend_projection_snapshots(
                projection,
                identity_id=identity_id,
                public_key_hex=public_key_hex,
                signer_ref=signer_ref,
                source_ref=source_ref,
            )
            runtime_snapshots = capture_runtime_snapshots(
                memory,
                identity_id=identity_id,
                source_ref="sqlite://memory.db",
            )
            latest = self.store.load_latest()
            sequence = 1 if latest is None else int(latest["sequence"]) + 1
            previous = None if latest is None else str(latest["checkpoint_id"])
            return self.store.prepare(
                identity_id=identity_id,
                sequence=sequence,
                previous_checkpoint_id=previous,
                snapshots=(*backend_snapshots, *runtime_snapshots),
            )
