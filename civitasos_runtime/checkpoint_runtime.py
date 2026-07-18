"""Fail-stop tick gate for checkpoint restore orchestration."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .checkpoint_files import read_private_json, secure_directory, write_private_json


class CheckpointTickBlocked(RuntimeError):
    """The Runtime must not execute ticks until checkpoint activation completes."""


class RuntimeTickLatch:
    """Process-local tick gate released only by a bound activation record."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocked_reason: str | None = None
        self._activation: dict[str, Any] | None = None

    def block(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("checkpoint tick latch reason is required")
        with self._lock:
            self._blocked_reason = reason
            self._activation = None

    def release_after_activation(self, record: dict[str, Any]) -> None:
        if record.get("status") != "activated":
            raise ValueError("checkpoint tick latch requires an activated record")
        if not record.get("checkpoint_id") or not record.get("manifest_hash"):
            raise ValueError("checkpoint activation binding is incomplete")
        if not isinstance(record.get("activation_fact_id"), str) or not record[
            "activation_fact_id"
        ].strip():
            raise ValueError("checkpoint activation Fact binding is missing")
        with self._lock:
            self._activation = dict(record)
            self._blocked_reason = None

    def require_tick_allowed(self) -> None:
        with self._lock:
            reason = self._blocked_reason
        if reason is not None:
            raise CheckpointTickBlocked(f"Runtime tick blocked: {reason}")

    @property
    def blocked(self) -> bool:
        with self._lock:
            return self._blocked_reason is not None

    @property
    def blocked_reason(self) -> str | None:
        with self._lock:
            return self._blocked_reason

    @property
    def activation(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._activation is None else dict(self._activation)


class RuntimeRestoreIntentStore:
    """Durable fail-stop intent written before any restore network call."""

    SCHEMA_VERSION = "civitasos-runtime-restore-intent:v1"

    def __init__(self, checkpoint_root: Path) -> None:
        self.directory = checkpoint_root / "runtime_restore_intents"
        secure_directory(self.directory)

    def needs_replay(self, manifest: dict[str, Any] | None) -> bool:
        blocked = []
        for path in sorted(self.directory.glob("*.json")):
            record = read_private_json(path)
            self._validate_record(record)
            if record["status"] == "blocked":
                blocked.append(record)
        if not blocked:
            return False
        if len(blocked) != 1:
            raise ValueError("multiple blocked Runtime restore intents")
        if manifest is None:
            raise ValueError("blocked restore intent has no active checkpoint")
        self._validate_binding(blocked[0], manifest)
        return True

    def begin(self, manifest: dict[str, Any]) -> dict[str, Any]:
        path = self._path(manifest["checkpoint_id"])
        if path.exists():
            record = read_private_json(path)
            self._validate_record(record)
            self._validate_binding(record, manifest)
            return record
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "checkpoint_id": manifest["checkpoint_id"],
            "manifest_hash": manifest["manifest_hash"],
            "sequence": manifest["sequence"],
            "identity_id": manifest["identity_id"],
            "status": "blocked",
            "activation_fact_id": None,
        }
        write_private_json(path, record)
        return record

    def complete(
        self,
        manifest: dict[str, Any],
        activation: dict[str, Any],
    ) -> dict[str, Any]:
        if activation.get("status") != "activated":
            raise ValueError("restore intent requires activated backend state")
        self._validate_binding(activation, manifest)
        if not isinstance(activation.get("activation_fact_id"), str) or not activation[
            "activation_fact_id"
        ].strip():
            raise ValueError("restore intent requires an activation Fact")
        path = self._path(manifest["checkpoint_id"])
        current = read_private_json(path)
        self._validate_record(current)
        self._validate_binding(current, manifest)
        record = {
            **current,
            "status": "activated",
            "activation_fact_id": activation.get("activation_fact_id"),
        }
        write_private_json(path, record)
        return record

    @classmethod
    def _validate_record(cls, record: dict[str, Any]) -> None:
        if (
            record.get("schema_version") != cls.SCHEMA_VERSION
            or record.get("status") not in {"blocked", "activated"}
            or not str(record.get("checkpoint_id", "")).startswith("aic:v1:")
            or not str(record.get("manifest_hash", "")).startswith("sha256:")
            or isinstance(record.get("sequence"), bool)
            or not isinstance(record.get("sequence"), int)
            or record.get("sequence", 0) < 1
            or not isinstance(record.get("identity_id"), str)
            or not record.get("identity_id", "").strip()
            or "/" in str(record.get("checkpoint_id", ""))
            or (
                record.get("status") == "activated"
                and (
                    not isinstance(record.get("activation_fact_id"), str)
                    or not record.get("activation_fact_id", "").strip()
                )
            )
        ):
            raise ValueError("Runtime restore intent is invalid")

    @staticmethod
    def _validate_binding(record: dict[str, Any], manifest: dict[str, Any]) -> None:
        if any(
            record.get(field) != manifest.get(field)
            for field in ("checkpoint_id", "manifest_hash", "sequence", "identity_id")
        ):
            raise ValueError("Runtime restore intent binding mismatch")

    def _path(self, checkpoint_id: str) -> Path:
        if not checkpoint_id.startswith("aic:v1:") or "/" in checkpoint_id:
            raise ValueError("invalid checkpoint id")
        return self.directory / f"{checkpoint_id}.json"
