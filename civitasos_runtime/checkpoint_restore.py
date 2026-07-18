"""Replayable Runtime restore coordinator for an active atomic checkpoint."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

from .atomic_checkpoint import AtomicCheckpointStore
from .checkpoint_adapters import restore_runtime_snapshots
from .checkpoint_capture import (
    BackendCheckpointClient,
    CheckpointSigner,
    verify_signer_possession,
)
from .checkpoint_files import (
    locked_file,
    read_private_json,
    secure_directory,
    write_private_json,
)
from .checkpoint_observer import (
    CheckpointRestoreMilestone,
    CheckpointRestoreObserver,
    emit_checkpoint_restore_milestone,
)
from .memory import LocalMemory


PauseFactory = Callable[[], AbstractContextManager[Any]]


class CheckpointRestoreIncomplete(RuntimeError):
    """Restore stopped after local materialization and must be replayed before ticks."""


class AtomicCheckpointRestore:
    def __init__(
        self,
        store: AtomicCheckpointStore,
        backend: BackendCheckpointClient,
        pause_runtime: PauseFactory,
        observer: CheckpointRestoreObserver | None = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.pause_runtime = pause_runtime
        self.observer = observer
        self.journal_dir = self.store.root / "restore_journal"
        secure_directory(self.journal_dir)
        self.lock_path = self.store.root / "restore.lock"

    def needs_replay(self) -> bool:
        """Return whether a durable incomplete journal blocks Runtime ticks."""
        return self.replay_required(self.store)

    @staticmethod
    def replay_required(store: AtomicCheckpointStore) -> bool:
        journal_dir = store.root / "restore_journal"
        if not journal_dir.exists():
            return False
        secure_directory(journal_dir)
        pending = []
        for path in sorted(journal_dir.glob("*.json")):
            record = read_private_json(path)
            AtomicCheckpointRestore._validate_journal_shape(record)
            status = record["status"]
            if status != "activated":
                pending.append(record)
        if not pending:
            return False
        if len(pending) != 1:
            raise ValueError("multiple incomplete Runtime restore journals")
        manifest = store.load_latest()
        if manifest is None:
            raise ValueError("incomplete restore journal has no active checkpoint")
        AtomicCheckpointRestore._validate_journal(pending[0], manifest)
        return True

    def restore_latest(
        self,
        *,
        identity_id: str,
        memory: LocalMemory,
        signer: CheckpointSigner,
    ) -> dict[str, Any]:
        with locked_file(self.lock_path):
            manifest = self.store.load_latest()
            if manifest is None:
                raise ValueError("no active checkpoint to restore")
            if manifest["identity_id"] != identity_id:
                raise ValueError("active checkpoint identity binding mismatch")
            journal_path = self._journal_path(manifest["checkpoint_id"])
            existing = read_private_json(journal_path) if journal_path.exists() else None
            if existing is not None:
                self._validate_journal(existing, manifest)
                if existing["status"] == "activated":
                    return existing

            with self.pause_runtime():
                snapshots = self.store.snapshots(manifest)
                identity = snapshots["identity"]
                public_key_hex = verify_signer_possession(identity_id, signer)
                if identity.payload.get("public_key_hex") != public_key_hex:
                    raise ValueError("restore signer does not match checkpoint identity")
                request = self._backend_request(manifest, snapshots)
                backend_record = self.backend.restore_preflight(identity_id, request)
                self._validate_backend_record(
                    backend_record,
                    manifest,
                    identity_id,
                    request["node_id"],
                    {"validated", "activated"},
                )
                self._observe(
                    CheckpointRestoreMilestone.BACKEND_PREFLIGHT_DURABLE,
                    manifest,
                    backend_record,
                )
                restoring = self._write_journal(
                    journal_path, manifest, "restoring", backend_record
                )
                self._observe(
                    CheckpointRestoreMilestone.RUNTIME_RESTORE_JOURNAL_DURABLE,
                    manifest,
                    restoring,
                )
                restore_runtime_snapshots(
                    memory,
                    identity_id=identity_id,
                    iem=snapshots["iem"],
                    memory_snapshot=snapshots["memory"],
                )
                self._observe(
                    CheckpointRestoreMilestone.RUNTIME_SQLITE_COMMITTED,
                    manifest,
                    {"status": "runtime_applied"},
                )
                runtime_applied = self._write_journal(
                    journal_path, manifest, "runtime_applied", backend_record
                )
                self._observe(
                    CheckpointRestoreMilestone.RUNTIME_APPLIED_JOURNAL_DURABLE,
                    manifest,
                    runtime_applied,
                )
                try:
                    activated = self.backend.restore_activate(
                        identity_id,
                        manifest["checkpoint_id"],
                        manifest["manifest_hash"],
                    )
                except Exception as error:
                    raise CheckpointRestoreIncomplete(
                        "backend activation failed after Runtime restore; replay before resuming ticks"
                    ) from error
                self._validate_backend_record(
                    activated,
                    manifest,
                    identity_id,
                    request["node_id"],
                    {"activated"},
                )
                self._observe(
                    CheckpointRestoreMilestone.BACKEND_ACTIVATION_DURABLE,
                    manifest,
                    activated,
                )
                record = self._write_journal(
                    journal_path, manifest, "activated", activated
                )
                self._observe(
                    CheckpointRestoreMilestone.RUNTIME_ACTIVATION_JOURNAL_DURABLE,
                    manifest,
                    record,
                )
                return record

    def _observe(
        self,
        milestone: CheckpointRestoreMilestone,
        manifest: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        emit_checkpoint_restore_milestone(
            self.observer,
            milestone,
            {**manifest, **record},
        )

    def _backend_request(self, manifest, snapshots) -> dict[str, Any]:  # noqa: ANN001
        identity_payload = snapshots["identity"].payload
        node_id = identity_payload.get("backend_node_id")
        backend_identity = identity_payload.get("backend_identity")
        if not isinstance(node_id, str) or not isinstance(backend_identity, dict):
            raise ValueError("checkpoint identity lacks backend materialization binding")
        return {
            "checkpoint_id": manifest["checkpoint_id"],
            "manifest_hash": manifest["manifest_hash"],
            "sequence": manifest["sequence"],
            "node_id": node_id,
            "materialization": {
                "identity": backend_identity,
                "credential": snapshots["credential"].payload,
                "r2r": snapshots["r2r"].payload,
                "economy": snapshots["economy"].payload,
            },
        }

    @staticmethod
    def _validate_backend_record(
        record: dict[str, Any],
        manifest: dict[str, Any],
        identity_id: str,
        node_id: str,
        expected_statuses: set[str],
    ) -> None:
        if (
            record.get("checkpoint_id") != manifest["checkpoint_id"]
            or record.get("manifest_hash") != manifest["manifest_hash"]
            or record.get("sequence") != manifest["sequence"]
            or record.get("identity_id") != identity_id
            or record.get("node_id") != node_id
            or record.get("status") not in expected_statuses
            or (
                record.get("status") == "activated"
                and not record.get("activation_fact_id")
            )
        ):
            raise ValueError("backend restore journal response binding mismatch")

    def _write_journal(
        self,
        path: Path,
        manifest: dict[str, Any],
        status: str,
        backend_record: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "schema_version": "civitasos-runtime-checkpoint-restore-journal:v1",
            "checkpoint_id": manifest["checkpoint_id"],
            "manifest_hash": manifest["manifest_hash"],
            "sequence": manifest["sequence"],
            "identity_id": manifest["identity_id"],
            "status": status,
            "backend_status": backend_record["status"],
            "activation_fact_id": backend_record.get("activation_fact_id"),
        }
        write_private_json(path, record)
        return record

    @staticmethod
    def _validate_journal(record: dict[str, Any], manifest: dict[str, Any]) -> None:
        AtomicCheckpointRestore._validate_journal_shape(record)
        if (
            record.get("checkpoint_id") != manifest["checkpoint_id"]
            or record.get("manifest_hash") != manifest["manifest_hash"]
            or record.get("sequence") != manifest["sequence"]
            or record.get("identity_id") != manifest["identity_id"]
        ):
            raise ValueError("Runtime restore journal binding mismatch")

    @staticmethod
    def _validate_journal_shape(record: dict[str, Any]) -> None:
        if (
            record.get("schema_version")
            != "civitasos-runtime-checkpoint-restore-journal:v1"
            or record.get("status")
            not in {"restoring", "runtime_applied", "activated"}
            or not str(record.get("checkpoint_id", "")).startswith("aic:v1:")
            or "/" in str(record.get("checkpoint_id", ""))
            or (
                record.get("status") == "activated"
                and not isinstance(record.get("activation_fact_id"), str)
            )
            or (
                record.get("status") == "activated"
                and not record.get("activation_fact_id", "").strip()
            )
        ):
            raise ValueError("Runtime restore journal is invalid")

    def _journal_path(self, checkpoint_id: str) -> Path:
        if not checkpoint_id.startswith("aic:v1:") or "/" in checkpoint_id:
            raise ValueError("invalid checkpoint id")
        return self.journal_dir / f"{checkpoint_id}.json"
