"""Prepare/commit storage for cross-domain identity checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .checkpoint_files import (
    locked_file,
    read_private_json,
    secure_directory,
    write_private_json,
)
from .checkpoint_models import (
    CHECKPOINT_SCHEMA,
    REQUIRED_DOMAINS,
    DomainSnapshot,
    checkpoint_body,
    sha256_payload,
)


class AtomicCheckpointStore:
    """Expose only a fully validated manifest selected by one commit marker."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.prepared = self.root / "prepared"
        self.committed = self.root / "committed"
        self.aborted = self.root / "aborted"
        for path in (self.root, self.prepared, self.committed, self.aborted):
            secure_directory(path)
        self.lock_path = self.root / "checkpoint.lock"
        self.latest_path = self.root / "LATEST"

    def prepare(
        self,
        *,
        identity_id: str,
        sequence: int,
        snapshots: Iterable[DomainSnapshot],
        previous_checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        items = list(snapshots)
        self._validate_domain_set(identity_id, items)
        body = checkpoint_body(identity_id, sequence, previous_checkpoint_id, items)
        checkpoint_id = f"aic:v1:{sha256_payload(body).removeprefix('sha256:')[:24]}"
        manifest = {**body, "checkpoint_id": checkpoint_id}
        manifest["manifest_hash"] = sha256_payload(manifest)
        self._validate_manifest(manifest)

        with self._lock():
            latest = self._load_latest_unlocked()
            self._validate_successor(manifest, latest)
            path = self._manifest_path(self.prepared, checkpoint_id)
            if path.exists():
                if read_private_json(path) != manifest:
                    raise ValueError(f"conflicting prepared checkpoint: {checkpoint_id}")
            else:
                write_private_json(path, manifest)
        return manifest

    def commit(self, checkpoint_id: str) -> dict[str, Any]:
        with self._lock():
            prepared_path = self._manifest_path(self.prepared, checkpoint_id)
            committed_path = self._manifest_path(self.committed, checkpoint_id)
            if prepared_path.exists():
                manifest = read_private_json(prepared_path)
            elif committed_path.exists():
                manifest = read_private_json(committed_path)
            else:
                raise ValueError(f"unknown prepared checkpoint: {checkpoint_id}")
            self._validate_manifest(manifest)
            latest = self._load_latest_unlocked()
            if latest and latest["checkpoint_id"] == checkpoint_id:
                return manifest
            self._validate_successor(manifest, latest)

            if committed_path.exists():
                if read_private_json(committed_path) != manifest:
                    raise ValueError(f"conflicting committed checkpoint: {checkpoint_id}")
            else:
                write_private_json(committed_path, manifest)
            marker = {
                "schema_version": CHECKPOINT_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "manifest_hash": manifest["manifest_hash"],
                "sequence": manifest["sequence"],
            }
            write_private_json(self.latest_path, marker)
            prepared_path.unlink(missing_ok=True)
            return manifest

    def abort(self, checkpoint_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("checkpoint abort reason is required")
        with self._lock():
            latest = self._load_latest_unlocked()
            if latest and latest["checkpoint_id"] == checkpoint_id:
                raise ValueError("cannot abort the active checkpoint")
            source = self._manifest_path(self.prepared, checkpoint_id)
            if not source.exists():
                raise ValueError(f"unknown prepared checkpoint: {checkpoint_id}")
            manifest = read_private_json(source)
            record = {"manifest": manifest, "reason": reason}
            write_private_json(self._manifest_path(self.aborted, checkpoint_id), record)
            source.unlink()

    def load_latest(self) -> dict[str, Any] | None:
        with self._lock():
            marker = self._load_latest_unlocked()
            if marker is None:
                return None
            manifest = read_private_json(
                self._manifest_path(self.committed, marker["checkpoint_id"])
            )
            self._validate_manifest(manifest)
            if manifest["manifest_hash"] != marker.get("manifest_hash"):
                raise ValueError("checkpoint commit marker hash mismatch")
            return manifest

    @staticmethod
    def snapshots(manifest: dict[str, Any]) -> dict[str, DomainSnapshot]:
        return {
            value["domain"]: DomainSnapshot.from_dict(value)
            for value in manifest["domains"]
        }

    def _load_latest_unlocked(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        marker = read_private_json(self.latest_path)
        checkpoint_id = str(marker.get("checkpoint_id", ""))
        manifest_path = self._manifest_path(self.committed, checkpoint_id)
        if not manifest_path.exists():
            raise ValueError("checkpoint commit marker references missing manifest")
        manifest = read_private_json(manifest_path)
        self._validate_manifest(manifest)
        if marker.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint commit marker schema")
        if marker.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("checkpoint commit marker hash mismatch")
        if marker.get("sequence") != manifest["sequence"]:
            raise ValueError("checkpoint commit marker sequence mismatch")
        return manifest

    @staticmethod
    def _validate_domain_set(identity_id: str, snapshots: list[DomainSnapshot]) -> None:
        domains = [item.domain for item in snapshots]
        if set(domains) != REQUIRED_DOMAINS or len(domains) != len(REQUIRED_DOMAINS):
            raise ValueError("checkpoint requires each of the six identity domains exactly once")
        for snapshot in snapshots:
            snapshot.validate()
            if snapshot.identity_id != identity_id:
                raise ValueError("checkpoint cannot mix identity bindings")

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint manifest schema")
        sequence = manifest.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("checkpoint sequence must be a positive integer")
        snapshots = [DomainSnapshot.from_dict(value) for value in manifest.get("domains", [])]
        self._validate_domain_set(str(manifest.get("identity_id", "")), snapshots)
        body = checkpoint_body(
            manifest["identity_id"],
            manifest["sequence"],
            manifest.get("previous_checkpoint_id"),
            snapshots,
        )
        expected_id = f"aic:v1:{sha256_payload(body).removeprefix('sha256:')[:24]}"
        if manifest.get("checkpoint_id") != expected_id:
            raise ValueError("checkpoint id mismatch")
        without_hash = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if manifest.get("manifest_hash") != sha256_payload(without_hash):
            raise ValueError("checkpoint manifest hash mismatch")

    @staticmethod
    def _validate_successor(
        manifest: dict[str, Any], latest: dict[str, Any] | None
    ) -> None:
        expected_sequence = 1 if latest is None else latest["sequence"] + 1
        expected_previous = None if latest is None else latest["checkpoint_id"]
        if manifest["sequence"] != expected_sequence:
            raise ValueError(f"checkpoint sequence must be {expected_sequence}")
        if manifest.get("previous_checkpoint_id") != expected_previous:
            raise ValueError("checkpoint previous id does not match active checkpoint")

    @staticmethod
    def _manifest_path(directory: Path, checkpoint_id: str) -> Path:
        if not checkpoint_id.startswith("aic:v1:") or "/" in checkpoint_id:
            raise ValueError("invalid checkpoint id")
        return directory / f"{checkpoint_id}.json"

    def _lock(self):
        return locked_file(self.lock_path)
