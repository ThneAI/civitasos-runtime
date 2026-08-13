"""Content-minimized cost measurement artifacts for Backend cost Facts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


COST_FACT_SCHEMA = "civitasos-cost-fact:v1"
COST_RATE_SCHEMA = "civitasos-cost-rate:v1"
STORAGE_SNAPSHOT_SCHEMA = "civitasos-storage-snapshot:v1"
NETWORK_MEASUREMENT_SCHEMA = "civitasos-network-measurement:v1"
OPERATOR_TIME_SCHEMA = "civitasos-operator-time-attestation:v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidenceReference:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri or len(self.uri) > 256 or any(
            ord(char) < 32 for char in self.uri
        ):
            raise ValueError("evidence URI is invalid")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("evidence SHA-256 is invalid")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class StorageEntry:
    path_sha256: str
    kind: str
    logical_bytes: int
    allocated_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class StorageSnapshot:
    measured_at: int
    root_sha256: str
    logical_bytes: int
    allocated_bytes: int
    regular_file_count: int
    directory_count: int
    entry_manifest_sha256: str
    entries: tuple[StorageEntry, ...]

    def as_artifact(self) -> dict[str, Any]:
        content = {
            "schema_version": STORAGE_SNAPSHOT_SCHEMA,
            **asdict(self),
            "raw_paths_recorded": False,
            "file_content_recorded": False,
            "symlinks_followed": False,
        }
        return {**content, "artifact_sha256": canonical_sha256(content)}


def collect_storage_snapshot(root: Path, *, measured_at: int) -> StorageSnapshot:
    if root.is_symlink():
        raise ValueError("storage snapshot refuses symbolic links")
    root = root.resolve(strict=True)
    if not root.is_dir() or measured_at < 0:
        raise ValueError("storage snapshot root or timestamp is invalid")
    entries: list[StorageEntry] = []
    logical_bytes = 0
    allocated_bytes = 0
    regular_file_count = 0
    directory_count = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise ValueError("storage snapshot refuses symbolic links")
            stat = path.stat(follow_symlinks=False)
            relative = path.relative_to(root).as_posix()
            is_file = path.is_file()
            if is_file:
                regular_file_count += 1
                logical_bytes += stat.st_size
            elif path.is_dir():
                directory_count += 1
            else:
                raise ValueError("storage snapshot accepts only files and directories")
            allocated = int(getattr(stat, "st_blocks", 0)) * 512
            allocated_bytes += allocated
            entries.append(
                StorageEntry(
                    path_sha256=hashlib.sha256(relative.encode()).hexdigest(),
                    kind="file" if is_file else "directory",
                    logical_bytes=stat.st_size if is_file else 0,
                    allocated_bytes=allocated,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    entry_manifest = tuple(entries)
    return StorageSnapshot(
        measured_at=measured_at,
        root_sha256=hashlib.sha256(str(root).encode()).hexdigest(),
        logical_bytes=logical_bytes,
        allocated_bytes=allocated_bytes,
        regular_file_count=regular_file_count,
        directory_count=directory_count,
        entry_manifest_sha256=canonical_sha256(
            [asdict(entry) for entry in entry_manifest]
        ),
        entries=entry_manifest,
    )


def storage_byte_seconds_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    snapshot: StorageSnapshot,
    retention_started_at: int,
    retention_ended_at: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if retention_started_at < 0 or retention_ended_at < retention_started_at:
        raise ValueError("storage retention window is invalid")
    if snapshot.measured_at > retention_started_at:
        raise ValueError("storage snapshot must exist before the retention window")
    duration = retention_ended_at - retention_started_at
    artifact = snapshot.as_artifact()
    source = EvidenceReference(
        uri=f"artifact:storage-snapshot:{artifact['artifact_sha256']}",
        sha256=artifact["artifact_sha256"],
    )
    request = _measurement_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        entry_id=entry_id,
        category="storage",
        quantity=snapshot.allocated_bytes * duration,
        unit="byte_seconds",
        measured_at=retention_ended_at,
        window={"started_at": retention_started_at, "ended_at": retention_ended_at},
        allocation_basis="immutable_allocated_bytes_times_retention_seconds",
        source_refs=[source],
    )
    return artifact, request


def network_bytes_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    call_id: str,
    request_bytes: int,
    response_bytes: int,
    connect_attempts: int,
    request_dispatched: bool,
    measured_at: int,
    transport_receipt: EvidenceReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    counters = (request_bytes, response_bytes, connect_attempts, measured_at)
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("network counters are invalid")
    if not call_id or connect_attempts < 1:
        raise ValueError("network call identity or connect attempts are invalid")
    content = {
        "schema_version": NETWORK_MEASUREMENT_SCHEMA,
        "call_id": call_id,
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "connect_attempts": connect_attempts,
        "request_dispatched": request_dispatched,
        "measured_at": measured_at,
        "transport_receipt": transport_receipt.as_dict(),
        "measurement_scope": "application_payload_bytes",
        "wire_bytes_observed": False,
        "raw_request_recorded": False,
        "raw_response_recorded": False,
    }
    artifact = {**content, "artifact_sha256": canonical_sha256(content)}
    source = EvidenceReference(
        uri=f"artifact:network-measurement:{artifact['artifact_sha256']}",
        sha256=artifact["artifact_sha256"],
    )
    request = _measurement_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        entry_id=entry_id,
        category="network",
        quantity=request_bytes + response_bytes,
        unit="bytes",
        measured_at=measured_at,
        window=None,
        allocation_basis="provider_broker_call_id",
        source_refs=[transport_receipt, source],
    )
    return artifact, request


def operator_time_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    operator_role: str,
    activity_code: str,
    started_at: int,
    ended_at: int,
    attestation_signature: EvidenceReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if ended_at < started_at or started_at < 0:
        raise ValueError("operator time window is invalid")
    if not operator_role or not _snake_case(activity_code):
        raise ValueError("operator role or activity code is invalid")
    unsigned = {
        "schema_version": OPERATOR_TIME_SCHEMA,
        "task_id": task_id,
        "operator_id": actor,
        "operator_role": operator_role,
        "activity_code": activity_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "active_seconds": ended_at - started_at,
        "hidden_reasoning_recorded": False,
    }
    artifact = {
        **unsigned,
        "unsigned_payload_sha256": canonical_sha256(unsigned),
        "signature_ref": attestation_signature.as_dict(),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    source = EvidenceReference(
        uri=f"artifact:operator-time:{artifact['artifact_sha256']}",
        sha256=artifact["artifact_sha256"],
    )
    request = _measurement_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        entry_id=entry_id,
        category="operator_review_time",
        quantity=ended_at - started_at,
        unit="seconds",
        measured_at=ended_at,
        window={"started_at": started_at, "ended_at": ended_at},
        allocation_basis="operator_signed_active_seconds",
        source_refs=[source, attestation_signature],
    )
    return artifact, request


def rate_publication_request(
    *,
    operation_id: str,
    actor: str,
    rate_id: str,
    category: str,
    unit: str,
    currency: str,
    price_microunits: int,
    per_quantity: int,
    effective_from: int,
    effective_until: int | None,
    source: EvidenceReference,
) -> dict[str, Any]:
    if price_microunits < 0 or per_quantity < 1 or effective_from < 0:
        raise ValueError("rate values are invalid")
    if effective_until is not None and effective_until <= effective_from:
        raise ValueError("rate effective window is invalid")
    return {
        "schema_version": COST_FACT_SCHEMA,
        "operation_id": operation_id,
        "task_id": None,
        "actor": actor,
        "event": {
            "kind": "rate_published",
            "data": {
                "schema_version": COST_RATE_SCHEMA,
                "rate_id": rate_id,
                "category": category,
                "unit": unit,
                "currency": currency,
                "price_microunits": price_microunits,
                "per_quantity": per_quantity,
                "effective_from": effective_from,
                "effective_until": effective_until,
                "source": source.as_dict(),
            },
        },
    }


def reservation_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    category: str,
    currency: str,
    reserved_microunits: int,
    source_refs: list[EvidenceReference],
) -> dict[str, Any]:
    if reserved_microunits < 1 or not source_refs:
        raise ValueError("reservation amount and sources are required")
    return _cost_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        kind="reserved",
        data={
            "entry_id": entry_id,
            "category": category,
            "currency": currency,
            "reserved_microunits": reserved_microunits,
            "source_refs": [reference.as_dict() for reference in source_refs],
        },
    )


def reconciliation_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    measurement_operation_id: str,
    reservation_operation_id: str | None,
    rate_id: str,
    source_refs: list[EvidenceReference],
) -> dict[str, Any]:
    if not measurement_operation_id or not rate_id or not source_refs:
        raise ValueError("reconciliation references are required")
    return _cost_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        kind="reconciled",
        data={
            "entry_id": entry_id,
            "measurement_operation_id": measurement_operation_id,
            "reservation_operation_id": reservation_operation_id,
            "rate_id": rate_id,
            "source_refs": [reference.as_dict() for reference in source_refs],
        },
    )


def unknown_cost_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    category: str,
    currency: str,
    upper_bound_microunits: int,
    reason_code: str,
    source_refs: list[EvidenceReference],
) -> dict[str, Any]:
    if upper_bound_microunits < 1 or not _snake_case(reason_code) or not source_refs:
        raise ValueError("unknown cost requires an upper bound, reason, and sources")
    return _cost_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        kind="unknown",
        data={
            "entry_id": entry_id,
            "category": category,
            "currency": currency,
            "upper_bound_microunits": upper_bound_microunits,
            "reason_code": reason_code,
            "source_refs": [reference.as_dict() for reference in source_refs],
        },
    )


def waived_cost_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    category: str,
    currency: str,
    reason_code: str,
    authority_refs: list[EvidenceReference],
) -> dict[str, Any]:
    if not _snake_case(reason_code) or not authority_refs:
        raise ValueError("waived cost requires a reason and authority")
    return _cost_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        kind="waived",
        data={
            "entry_id": entry_id,
            "category": category,
            "currency": currency,
            "reason_code": reason_code,
            "authority_refs": [reference.as_dict() for reference in authority_refs],
        },
    )


def _measurement_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    entry_id: str,
    category: str,
    quantity: int,
    unit: str,
    measured_at: int,
    window: dict[str, int] | None,
    allocation_basis: str,
    source_refs: list[EvidenceReference],
) -> dict[str, Any]:
    return _cost_request(
        operation_id=operation_id,
        task_id=task_id,
        actor=actor,
        kind="measured",
        data={
            "entry_id": entry_id,
            "category": category,
            "quantity": quantity,
            "unit": unit,
            "measured_at": measured_at,
            "window": window,
            "allocation_basis": allocation_basis,
            "source_refs": [reference.as_dict() for reference in source_refs],
        },
    )


def _cost_request(
    *,
    operation_id: str,
    task_id: str,
    actor: str,
    kind: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": COST_FACT_SCHEMA,
        "operation_id": operation_id,
        "task_id": task_id,
        "actor": actor,
        "event": {"kind": kind, "data": data},
    }


def _snake_case(value: str) -> bool:
    return bool(value) and len(value) <= 64 and all(
        char in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in value
    )
