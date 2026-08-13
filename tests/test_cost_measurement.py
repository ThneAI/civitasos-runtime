from __future__ import annotations

import json
from pathlib import Path

import pytest

from civitasos_runtime.cost_measurement import (
    EvidenceReference,
    collect_storage_snapshot,
    network_bytes_request,
    operator_time_request,
    rate_publication_request,
    reconciliation_request,
    reservation_request,
    storage_byte_seconds_request,
    unknown_cost_request,
    waived_cost_request,
)


def evidence(seed: str) -> EvidenceReference:
    return EvidenceReference(uri=f"evidence:test:{seed}", sha256=seed * 64)


def test_storage_snapshot_is_content_minimized_and_builds_byte_seconds(
    tmp_path: Path,
) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "secret-name.txt").write_bytes(b"abc")
    snapshot = collect_storage_snapshot(tmp_path, measured_at=10)
    artifact, request = storage_byte_seconds_request(
        operation_id="cost:storage:measure:1",
        task_id="task-1",
        actor="did:civ:meter",
        entry_id="cost:storage:task-1",
        snapshot=snapshot,
        retention_started_at=10,
        retention_ended_at=20,
    )
    assert snapshot.logical_bytes == 3
    assert request["event"]["data"]["quantity"] == snapshot.allocated_bytes * 10
    assert request["event"]["data"]["unit"] == "byte_seconds"
    encoded = json.dumps(artifact)
    assert "secret-name.txt" not in encoded
    assert artifact["raw_paths_recorded"] is False
    assert artifact["file_content_recorded"] is False
    assert len(artifact["entries"]) == 2
    assert all(len(entry["path_sha256"]) == 64 for entry in artifact["entries"])
    assert artifact["entry_manifest_sha256"]


def test_storage_snapshot_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("value")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        collect_storage_snapshot(tmp_path, measured_at=1)


def test_storage_snapshot_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "root-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        collect_storage_snapshot(link, measured_at=1)


def test_network_measurement_records_only_counters_and_hashes() -> None:
    artifact, request = network_bytes_request(
        operation_id="cost:network:measure:1",
        task_id="task-1",
        actor="did:civ:broker",
        entry_id="cost:network:call-1",
        call_id="call-1",
        request_bytes=101,
        response_bytes=203,
        connect_attempts=2,
        request_dispatched=True,
        measured_at=20,
        transport_receipt=evidence("a"),
    )
    assert request["event"]["data"]["quantity"] == 304
    assert artifact["raw_request_recorded"] is False
    assert artifact["raw_response_recorded"] is False
    assert set(artifact).isdisjoint({"request", "response", "api_key"})


def test_operator_time_requires_explicit_signature_reference() -> None:
    artifact, request = operator_time_request(
        operation_id="cost:operator:measure:1",
        task_id="task-1",
        actor="did:civ:operator",
        entry_id="cost:operator:review-1",
        operator_role="independent_reviewer",
        activity_code="protocol_review",
        started_at=100,
        ended_at=160,
        attestation_signature=evidence("b"),
    )
    assert artifact["active_seconds"] == 60
    assert artifact["hidden_reasoning_recorded"] is False
    assert request["event"]["data"]["quantity"] == 60
    assert request["event"]["data"]["allocation_basis"] == (
        "operator_signed_active_seconds"
    )


def test_rate_publication_matches_backend_wire_contract() -> None:
    request = rate_publication_request(
        operation_id="cost:rate:storage:1",
        actor="did:civ:operator",
        rate_id="rate:storage:2026-08",
        category="storage",
        unit="byte_seconds",
        currency="USD",
        price_microunits=1,
        per_quantity=1_000_000,
        effective_from=1,
        effective_until=None,
        source=evidence("c"),
    )
    assert request["schema_version"] == "civitasos-cost-fact:v1"
    assert request["task_id"] is None
    assert request["event"]["kind"] == "rate_published"


def test_terminal_cost_requests_match_backend_wire_contract() -> None:
    reservation = reservation_request(
        operation_id="cost:reserve:1",
        task_id="task-1",
        actor="did:civ:meter",
        entry_id="cost:call-1",
        category="network",
        currency="USD",
        reserved_microunits=20,
        source_refs=[evidence("d")],
    )
    reconciled = reconciliation_request(
        operation_id="cost:reconcile:1",
        task_id="task-1",
        actor="did:civ:meter",
        entry_id="cost:call-1",
        measurement_operation_id="cost:measure:1",
        reservation_operation_id="cost:reserve:1",
        rate_id="rate:network:1",
        source_refs=[evidence("e")],
    )
    unknown = unknown_cost_request(
        operation_id="cost:unknown:1",
        task_id="task-1",
        actor="did:civ:meter",
        entry_id="cost:unknown-1",
        category="storage",
        currency="USD",
        upper_bound_microunits=30,
        reason_code="rate_unavailable",
        source_refs=[evidence("f")],
    )
    waived = waived_cost_request(
        operation_id="cost:waived:1",
        task_id="task-1",
        actor="did:civ:meter",
        entry_id="cost:waived-1",
        category="network",
        currency="USD",
        reason_code="documented_free_tier",
        authority_refs=[evidence("1")],
    )
    assert reservation["event"]["kind"] == "reserved"
    assert reconciled["event"]["kind"] == "reconciled"
    assert unknown["event"]["data"]["upper_bound_microunits"] == 30
    assert waived["event"]["data"]["authority_refs"]


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "not-a-hash"])
def test_evidence_reference_rejects_noncanonical_hash(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        EvidenceReference(uri="evidence:test", sha256=digest)
