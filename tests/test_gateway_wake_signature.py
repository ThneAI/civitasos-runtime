from __future__ import annotations

from civitasos_runtime.gateway import (
    build_wake_audit_context,
    build_wake_event_record,
    build_wake_signature,
    validate_wake_signature,
)


def test_wake_signature_accepts_when_secret_not_configured() -> None:
    ok, reason = validate_wake_signature(None, {}, b"{}")

    assert ok is True
    assert reason == "signature_not_required"


def test_wake_signature_validates_body_and_timestamp() -> None:
    body = b'{"event":"task.posted"}'
    headers = {
        "X-Civitas-Webhook-Issuer": "civitasos-backend",
        "X-Civitas-Webhook-Timestamp": "1700000000",
        "X-Civitas-Webhook-Signature": build_wake_signature(
            "secret",
            "1700000000",
            body,
        ),
    }

    ok, reason = validate_wake_signature(
        "secret",
        headers,
        body,
        now=1700000000,
    )

    assert ok is True
    assert reason == "signature_valid"


def test_wake_signature_rejects_tampered_body_and_stale_timestamp() -> None:
    body = b'{"event":"task.posted"}'
    headers = {
        "X-Civitas-Webhook-Issuer": "civitasos-backend",
        "X-Civitas-Webhook-Timestamp": "1700000000",
        "X-Civitas-Webhook-Signature": build_wake_signature(
            "secret",
            "1700000000",
            body,
        ),
    }

    ok, reason = validate_wake_signature(
        "secret",
        headers,
        b'{"event":"task.claimed"}',
        now=1700000000,
    )
    assert ok is False
    assert reason == "signature_mismatch"

    ok, reason = validate_wake_signature("secret", headers, body, now=1700001000)
    assert ok is False
    assert reason == "stale_signature_timestamp"


def test_wake_signature_rejects_wrong_issuer() -> None:
    body = b'{"event":"task.posted"}'
    headers = {
        "X-Civitas-Webhook-Issuer": "other-issuer",
        "X-Civitas-Webhook-Timestamp": "1700000000",
        "X-Civitas-Webhook-Signature": build_wake_signature(
            "secret",
            "1700000000",
            body,
            issuer="other-issuer",
        ),
    }

    ok, reason = validate_wake_signature(
        "secret",
        headers,
        body,
        now=1700000000,
    )

    assert ok is False
    assert reason == "invalid_signature_issuer"


def test_wake_audit_context_extracts_event_and_task_id() -> None:
    context = build_wake_audit_context(
        {"X-Civitas-Webhook-Issuer": "civitasos-backend"},
        {
            "event": "task.posted",
            "subscription_id": "sub-1",
            "data": {"task_id": "task-1"},
        },
    )

    assert context == {
        "schema_version": "civitasos-wake-audit-context:v1",
        "issuer": "civitasos-backend",
        "event": "task.posted",
        "task_id": "task-1",
        "subscription_id": "sub-1",
        "signature_present": False,
    }


def test_wake_event_record_preserves_lifecycle_payload() -> None:
    record = build_wake_event_record(
        {
            "X-Civitas-Webhook-Issuer": "civitasos-backend",
            "X-Civitas-Webhook-Signature": "sha256=abc",
        },
        {
            "event": "task.delivered",
            "subscription_id": "sub-1",
            "agent_id": "did:civ:agent",
            "timestamp": "2026-05-17T12:00:00Z",
            "data": {
                "task_id": "task-1",
                "required_capability": "review",
            },
        },
    )

    assert record == {
        "schema_version": "civitasos-wake-event-record:v1",
        "event": "task.delivered",
        "task_id": "task-1",
        "agent_id": "did:civ:agent",
        "subscription_id": "sub-1",
        "issuer": "civitasos-backend",
        "backend_timestamp": "2026-05-17T12:00:00Z",
        "signature_present": True,
        "data": {
            "task_id": "task-1",
            "required_capability": "review",
        },
    }
