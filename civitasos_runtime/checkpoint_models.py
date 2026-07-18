"""Canonical data models for an atomic identity checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


CHECKPOINT_SCHEMA = "civitasos-atomic-identity-checkpoint:v1"
DOMAIN_SCHEMA = "civitasos-atomic-identity-domain:v1"
REQUIRED_DOMAINS = frozenset({"identity", "credential", "iem", "memory", "r2r", "economy"})
_FORBIDDEN_KEYS = {
    "jwt",
    "password",
    "pin",
    "private_key",
    "seed_hex",
    "secret",
    "signing_key",
    "signing_seed",
    "token",
}
_FORBIDDEN_SUFFIXES = ("_jwt", "_password", "_pin", "_secret", "_token")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"


def _reject_secrets(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS or normalized.endswith(_FORBIDDEN_SUFFIXES):
                raise ValueError(f"checkpoint contains forbidden secret field: {path}.{key}")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


@dataclass(frozen=True)
class DomainSnapshot:
    domain: str
    identity_id: str
    revision: str
    source_ref: str
    payload: dict[str, Any]
    payload_hash: str
    schema_version: str = DOMAIN_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        domain: str,
        identity_id: str,
        revision: str,
        source_ref: str,
        payload: dict[str, Any],
    ) -> "DomainSnapshot":
        snapshot = cls(
            domain=domain,
            identity_id=identity_id,
            revision=revision,
            source_ref=source_ref,
            payload=payload,
            payload_hash=sha256_payload(payload),
        )
        snapshot.validate()
        return snapshot

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DomainSnapshot":
        try:
            snapshot = cls(
                schema_version=str(value["schema_version"]),
                domain=str(value["domain"]),
                identity_id=str(value["identity_id"]),
                revision=str(value["revision"]),
                source_ref=str(value["source_ref"]),
                payload=dict(value["payload"]),
                payload_hash=str(value["payload_hash"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint domain snapshot") from error
        snapshot.validate()
        return snapshot

    def validate(self) -> None:
        if self.schema_version != DOMAIN_SCHEMA:
            raise ValueError(f"unsupported domain schema: {self.schema_version}")
        if self.domain not in REQUIRED_DOMAINS:
            raise ValueError(f"unsupported checkpoint domain: {self.domain}")
        if not self.identity_id or not self.revision or not self.source_ref:
            raise ValueError(f"{self.domain} checkpoint binding is incomplete")
        _reject_secrets(self.payload)
        if self.payload_hash != sha256_payload(self.payload):
            raise ValueError(f"{self.domain} checkpoint payload hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "identity_id": self.identity_id,
            "revision": self.revision,
            "source_ref": self.source_ref,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
        }


def checkpoint_body(
    identity_id: str,
    sequence: int,
    previous_checkpoint_id: str | None,
    snapshots: list[DomainSnapshot],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "identity_id": identity_id,
        "sequence": sequence,
        "previous_checkpoint_id": previous_checkpoint_id,
        "domains": [item.as_dict() for item in sorted(snapshots, key=lambda item: item.domain)],
    }
