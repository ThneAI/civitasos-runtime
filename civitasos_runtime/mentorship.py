"""J1-C authenticated, fail-closed mentorship advice projection."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

_MAX_RESPONSE_BYTES = 1_048_576
_MAX_PROJECTION_ADVICE = 128


class AdviceProvider(Protocol):
    async def fetch(self, apprentice_did: str) -> dict[str, Any]: ...


Transport = Callable[[urllib.request.Request, float], dict[str, Any]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _default_transport(
    request: urllib.request.Request, timeout: float
) -> dict[str, Any]:
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"backend mentorship HTTP status {response.status}")
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("backend mentorship response is too large")
            value = json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"backend mentorship HTTP status {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("backend mentorship request failed") from error
    if not isinstance(value, dict):
        raise RuntimeError("backend mentorship response must be a JSON object")
    return value


@dataclass(frozen=True)
class BackendMentorshipClient:
    base_url: str
    bearer_token: str
    timeout: float = 5.0
    transport: Transport = _default_transport

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("backend mentorship URL must use HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "backend mentorship URL must not contain credentials or query data"
            )
        if not self.bearer_token or any(char.isspace() for char in self.bearer_token):
            raise ValueError("backend mentorship bearer token is invalid")
        if self.timeout <= 0:
            raise ValueError("backend mentorship timeout must be positive")

    def fetch_current_advice(self, relation_id: str) -> dict[str, Any]:
        if not relation_id.strip() or len(relation_id) > 2048:
            raise ValueError("mentorship relation_id must contain 1 to 2048 characters")
        relation = urllib.parse.quote(relation_id, safe="")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/v1/mentorship/relations/{relation}/advice",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            },
            method="GET",
        )
        return self.transport(request, self.timeout)


@dataclass(frozen=True)
class BackendMentorshipAdviceProvider:
    client: BackendMentorshipClient
    relation_ids: tuple[str, ...]
    max_advice: int = 8

    def __post_init__(self) -> None:
        relation_ids = tuple(
            dict.fromkeys(value.strip() for value in self.relation_ids if value.strip())
        )
        if not relation_ids or len(relation_ids) > 32:
            raise ValueError("mentorship provider requires 1 to 32 relation ids")
        if any(len(value) > 2048 for value in relation_ids):
            raise ValueError("mentorship relation ids must not exceed 2048 characters")
        if self.max_advice < 1 or self.max_advice > 32:
            raise ValueError("mentorship max_advice must be between 1 and 32")
        object.__setattr__(self, "relation_ids", relation_ids)

    async def fetch(self, apprentice_did: str) -> dict[str, Any]:
        if not apprentice_did.startswith("did:"):
            raise ValueError("mentorship apprentice identity must be a DID")
        results = await asyncio.gather(
            *(
                asyncio.to_thread(self.client.fetch_current_advice, relation)
                for relation in self.relation_ids
            ),
            return_exceptions=True,
        )
        advice: list[dict[str, Any]] = []
        unavailable: list[str] = []
        omitted = 0
        conflict = False
        for relation_id, result in zip(self.relation_ids, results, strict=True):
            if isinstance(result, BaseException):
                unavailable.append(relation_id)
                continue
            try:
                projection = _projection_data(result, relation_id, apprentice_did)
                raw_omitted = projection.get("omitted_fact_count", 0)
                if type(raw_omitted) is not int or raw_omitted < 0:
                    raise ValueError("mentorship omitted_fact_count is invalid")
                if type(projection.get("conflicting_advice_present")) is not bool:
                    raise ValueError("mentorship conflict marker is invalid")
                normalized_items = [
                    _validated_advice(value, relation_id, apprentice_did)
                    for value in projection["advice"]
                ]
            except ValueError:
                unavailable.append(relation_id)
                continue
            conflict = conflict or projection.get("conflicting_advice_present") is True
            omitted += raw_omitted
            for normalized in normalized_items:
                if normalized is None:
                    omitted += 1
                else:
                    advice.append(normalized)
        advice.sort(
            key=lambda item: (
                item["relation_id"],
                item["advice_id"],
                item["source_fact_hash"],
            )
        )
        if len(advice) > self.max_advice:
            omitted += len(advice) - self.max_advice
            advice = advice[: self.max_advice]
        conflict = conflict or len({item["recommendation"] for item in advice}) > 1
        status = "available" if advice else "empty"
        if unavailable:
            status = "partial" if advice else "unavailable"
        return {
            "schema_version": "j1-runtime-advice-context:v1",
            "status": status,
            "apprentice_did": apprentice_did,
            "advice": advice,
            "omitted_fact_count": omitted,
            "unavailable_relation_ids": unavailable,
            "conflicting_advice_present": conflict,
            "automatic_execution_allowed": False,
        }


def _projection_data(
    value: dict[str, Any], relation_id: str, apprentice_did: str
) -> dict[str, Any]:
    data = value.get("data") if value.get("success") is True else None
    if not isinstance(data, dict):
        raise ValueError("mentorship response does not contain a successful projection")
    if data.get("schema_version") != "j1-apprentice-advice-projection:v1":
        raise ValueError("unsupported mentorship advice projection schema")
    if (
        data.get("relation_id") != relation_id
        or data.get("apprentice_did") != apprentice_did
    ):
        raise ValueError("mentorship advice projection identity binding mismatch")
    advice = data.get("advice")
    if data.get("automatic_execution_allowed") is not False or not isinstance(
        advice, list
    ):
        raise ValueError(
            "mentorship advice projection violates the non-execution boundary"
        )
    if len(advice) > _MAX_PROJECTION_ADVICE:
        raise ValueError("mentorship advice projection exceeds the item limit")
    return data


def _validated_advice(
    value: Any, relation_id: str, apprentice_did: str
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    strings = {
        key: value.get(key)
        for key in (
            "advice_id",
            "observation_id",
            "mentor_did",
            "recommendation",
            "source_fact_hash",
            "observation_fact_hash",
            "activation_semantic_hash",
        )
    }
    if any(not isinstance(item, str) or not item for item in strings.values()):
        return None
    if (
        not strings["mentor_did"].startswith("did:")
        or strings["mentor_did"] == apprentice_did
    ):
        return None
    if len(strings["recommendation"]) > 2048:
        return None
    if any(
        not _sha256_hex(strings[key])
        for key in (
            "source_fact_hash",
            "observation_fact_hash",
            "activation_semantic_hash",
        )
    ):
        return None
    expires_at = value.get("expires_at")
    if not isinstance(expires_at, int) or expires_at <= int(time.time()):
        return None
    expected_boundary = {
        "advisory_only": True,
        "direct_execution_allowed": False,
        "constitution_override_allowed": False,
        "normative_mutation_allowed": False,
        "identity_mutation_allowed": False,
        "memory_mutation_allowed": False,
    }
    if any(
        value.get(key) is not expected for key, expected in expected_boundary.items()
    ):
        return None
    return {
        "relation_id": relation_id,
        **strings,
        "expires_at": expires_at,
        **expected_boundary,
    }


def _sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
