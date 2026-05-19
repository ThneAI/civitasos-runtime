from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from civitasos_runtime.runner import AgentRunner


@dataclass
class _FakeResponse:
    success: bool
    data: dict | None = None
    error: str | None = None
    hint: str | None = None


class _FakeAgent:
    def __init__(self) -> None:
        self._public_key_hex = "ab" * 32
        self._agent_id = None
        self.quickstart_calls = 0
        self.register_calls = 0
        self.post_calls: list[tuple[str, dict]] = []

    def a2a_quickstart(self, **_kwargs) -> None:
        self.quickstart_calls += 1
        raise AssertionError("a2a_quickstart should not be called in this scenario")

    def register(self, **_kwargs) -> None:
        self.register_calls += 1
        raise AssertionError("register should not be called in this scenario")

    def _post(self, path: str, payload: dict) -> _FakeResponse:
        self.post_calls.append((path, payload))
        return _FakeResponse(
            success=True,
            data={"agent": {"did": "did:civ:devnet:test-runner"}},
        )


class _SponsorRequiredAgent(_FakeAgent):
    def a2a_quickstart(self, **_kwargs) -> None:
        self.quickstart_calls += 1
        raise RuntimeError("sponsor_required: institutional identity is enabled")

    def register(self, **_kwargs) -> None:
        self.register_calls += 1
        raise RuntimeError("sponsor_required: institutional identity is enabled")


class _QuickstartAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.saved_paths: list[str] = []
        self.authenticated = False
        self._jwt_auth_context = {
            "auth_method": "none",
            "production_allowed": False,
            "evidence_allowed": False,
        }

    def a2a_quickstart(self, **_kwargs) -> None:
        self.quickstart_calls += 1
        self._agent_id = "did:civ:devnet:quickstart"

    def save_identity(self, path: str) -> None:
        self.saved_paths.append(path)

    def authenticate(self, *, allow_legacy_fallback: bool = True) -> str:
        assert allow_legacy_fallback is False
        self.authenticated = True
        self._jwt_auth_context = {
            "auth_method": "did_signature_challenge",
            "production_allowed": False,
            "evidence_allowed": True,
        }
        return "jwt-challenge"

    @property
    def jwt_auth_context(self) -> dict:
        return dict(self._jwt_auth_context)


class _ServiceTokenQuickstartAgent(_QuickstartAgent):
    def __init__(self) -> None:
        super().__init__()
        self._jwt_token = None
        self._jwt_expires_at = 0
        self.service_token_calls: list[dict] = []

    def authenticate_service_token(
        self,
        *,
        service_id: str,
        secret: str,
        scopes: list[str],
    ) -> str:
        self.service_token_calls.append(
            {
                "service_id": service_id,
                "secret": secret,
                "scopes": scopes,
            }
        )
        self._jwt_token = "service-jwt"
        self._jwt_auth_context = {
            "auth_method": "service_token",
            "production_allowed": False,
            "evidence_allowed": False,
            "service_id": service_id,
            "scopes": scopes,
        }
        return self._jwt_token


class _NoopLLM:
    async def decide(self, _prompt: str, **_kwargs) -> dict:
        return {"action": "wait", "params": {}, "reasoning": "noop"}

    async def reflect(self, _context: str, **_kwargs) -> str:
        return "noop"


def _new_runner() -> AgentRunner:
    return AgentRunner(
        base_url="http://localhost:8099",
        name="Alpha Trader",
        capabilities=["trading", "analysis"],
        llm=_NoopLLM(),
    )


def test_register_uses_birth_proposal_when_institutional_enabled(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", "true")
    monkeypatch.setenv("CIVITASOS_BIRTH_SPONSOR", "alphatrader")

    runner = _new_runner()
    fake = _FakeAgent()
    runner._agent = fake
    monkeypatch.setattr(runner, "_bootstrap_demo_jwt", lambda: None)

    asyncio.run(runner._register())

    assert fake.quickstart_calls == 0
    assert fake.register_calls == 0
    assert len(fake.post_calls) == 1
    path, payload = fake.post_calls[0]
    assert path == "/agents/birth-proposal"
    assert payload["sponsor"] == "alphatrader"
    assert payload["public_key"] == fake._public_key_hex
    assert fake._agent_id == "did:civ:devnet:test-runner"


def test_register_falls_back_to_birth_proposal_on_sponsor_required(monkeypatch) -> None:
    monkeypatch.delenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", raising=False)
    monkeypatch.setenv("BENCHMARK_BIRTH_SPONSOR", "alphatrader")

    runner = _new_runner()
    fake = _SponsorRequiredAgent()
    runner._agent = fake
    monkeypatch.setattr(runner, "_bootstrap_demo_jwt", lambda: None)

    asyncio.run(runner._register())

    assert fake.quickstart_calls == 1
    assert fake.register_calls == 0
    assert len(fake.post_calls) == 1
    path, payload = fake.post_calls[0]
    assert path == "/agents/birth-proposal"
    assert payload["sponsor"] == "alphatrader"
    assert fake._agent_id == "did:civ:devnet:test-runner"


def test_register_persists_did_and_prefers_challenge_auth(tmp_path) -> None:
    runner = _new_runner()
    runner._identity_file = str(tmp_path / "agent.identity.json")
    fake = _QuickstartAgent()
    runner._agent = fake

    asyncio.run(runner._register())

    assert fake.quickstart_calls == 1
    assert fake.saved_paths == [runner._identity_file]
    assert fake.authenticated is True
    assert fake.jwt_auth_context["auth_method"] == "did_signature_challenge"


def test_register_prefers_service_token_bootstrap_before_quickstart(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("L1_SERVICE_TOKEN_SECRET", "service-secret")
    monkeypatch.setenv("L1_SERVICE_ID", "l1_contract_runner")
    monkeypatch.setenv("L1_SERVICE_TOKEN_SCOPES", "agents:read,agents:write")

    runner = _new_runner()
    runner._identity_file = str(tmp_path / "agent.identity.json")
    fake = _ServiceTokenQuickstartAgent()
    runner._agent = fake
    monkeypatch.setattr(
        runner,
        "_bootstrap_demo_jwt",
        lambda: (_ for _ in ()).throw(
            AssertionError("service-token bootstrap must not call demo-login")
        ),
    )

    asyncio.run(runner._register())

    assert fake.service_token_calls == [
        {
            "service_id": "l1_contract_runner",
            "secret": "service-secret",
            "scopes": ["agents:read", "agents:write"],
        }
    ]
    assert fake.quickstart_calls == 1
    assert fake.saved_paths == [runner._identity_file]
    assert fake.authenticated is True
    assert fake.jwt_auth_context["auth_method"] == "did_signature_challenge"


def test_register_fails_closed_when_service_token_required_without_secret(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_RUNTIME_REQUIRE_SERVICE_TOKEN_BOOTSTRAP", "1")
    monkeypatch.delenv("CIVITASOS_RUNTIME_SERVICE_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("L1_SERVICE_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("L1_PILOT_001_SERVICE_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("CIVITASOS_SERVICE_TOKEN_SECRET", raising=False)

    runner = _new_runner()
    fake = _QuickstartAgent()
    runner._agent = fake
    monkeypatch.setattr(
        runner,
        "_bootstrap_demo_jwt",
        lambda: (_ for _ in ()).throw(
            AssertionError("strict service-token mode must not call demo-login")
        ),
    )

    with pytest.raises(RuntimeError, match="service-token bootstrap is required"):
        asyncio.run(runner._register())

    assert fake.quickstart_calls == 0
