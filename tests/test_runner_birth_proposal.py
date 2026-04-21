from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
