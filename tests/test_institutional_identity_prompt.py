from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.models import LLMResponse
from civitasos_runtime.tools import ToolRegistry


class _FakeResponse:
    def __init__(self, *, success: bool, data: dict | None = None, error: str | None = None) -> None:
        self.success = success
        self.data = data
        self.error = error


class _FakeAgent:
    def __init__(self) -> None:
        self.agent_id = "did:civ:devnet:test-agent"

    def briefing(self) -> dict:
        return {"economics": {"current_epoch": 7}}

    def _get(self, path: str) -> _FakeResponse:
        assert path == "/agents/did%3Aciv%3Adevnet%3Atest-agent/identity-state"
        return _FakeResponse(
            success=True,
            data={
                "did": self.agent_id,
                "state": "PROVISIONAL",
                "sponsor_did": "did:civ:devnet:guardian",
                "obligations": ["complete_assigned_tasks"],
                "obligation_expiry_epoch": 9,
                "age_in_epochs": 1,
                "stake_locked": 100,
            },
        )


class _CaptureLLM:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    async def chat(self, messages, tools=None, temperature=0.3) -> LLMResponse:  # noqa: ANN001
        self.messages = messages
        return LLMResponse(content="wait")


def test_perceive_injects_institutional_identity_summary(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", "true")

    loop = CognitiveLoop(
        _FakeAgent(),
        llm=_CaptureLLM(),
        agent_name="ProbeAgent",
        capabilities=["analysis"],
        tools=ToolRegistry(),
    )

    briefing = asyncio.run(loop._perceive())

    assert briefing["_identity_prompt_injected"] is False
    assert briefing["identity"]["state"] == "PROVISIONAL"
    assert briefing["identity"]["remaining_epochs"] == 2
    assert briefing["identity"]["at_risk"] is False
    assert briefing["identity"]["obligations"] == ["complete_assigned_tasks"]


def test_perceive_uses_identity_state_current_epoch_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", "true")

    class _NoEconomicsAgent(_FakeAgent):
        def briefing(self) -> dict:
            return {}

        def _get(self, path: str) -> _FakeResponse:
            resp = super()._get(path)
            assert isinstance(resp.data, dict)
            resp.data["current_epoch"] = 8
            return resp

    loop = CognitiveLoop(
        _NoEconomicsAgent(),
        llm=_CaptureLLM(),
        agent_name="ProbeAgent",
        capabilities=["analysis"],
        tools=ToolRegistry(),
    )

    briefing = asyncio.run(loop._perceive())

    assert briefing["identity"]["remaining_epochs"] == 1
    assert briefing["identity"]["at_risk"] is True


def test_perceive_reuses_cached_identity_after_transient_fetch_failure(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", "true")

    class _FlakyIdentityAgent(_FakeAgent):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def briefing(self) -> dict:
            return {"economics": {"current_epoch": 8}}

        def _get(self, path: str) -> _FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return _FakeResponse(
                    success=True,
                    data={
                        "did": self.agent_id,
                        "state": "PROVISIONAL",
                        "sponsor_did": "did:civ:devnet:guardian",
                        "obligations": ["complete_assigned_tasks"],
                        "obligation_expiry_epoch": 10,
                        "age_in_epochs": 1,
                        "stake_locked": 100,
                    },
                )
            return _FakeResponse(success=False, error="rate limit")

    loop = CognitiveLoop(
        _FlakyIdentityAgent(),
        llm=_CaptureLLM(),
        agent_name="ProbeAgent",
        capabilities=["analysis"],
        tools=ToolRegistry(),
    )

    first = asyncio.run(loop._perceive())
    second = asyncio.run(loop._perceive())

    assert first["identity"]["remaining_epochs"] == 2
    assert second["identity"]["state"] == "PROVISIONAL"
    assert second["identity"]["remaining_epochs"] == 2
    assert second["identity"]["obligations"] == ["complete_assigned_tasks"]


def test_decide_llm_injects_institutional_identity_prompt(monkeypatch) -> None:
    monkeypatch.setenv("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED", "true")
    llm = _CaptureLLM()
    loop = CognitiveLoop(
        _FakeAgent(),
        llm=llm,
        agent_name="ProbeAgent",
        capabilities=["analysis"],
        tools=ToolRegistry(),
    )

    briefing = {
        "active_tasks": [],
        "_identity_prompt_injected": False,
        "identity": {
            "state": "PROVISIONAL",
            "sponsor_did": "did:civ:devnet:guardian",
            "age_in_epochs": 1,
            "remaining_epochs": 1,
            "obligations": ["complete_assigned_tasks"],
            "at_risk": True,
        },
    }

    decision = asyncio.run(loop._decide_llm(briefing, memories={}))

    assert decision is not None
    assert decision.action == "wait"
    assert briefing["_identity_prompt_injected"] is True
    assert llm.messages is not None
    system = llm.messages[0]["content"]
    assert "制度性身份状态" in system
    assert "PROVISIONAL" in system
    assert "complete_assigned_tasks" in system
    assert "避免失权或清算" in system
