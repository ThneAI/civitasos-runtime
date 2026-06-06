from __future__ import annotations

from typing import Any

import pytest

from civitasos_runtime.llm import LLMAdapter
from civitasos_runtime.memory import HybridMemory
from civitasos_runtime.models import LLMResponse
from civitasos_runtime.runner import AgentRunner


class _NoopLLM(LLMAdapter):
    async def _do_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return LLMResponse(content="wait")


class _ProbeAgent:
    def __init__(self, agent_id: str, public_key_hex: str) -> None:
        self.agent_id = agent_id
        self.public_key_hex = public_key_hex

    def briefing(self) -> dict[str, Any]:
        return {"active_tasks": []}


@pytest.mark.asyncio
async def test_runner_records_identity_continuity_across_restart(tmp_path) -> None:
    agent = _ProbeAgent("did:civ:devnet:z-probe", "ab" * 32)
    first = AgentRunner(llm=_NoopLLM(), data_dir=str(tmp_path))
    first._agent = agent
    first._memory = HybridMemory(agent=None, data_dir=tmp_path)

    await first._shutdown_cleanup()

    second = AgentRunner(llm=_NoopLLM(), data_dir=str(tmp_path))
    second._agent = agent
    second._memory = HybridMemory(agent=None, data_dir=tmp_path)
    evidence = await second._recover()

    assert evidence is not None
    assert evidence["identity_continuous"] is True
    assert evidence["agent_id_continuous"] is True
    assert evidence["public_key_continuous"] is True
    assert evidence["previous_runtime_instance_id"] == first._runtime_instance_id
    assert evidence["current_runtime_instance_id"] == second._runtime_instance_id
    assert evidence["previous_runtime_instance_id"] != evidence["current_runtime_instance_id"]
    assert second._memory.recall("restart_continuity_state") == evidence
    second._memory.close()


@pytest.mark.asyncio
async def test_runner_fails_continuity_when_key_changes(tmp_path) -> None:
    first = AgentRunner(llm=_NoopLLM(), data_dir=str(tmp_path))
    first._agent = _ProbeAgent("did:civ:devnet:z-probe", "ab" * 32)
    first._memory = HybridMemory(agent=None, data_dir=tmp_path)
    await first._shutdown_cleanup()

    second = AgentRunner(llm=_NoopLLM(), data_dir=str(tmp_path))
    second._agent = _ProbeAgent("did:civ:devnet:z-probe", "cd" * 32)
    second._memory = HybridMemory(agent=None, data_dir=tmp_path)
    evidence = await second._recover()

    assert evidence is not None
    assert evidence["identity_continuous"] is False
    assert evidence["agent_id_continuous"] is True
    assert evidence["public_key_continuous"] is False
    second._memory.close()
