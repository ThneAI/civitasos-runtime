from __future__ import annotations

import asyncio

from civitasos_runtime.loop import CognitiveLoop


class _Agent:
    agent_id = "did:civ:test:agent"

    def briefing(self) -> dict:
        return {
            "agent": {"did": self.agent_id, "capabilities": ["review"]},
            "active_tasks": [],
            "opportunities": [],
        }

    def recall(self, _key: str):
        return None

    def recall_similar(self, _query: str, top_k: int = 3) -> list:
        return []

    def pool_claim(self, task_id: str, agent_id: str | None = None, stake_amount: int = 0):
        return {
            "success": True,
            "task_id": task_id,
            "agent_id": agent_id,
            "stake_amount": stake_amount,
        }


class _LLM:
    async def chat(self, *_args, **_kwargs):
        raise AssertionError("LLM should not be called")


def test_wake_events_are_exposed_once_in_next_briefing() -> None:
    loop = CognitiveLoop(_Agent(), llm=_LLM())
    loop.record_wake_event(
        {
            "schema_version": "civitasos-wake-event-record:v1",
            "event": "task.delivered",
            "task_id": "task-1",
        }
    )

    first = asyncio.run(loop._perceive())
    second = asyncio.run(loop._perceive())

    assert first["backend_wake_events"] == [
        {
            "schema_version": "civitasos-wake-event-record:v1",
            "event": "task.delivered",
            "task_id": "task-1",
        }
    ]
    assert first["latest_backend_wake_event"]["task_id"] == "task-1"
    assert "backend_wake_events" not in second


def test_wake_event_queue_is_bounded() -> None:
    loop = CognitiveLoop(_Agent(), llm=_LLM())
    for index in range(40):
        loop.record_wake_event({"event": "task.posted", "task_id": f"task-{index}"})

    briefing = asyncio.run(loop._perceive())

    assert len(briefing["backend_wake_events"]) == 32
    assert briefing["backend_wake_events"][0]["task_id"] == "task-8"
    assert briefing["backend_wake_events"][-1]["task_id"] == "task-39"


def test_task_posted_wake_builds_pool_claim_action_bias() -> None:
    loop = CognitiveLoop(_Agent(), llm=_LLM(), capabilities=["review"])
    loop.record_wake_event(
        {
            "schema_version": "civitasos-wake-event-record:v1",
            "event": "task.posted",
            "task_id": "task-review-1",
            "data": {
                "task_id": "task-review-1",
                "requester": "did:civ:test:requester",
                "required_capability": "review",
            },
        }
    )

    briefing = asyncio.run(loop._perceive())
    decision = asyncio.run(loop._decide(briefing, memories={}))

    assert briefing["backend_wake_action_bias"] == {
        "schema_version": "civitasos-wake-action-bias:v1",
        "source_event": "task.posted",
        "action": "pool_claim",
        "task_id": "task-review-1",
        "required_capability": "review",
        "requester": "did:civ:test:requester",
        "confidence": 0.92,
        "reason": "backend task.posted event matches local capability and no active task is held",
        "non_claims": [
            "wake_bias_does_not_skip_conscience",
            "wake_bias_does_not_execute_task_output",
        ],
    }
    assert decision is not None
    assert decision.action == "pool_claim"
    assert decision.params["task_id"] == "task-review-1"
    assert decision.params["_wake_event_driven"] is True


def test_task_posted_wake_ignores_nonmatching_or_self_posted_tasks() -> None:
    loop = CognitiveLoop(_Agent(), llm=_LLM(), capabilities=["review"])
    loop.record_wake_event(
        {
            "event": "task.posted",
            "task_id": "task-implementation-1",
            "data": {
                "task_id": "task-implementation-1",
                "requester": "did:civ:test:requester",
                "required_capability": "implementation",
            },
        }
    )
    loop.record_wake_event(
        {
            "event": "task.posted",
            "task_id": "task-self-1",
            "data": {
                "task_id": "task-self-1",
                "requester": "did:civ:test:agent",
                "required_capability": "review",
            },
        }
    )

    briefing = asyncio.run(loop._perceive())

    assert "backend_wake_action_bias" not in briefing


def test_task_posted_wake_matches_briefing_capability_objects() -> None:
    class AgentWithObjectCaps(_Agent):
        def briefing(self) -> dict:
            return {
                "agent": {
                    "did": self.agent_id,
                    "capabilities": [{"id": "boundary_check"}],
                },
                "active_tasks": [],
                "opportunities": [],
            }

    loop = CognitiveLoop(AgentWithObjectCaps(), llm=_LLM(), capabilities=[])
    loop.record_wake_event(
        {
            "event": "task.posted",
            "task_id": "task-boundary-1",
            "data": {
                "task_id": "task-boundary-1",
                "requester": "did:civ:test:requester",
                "required_capability": "boundary_check",
            },
        }
    )

    briefing = asyncio.run(loop._perceive())

    assert briefing["backend_wake_action_bias"]["action"] == "pool_claim"
    assert briefing["backend_wake_action_bias"]["task_id"] == "task-boundary-1"


def test_task_delivered_wake_builds_review_followup_bias_without_autodecision() -> None:
    loop = CognitiveLoop(_Agent(), llm=_LLM(), capabilities=["review"])
    loop.record_wake_event(
        {
            "event": "task.delivered",
            "task_id": "task-delivered-1",
            "data": {
                "task_id": "task-delivered-1",
                "requester": "did:civ:test:requester",
                "agent_id": "did:civ:test:worker",
                "required_capability": "implementation",
                "status": "Delivered",
            },
        }
    )

    briefing = asyncio.run(loop._perceive())

    assert briefing["backend_wake_action_bias"]["action"] == "review_delivery"
    assert briefing["backend_wake_action_bias"]["source_event"] == "task.delivered"
    assert briefing["backend_wake_action_bias"]["task_id"] == "task-delivered-1"
    assert "inspect delivered output before confirmation or dispute" in briefing[
        "backend_wake_action_bias"
    ]["followup_focus"]
    assert loop._decision_from_wake_action_bias(briefing) is None


def test_task_failed_wake_builds_repair_followup_bias_even_with_active_tasks() -> None:
    class AgentWithActiveTask(_Agent):
        def briefing(self) -> dict:
            briefing = super().briefing()
            briefing["active_tasks"] = [{"task_id": "active-1"}]
            return briefing

    loop = CognitiveLoop(AgentWithActiveTask(), llm=_LLM(), capabilities=["repair"])
    loop.record_wake_event(
        {
            "event": "task.failed",
            "task_id": "task-failed-1",
            "data": {
                "task_id": "task-failed-1",
                "requester": "did:civ:test:requester",
                "agent_id": "did:civ:test:worker",
                "required_capability": "repair",
                "failure_reason": "delivery contract violation",
            },
        }
    )

    briefing = asyncio.run(loop._perceive())

    assert briefing["backend_wake_action_bias"]["action"] == "repair_or_review_failure"
    assert briefing["backend_wake_action_bias"]["source_event"] == "task.failed"
    assert briefing["backend_wake_action_bias"]["failure_reason"] == "delivery contract violation"
    assert "do not silently retry" in briefing["backend_wake_action_bias"]["followup_focus"][-1]
    assert loop._decision_from_wake_action_bias(briefing) is None
