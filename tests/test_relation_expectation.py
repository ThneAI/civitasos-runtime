from __future__ import annotations

import asyncio

from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.models import LLMResponse, TickContext
from civitasos_runtime.relation_expectation import apply_relation_matrix_expectation


RELATION_ID = "rel:did:civ:test:requester:did:civ:test:worker"
REQUESTER = "did:civ:test:requester"
WORKER = "did:civ:test:worker"


def _relation_context(**extra):
    ctx = {
        "id": f"{RELATION_ID}:task:backend-1",
        "relation_id": RELATION_ID,
        "relation_id_source": "r2r_registry",
        "relation_pair": {
            "requester": REQUESTER,
            "worker": WORKER,
            "agents": [REQUESTER, WORKER],
        },
        "memory_refs": [
            f"failure:{RELATION_ID}:failed-1:2026-05-01T00_00_00Z",
            "challenge:backend-1:latest",
        ],
        "recent_failures": [
            {
                "task_id": "failed-2",
                "relation_id": RELATION_ID,
                "failed_at": "2026-05-01T00:01:00Z",
            }
        ],
        "source": "backend_read_model",
    }
    ctx.update(extra)
    return ctx


def test_failure_refs_generate_directed_relation_expectation() -> None:
    ctx = TickContext(briefing={"relation_context": _relation_context()})

    assert apply_relation_matrix_expectation(ctx, local_identity=WORKER) is True

    key, expectation = next(iter(ctx.expectations["relation"].items()))
    vector = expectation["expectation"]
    assert key == f"{RELATION_ID}|{WORKER}->{REQUESTER}|predicted"
    assert expectation["from_identity"] == WORKER
    assert expectation["to_identity"] == REQUESTER
    assert vector["expected_trust"] < 0.72
    assert vector["expected_delivery_quality"] < 0.72
    assert vector["expected_betrayal_risk"] > 0.12
    assert vector["precision"] > 0.35

    surprise = ctx.surprise["relation"][key]
    assert surprise["domain"] == "relation"
    assert surprise["state_kind"] == "predicted"
    assert surprise["lifecycle_state"] == "violated"
    assert surprise["surprise_score"] > 0

    bias = ctx.action_bias["relation"][key]
    assert bias["verification_level"] == "strict"
    assert bias["required_stake_multiplier"] > 1.0
    assert bias["trust_hint"] == vector["expected_trust"]
    assert len(bias["reason_event_ids"]) == 2

    changed = {
        update.parameter_name: update
        for update in ctx.expectation_updates
        if update.target == f"relation_expectation:{key}"
    }
    assert changed["expected_trust"].old_value == 0.72
    assert changed["expected_trust"].new_value == vector["expected_trust"]
    assert changed["expected_betrayal_risk"].new_value == vector["expected_betrayal_risk"]
    assert any(update.local_update_blocked for update in ctx.expectation_updates)


def test_repair_refs_recover_slowly_without_erasing_history() -> None:
    relation = _relation_context(
        memory_refs=[f"repair:{RELATION_ID}:repair-1:2026-05-01T00_03_00Z"],
        recent_failures=[],
        recent_repairs=[],
        relation_expectation={
            "expected_trust": 0.40,
            "expected_delivery_quality": 0.45,
            "expected_cooperation": 0.42,
            "expected_betrayal_risk": 0.55,
            "expected_repair_probability": 0.25,
            "precision": 0.50,
        },
    )
    ctx = TickContext(briefing={"relation_context": relation})

    assert apply_relation_matrix_expectation(ctx, local_identity=WORKER) is True

    key, expectation = next(iter(ctx.expectations["relation"].items()))
    vector = expectation["expectation"]
    assert vector["expected_trust"] > 0.40
    assert vector["expected_delivery_quality"] > 0.45
    assert vector["expected_betrayal_risk"] < 0.55
    assert vector["expected_repair_probability"] > 0.25
    assert ctx.surprise["relation"][key]["lifecycle_state"] == "confirmed"
    assert ctx.action_bias["relation"][key]["verification_level"] in {"elevated", "strict"}


def test_relation_context_without_failure_or_repair_is_not_traced() -> None:
    relation = _relation_context(memory_refs=["challenge:backend-1"], recent_failures=[])
    ctx = TickContext(briefing={"relation_context": relation})

    assert apply_relation_matrix_expectation(ctx, local_identity=WORKER) is False
    assert ctx.expectations == {}
    assert ctx.surprise == {}
    assert ctx.action_bias == {}
    assert ctx.expectation_updates == []


class _WaitLLM:
    async def chat(self, *_args, **_kwargs):
        return LLMResponse(content="wait")


class _RelationAgent:
    agent_id = WORKER

    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def briefing(self):
        return {
            "agent": {"did": self.agent_id},
            "active_tasks": [],
            "opportunities": [],
            "relation_context": _relation_context(),
        }

    def recall(self, key):
        return self.saved.get(key)

    def remember(self, key, value):
        self.saved[key] = value

    def recall_similar(self, _query, top_k=3):
        return []

    def log_episode(self, _key, _value):
        return None


class _PlainAgent(_RelationAgent):
    def briefing(self):
        return {
            "agent": {"did": self.agent_id},
            "active_tasks": [],
            "opportunities": [],
        }


def test_loop_runs_relation_expectation_before_decide_and_persists_matrix() -> None:
    agent = _RelationAgent()
    loop = CognitiveLoop(agent, llm=_WaitLLM())

    ctx = asyncio.run(loop.tick())

    assert ctx.decision is not None
    assert ctx.decision.action == "wait"
    key, expectation = next(iter(ctx.expectations["relation"].items()))
    assert ctx.briefing["h0_relation_expectation"] == expectation
    assert ctx.briefing["h0_relation_action_bias"] == ctx.action_bias["relation"][key]
    assert f"relation_expectation:{key}" in agent.saved
    saved = agent.saved[f"relation_expectation:{key}"]
    assert isinstance(saved, dict)
    assert saved["relation_id"] == RELATION_ID
    assert "expectation_update_log" in agent.saved
    assert "identity_iem_state" in agent.saved
    assert "identity_iem_anchor" in agent.saved
    anchor = agent.saved["identity_iem_anchor"]
    assert isinstance(anchor, dict)
    assert anchor["version_id"].startswith("iem:v1:")
    assert anchor["state_hash"].startswith("sha256:")
    assert anchor["latest_update_log_hash"].startswith("sha256:")
    assert ctx.briefing["iem_anchor"] == anchor
    state = agent.saved["identity_iem_state"]
    assert key in state["relation_expectation_matrix"]


def test_on_perceive_hook_can_feed_relation_context_to_expect_phase() -> None:
    agent = _PlainAgent()
    loop = CognitiveLoop(agent, llm=_WaitLLM())

    @loop.on_perceive
    def inject_relation_context(briefing: dict) -> None:
        briefing["relation_context"] = _relation_context()

    ctx = asyncio.run(loop.tick())

    assert "relation" in ctx.expectations
    key, expectation = next(iter(ctx.expectations["relation"].items()))
    assert ctx.briefing["relation_context"]["relation_id"] == RELATION_ID
    assert ctx.briefing["h0_relation_expectation"] == expectation
    assert f"relation_expectation:{key}" in agent.saved