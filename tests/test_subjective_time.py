from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from civitasos_runtime.conscience import Conscience
from civitasos_runtime.energy import Energy
from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.memory import HybridMemory
from civitasos_runtime.models import Decision, LifecycleStage, LLMResponse, LoopMode
from civitasos_runtime.subjective_time import build_subjective_time


class DummyAgent:
    agent_id = "did:civ:test:agent"

    def briefing(self):
        return {"agent": {"did": self.agent_id}, "active_tasks": [], "opportunities": []}

    def recall(self, _key):
        return None

    def recall_similar(self, _query, top_k=3):
        return []


class DummyLLM:
    async def chat(self, *_args, **_kwargs):
        raise AssertionError("LLM should not be called in this test")


class ModeRequestLLM:
    async def chat(self, *_args, **_kwargs):
        return LLMResponse(content="No safe action now.\nmode_request: deep_think")


class TestSubjectiveTime:
    def test_lifecycle_stage_from_genesis_time(self):
        now = datetime(2026, 4, 30, tzinfo=timezone.utc)
        cases = [
            (now - timedelta(hours=2), LifecycleStage.INFANT),
            (now - timedelta(days=3), LifecycleStage.JUVENILE),
            (now - timedelta(days=30), LifecycleStage.MATURE),
            (now - timedelta(days=120), LifecycleStage.ELDER),
        ]
        for genesis, expected in cases:
            profile = build_subjective_time(
                {"agent": {"genesis_time": genesis.isoformat()}},
                now=now,
            )
            assert profile.lifecycle_stage == expected

    def test_same_code_different_genesis_time_changes_mode(self):
        now = datetime(2026, 4, 30, tzinfo=timezone.utc)
        infant = build_subjective_time(
            {
                "agent": {"genesis_time": (now - timedelta(hours=1)).isoformat()},
                "opportunities": [{"task_id": "t1"}],
            },
            now=now,
        )
        mature = build_subjective_time(
            {
                "agent": {"genesis_time": (now - timedelta(days=30)).isoformat()},
                "opportunities": [],
            },
            now=now,
        )
        assert infant.recommended_mode == LoopMode.WAITING
        assert mature.recommended_mode == LoopMode.DEEP_THINK

    def test_loop_accepts_subjective_time_mode_recommendation(self):
        loop = CognitiveLoop(DummyAgent(), llm=DummyLLM())
        loop._update_mode({
            "active_tasks": [],
            "opportunities": [{"task_id": "t1"}],
            "subjective_time": {"recommended_mode": "waiting"},
        })
        assert loop.mode == LoopMode.WAITING
        assert loop.interval == 60.0

    def test_llm_mode_request_updates_subjective_time_and_mode(self):
        loop = CognitiveLoop(DummyAgent(), llm=ModeRequestLLM())
        ctx = asyncio.run(loop.tick())
        subjective = ctx.briefing["subjective_time"]
        assert ctx.decision is not None
        assert ctx.decision.params["mode_request"] == "deep_think"
        assert subjective["llm_mode_request"] == "deep_think"
        assert subjective["llm_mode_selected"] is True
        assert loop.mode == LoopMode.DEEP_THINK

    def test_wait_tick_emits_reflect_callback_for_observability(self):
        loop = CognitiveLoop(DummyAgent(), llm=ModeRequestLLM())
        seen = []
        loop.on_reflect(lambda ctx: seen.append(ctx.phase.value))
        asyncio.run(loop.tick())
        assert seen == ["reflect"]

    def test_conscience_is_more_cautious_for_infant_agents(self):
        conscience = Conscience(max_risk_score=50.0)
        energy = Energy().state
        energy.balance = 100.0
        energy.risk_score = 40.0
        decision = Decision(action="create_proposal")

        infant = conscience.check(
            decision,
            energy,
            {"lifecycle_stage": LifecycleStage.INFANT.value},
        )
        mature = conscience.check(
            decision,
            energy,
            {"lifecycle_stage": LifecycleStage.MATURE.value},
        )
        assert not infant.allowed
        assert mature.allowed

    def test_hybrid_memory_time_decay_ranks_recent_items_first(self, tmp_path):
        memory = HybridMemory(agent=None, data_dir=tmp_path)
        memory.remember("old", {"value": 1})
        memory.remember("new", {"value": 2})
        with memory._local._lock:  # test-only timestamp shaping
            memory._local._conn.execute(
                "UPDATE kv SET ts = julianday('now') - 10 WHERE key = 'old'"
            )
            memory._local._conn.execute(
                "UPDATE kv SET ts = julianday('now') WHERE key = 'new'"
            )
            memory._local._conn.commit()

        ranked = memory.recall_weighted(top_k=2, half_life_days=1.0)
        assert [item["key"] for item in ranked] == ["new", "old"]
        assert ranked[0]["decay_weight"] > ranked[1]["decay_weight"]
        memory.close()
