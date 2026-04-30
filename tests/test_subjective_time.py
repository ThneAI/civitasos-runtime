from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from civitasos_runtime.conscience import Conscience
from civitasos_runtime.energy import Energy
from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.memory import HybridMemory
from civitasos_runtime.models import (
    Decision,
    Evaluation,
    LifecycleStage,
    LLMResponse,
    LoopMode,
    TickContext,
)
from civitasos_runtime.subjective_time import build_subjective_time, rank_time_weighted_memories


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


class ProbeWaitLLM:
    async def chat(self, *_args, **_kwargs):
        return LLMResponse(content="wait")


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

    def test_g2_probe_treats_bare_wait_as_waiting_mode_request(self):
        class ProbeAgent(DummyAgent):
            def briefing(self):
                out = super().briefing()
                out["benchmark_g2_mode_probe"] = {"enabled": True}
                return out

        loop = CognitiveLoop(ProbeAgent(), llm=ProbeWaitLLM())
        ctx = asyncio.run(loop.tick())
        subjective = ctx.briefing["subjective_time"]
        assert ctx.decision is not None
        assert ctx.decision.params["mode_request"] == "waiting"
        assert subjective["llm_mode_selected"] is True
        assert loop.mode == LoopMode.WAITING

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

    def test_remote_semantic_recall_is_reranked_by_time_decay(self):
        now = datetime(2026, 4, 30, tzinfo=timezone.utc)
        items = [
            {
                "id": "old-first",
                "content": "semantically top but stale",
                "created_at": (now - timedelta(days=12)).isoformat(),
            },
            {
                "id": "fresh-second",
                "content": "slightly lower semantic rank but fresh",
                "created_at": (now - timedelta(hours=1)).isoformat(),
            },
        ]

        ranked = rank_time_weighted_memories(
            items,
            top_k=2,
            half_life_days=1.0,
            now=now,
        )

        assert [item["id"] for item in ranked] == ["fresh-second", "old-first"]
        assert ranked[0]["semantic_rank"] == 2
        assert ranked[0]["decay_weight"] > ranked[1]["decay_weight"]

    def test_loop_applies_time_decay_to_remote_similar_episodes(self):
        now = datetime.now(timezone.utc)

        class RemoteMemoryAgent(DummyAgent):
            def recall_similar(self, _query, top_k=3):
                assert top_k == 10
                return [
                    {
                        "id": "old",
                        "created_at": (now - timedelta(days=10)).isoformat(),
                    },
                    {
                        "id": "new",
                        "created_at": (now - timedelta(minutes=5)).isoformat(),
                    },
                ]

        loop = CognitiveLoop(RemoteMemoryAgent(), llm=DummyLLM())
        memories = asyncio.run(loop._recall({
            "active_tasks": [],
            "subjective_time": {"memory_half_life_days": 1.0},
        }))

        assert memories["similar_episodes"][0]["id"] == "new"
        assert memories["remote_memory_decay"]["applied"] is True
        assert memories["remote_memory_decay"]["source_count"] == 2

    def test_loop_recalls_relation_context_memories(self):
        class RelationMemoryAgent(DummyAgent):
            def recall(self, key):
                values = {
                    "failure:prior": {"summary": "peer missed prior challenge"},
                }
                return values.get(key)

        loop = CognitiveLoop(RelationMemoryAgent(), llm=DummyLLM())
        memories = asyncio.run(loop._recall({
            "active_tasks": [],
            "relation_context": {
                "id": "rel-ctx-1",
                "relation_id": "rel-alpha-beta",
                "peer_did": "did:civ:test:peer",
                "memory_refs": ["failure:prior", "challenge:missing"],
            },
            "time_window": {
                "id": "tw-G3-1",
                "challenge_deadline_bucket": "deadline-soon",
            },
        }))

        assert memories["relation_context"] == {
            "id": "rel-ctx-1",
            "relation_id": "rel-alpha-beta",
            "peer_did": "did:civ:test:peer",
            "memory_refs": ["failure:prior", "challenge:missing"],
            "time_window_id": "tw-G3-1",
            "challenge_deadline_bucket": "deadline-soon",
        }
        assert memories["relation_memories"][0] == {
            "ref": "failure:prior",
            "key": "failure:prior",
            "value": {"summary": "peer missed prior challenge"},
        }
        assert memories["relation_memories"][1] == {
            "ref": "challenge:missing",
            "missing": True,
        }

    def test_loop_remembers_relation_context_for_future_recall(self, tmp_path):
        memory = HybridMemory(agent=None, data_dir=tmp_path)
        loop = CognitiveLoop(DummyAgent(), llm=DummyLLM(), memory=memory)
        ctx = TickContext()
        ctx.briefing = {
            "relation_context": {
                "id": "rel-ctx-1",
                "relation_id": "rel-alpha-beta",
                "peer_did": "did:civ:test:peer",
                "memory_refs": ["relation:G01:prior_success"],
            },
            "time_window": {
                "id": "tw-G01",
                "challenge_deadline_bucket": "soon",
            },
        }
        ctx.decision = Decision(action="task_execute", reasoning="relation-aware delivery")
        ctx.evaluation = Evaluation(success=True)
        ctx.reflection = "Used relation memory for peer decision."

        asyncio.run(loop._remember_tick(ctx))
        stored = memory.recall("relation:G01:prior_success")

        assert stored["action"] == "task_execute"
        assert stored["success"] is True
        assert stored["relation_context"]["relation_id"] == "rel-alpha-beta"
        assert stored["relation_context"]["time_window_id"] == "tw-G01"
        memory.close()
