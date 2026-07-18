from types import SimpleNamespace

import pytest

from civitasos_runtime.checkpoint_runtime import CheckpointTickBlocked
from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.memory import HybridMemory, LocalMemory
from civitasos_runtime.runner import AgentRunner

from test_checkpoint_adapters import IDENTITY_ID
from test_checkpoint_capture import _Signer
from test_checkpoint_restore import _BackendTransport, _active_checkpoint


class _RunnerAgent:
    def __init__(self, signer: _Signer) -> None:
        self.agent_id = IDENTITY_ID
        self.public_key_hex = signer.public_key_hex
        self.base_url = "https://backend.internal:8443"
        self._jwt_token = "owner-jwt"
        self._signer = signer
        self.briefing_calls = 0

    def sign(self, message: bytes) -> str:
        return self._signer.sign(message).hex()

    def generate_keys(self) -> str:
        return self.public_key_hex

    def briefing(self):
        self.briefing_calls += 1
        return {"active_tasks": []}


class _PreflightFailureTransport(_BackendTransport):
    def __init__(self, signer: _Signer) -> None:
        super().__init__(signer)
        self.fail_preflights = 1

    def __call__(self, request, timeout):  # noqa: ANN001
        if request.full_url.endswith("/restore/preflight") and self.fail_preflights:
            self.fail_preflights -= 1
            raise RuntimeError("injected preflight failure")
        return super().__call__(request, timeout)


class _StartProbeRunner(AgentRunner):
    def __init__(self, agent, **kwargs):  # noqa: ANN001
        self.probe_agent = agent
        super().__init__(**kwargs)

    def _create_agent(self):
        return self.probe_agent

    async def _register(self) -> None:
        return None


def _runner(
    tmp_path,
    store,
    client,
    signer,
    *,
    restore_on_start,
    observer=None,
):  # noqa: ANN001
    runner = AgentRunner(
        llm=SimpleNamespace(),
        data_dir=str(tmp_path / "memory"),
        checkpoint_root=str(store.root),
        restore_checkpoint_on_start=restore_on_start,
        checkpoint_backend_client=client,
        checkpoint_signer=signer,
        checkpoint_restore_observer=observer,
    )
    runner._agent = _RunnerAgent(signer)
    runner._memory = HybridMemory(agent=None, data_dir=tmp_path / "memory")
    return runner


def test_runner_restores_before_ticks_and_releases_only_after_activation(tmp_path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])
    memory.close()
    runner = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=True,
    )

    record = runner._restore_checkpoint_before_ticks()

    assert record["status"] == "activated"
    assert runner.checkpoint_latch.blocked is False
    assert runner.checkpoint_latch.activation["checkpoint_id"] == record["checkpoint_id"]
    assert runner._memory.local_store.snapshot() == expected
    runner._memory.close()


def test_runner_emits_only_bound_restore_milestones_in_durable_order(tmp_path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer)
    store, memory, client, manifest = _active_checkpoint(tmp_path, transport)
    memory.close()
    events = []
    runner = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=True,
        observer=lambda milestone, event: events.append((milestone, event)),
    )

    runner._restore_checkpoint_before_ticks()

    assert [milestone for milestone, _event in events] == [
        "runtime_intent_durable",
        "backend_preflight_durable",
        "runtime_restore_journal_durable",
        "runtime_sqlite_committed",
        "runtime_applied_journal_durable",
        "backend_activation_durable",
        "runtime_activation_journal_durable",
        "runtime_intent_activated_durable",
        "runtime_tick_latch_released",
    ]
    allowed = {
        "schema_version",
        "milestone",
        "checkpoint_id",
        "manifest_hash",
        "sequence",
        "identity_id",
        "node_id",
        "status",
        "backend_status",
        "activation_fact_id",
    }
    assert all(set(event) <= allowed for _milestone, event in events)
    assert all(event["checkpoint_id"] == manifest["checkpoint_id"] for _, event in events)
    assert all("owner-jwt" not in str(event) for _, event in events)
    runner._memory.close()


@pytest.mark.asyncio
async def test_runner_failure_stays_latched_and_cleanup_has_no_side_effects(tmp_path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer, fail_activations=1)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])
    memory.close()
    runner = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=True,
    )

    with pytest.raises(CheckpointTickBlocked, match="checkpoint_restore_incomplete"):
        runner._restore_checkpoint_before_ticks()

    assert runner.checkpoint_latch.blocked is True
    assert runner.fail_stop_reason.startswith("checkpoint_restore_incomplete")
    assert runner._memory.local_store.snapshot() == expected
    await runner._shutdown_cleanup()
    assert runner._agent.briefing_calls == 0

    reopened = LocalMemory(tmp_path / "memory")
    assert reopened.snapshot() == expected
    assert reopened.get("shutdown_state") is None
    reopened.close()


@pytest.mark.asyncio
async def test_start_fails_before_loop_gateway_or_heartbeat_on_restore_error(tmp_path) -> None:
    signer = _Signer()
    transport = _PreflightFailureTransport(signer)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    memory.close()
    agent = _RunnerAgent(signer)
    runner = _StartProbeRunner(
        agent,
        llm=SimpleNamespace(),
        data_dir=str(tmp_path / "memory"),
        checkpoint_root=str(store.root),
        restore_checkpoint_on_start=True,
        checkpoint_backend_client=client,
        checkpoint_signer=signer,
    )

    with pytest.raises(CheckpointTickBlocked):
        await runner.start()

    assert runner.loop is None
    assert runner._gateway is None
    assert runner._cleanup_done is True
    assert runner.checkpoint_latch.blocked is True
    assert agent.briefing_calls == 0


def test_runner_restart_replays_incomplete_journal_without_restore_flag(tmp_path) -> None:
    signer = _Signer()
    transport = _BackendTransport(signer, fail_activations=1)
    store, memory, client, _manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])
    memory.close()
    first = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=True,
    )
    with pytest.raises(CheckpointTickBlocked):
        first._restore_checkpoint_before_ticks()
    first._memory.close()

    restarted = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=False,
    )
    record = restarted._restore_checkpoint_before_ticks()

    assert record["status"] == "activated"
    assert restarted.checkpoint_latch.blocked is False
    assert restarted._memory.local_store.snapshot() == expected
    restarted._memory.close()


def test_restore_intent_survives_failure_before_restore_journal(tmp_path) -> None:
    signer = _Signer()
    transport = _PreflightFailureTransport(signer)
    store, memory, client, manifest = _active_checkpoint(tmp_path, transport)
    expected = memory.snapshot()
    memory.put("lessons", ["changed"])
    memory.close()
    first = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=True,
    )
    with pytest.raises(CheckpointTickBlocked):
        first._restore_checkpoint_before_ticks()
    first._memory.close()

    journal = store.root / "restore_journal" / f"{manifest['checkpoint_id']}.json"
    intent = (
        store.root
        / "runtime_restore_intents"
        / f"{manifest['checkpoint_id']}.json"
    )
    assert journal.exists() is False
    assert intent.exists() is True

    restarted = _runner(
        tmp_path,
        store,
        client,
        signer,
        restore_on_start=False,
    )
    record = restarted._restore_checkpoint_before_ticks()

    assert record["status"] == "activated"
    assert restarted.checkpoint_latch.blocked is False
    assert restarted._memory.local_store.snapshot() == expected
    restarted._memory.close()


class _LatchProbeLoop:
    def __init__(self, runner: AgentRunner) -> None:
        self.runner = runner
        self.tick_count = 0
        self.interval = 0
        self.mode = SimpleNamespace(value="awake")
        self.wake_event = None

    def bind_wake_event(self, event):  # noqa: ANN001
        self.wake_event = event

    def wake(self, _reason):  # noqa: ANN001
        if self.wake_event is not None:
            self.wake_event.set()

    async def tick(self):
        self.tick_count += 1
        self.runner.checkpoint_latch.block("injected_after_first_tick")
        return SimpleNamespace(decision=None, evaluation=None)


@pytest.mark.asyncio
async def test_runner_checks_latch_before_every_tick() -> None:
    runner = AgentRunner(llm=SimpleNamespace())
    runner._loop = _LatchProbeLoop(runner)

    await runner._cognitive_loop()

    assert runner._loop.tick_count == 1
    assert runner._shutting_down is True
    assert runner.fail_stop_reason is not None


@pytest.mark.asyncio
async def test_cognitive_loop_cannot_bypass_runner_tick_latch() -> None:
    runner = AgentRunner(llm=SimpleNamespace())
    runner.checkpoint_latch.block("restore_not_activated")
    loop = CognitiveLoop(
        SimpleNamespace(),
        llm=SimpleNamespace(),
        tick_guard=runner.checkpoint_latch.require_tick_allowed,
    )

    with pytest.raises(CheckpointTickBlocked, match="restore_not_activated"):
        await loop.tick()

    assert loop.tick_count == 0
