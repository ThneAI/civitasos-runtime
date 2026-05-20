from civitasos_runtime.loop import (
    _active_task_focus_block,
    _fallback_output_text,
    _looks_like_upstream_replay,
    _repair_replay_output,
)
from civitasos_runtime.tools import ToolRegistry


class _FakeAgent:
    def webhook_register(self, callback_url: str, events: list[str] | None = None) -> dict:
        return {}

    def webhook_unregister(self, subscription_id: str) -> dict:
        return {}

    def ask_guardian(self, action: str) -> dict:
        return {"action": action}

    def pool_discover(
        self,
        capability: str | None = None,
        capabilities: list[str] | None = None,
        min_reputation: float = 0.0,
    ) -> list[dict]:
        return [{
            "capability": capability,
            "capabilities": capabilities,
            "min_reputation": min_reputation,
        }]

    def task_execute(self, task_id: str, output: str, success: bool = True) -> dict:
        return {}


def test_webhook_lifecycle_methods_are_not_exposed_to_llm() -> None:
    registry = ToolRegistry(_FakeAgent())

    tool_names = {
        item["function"]["name"]
        for item in registry.to_openai_tools()
    }

    assert "task_execute" in tool_names
    assert "webhook_register" not in tool_names
    assert "webhook_unregister" not in tool_names
    assert "ask_guardian" not in tool_names


def test_task_execute_result_alias_is_normalized_to_output() -> None:
    agent = _FakeAgent()

    params = ToolRegistry._filter_params(
        agent.task_execute,
        {
            "task_id": "task-1",
            "result": "implementation delta",
            "worker_agent": "ignored",
        },
    )

    assert params == {
        "task_id": "task-1",
        "output": "implementation delta",
    }


def test_array_schema_and_string_args_are_normalized() -> None:
    agent = _FakeAgent()
    registry = ToolRegistry(agent)
    pool_discover = next(
        item for item in registry.to_openai_tools()
        if item["function"]["name"] == "pool_discover"
    )

    assert pool_discover["function"]["parameters"]["properties"]["capabilities"]["type"] == "array"

    assert ToolRegistry._filter_params(
        agent.pool_discover,
        {"capabilities": "implementation, documentation, repair"},
    )["capabilities"] == ["implementation", "documentation", "repair"]
    assert ToolRegistry._filter_params(
        agent.pool_discover,
        {"capabilities": '["review","boundary_check","audit"]'},
    )["capabilities"] == ["review", "boundary_check", "audit"]


def test_active_task_replay_guard_detects_nested_upstream_result() -> None:
    task = {
        "task_id": "task-1",
        "required_capability": "implementation",
        "input": {
            "instruction": "Produce a beta delta artifact.",
            "alpha_output": '{"result": "Alpha plan text"}',
            "expected_artifact_name": "DELTA.md",
        },
    }

    assert _looks_like_upstream_replay("Alpha plan text", task)
    assert "forbidden_replay_sources" in _active_task_focus_block([task])

    repaired = _repair_replay_output(task, "task-1")

    assert "DELTA.md" in repaired
    assert "Alpha plan text" not in repaired


def test_task_text_fallback_keeps_artifact_length() -> None:
    text = "x" * 1200

    assert _fallback_output_text(text, "task-1") == text
