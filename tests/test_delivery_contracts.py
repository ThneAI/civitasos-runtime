from __future__ import annotations

from civitasos_runtime.delivery_contracts import build_task_contract, verify_task_delivery
from civitasos_runtime.loop import CognitiveLoop
from civitasos_runtime.models import Decision, LLMResponse, TickContext
from civitasos_runtime.tools import ToolRegistry


class _WaitLLM:
    async def chat(self, messages, tools=None, temperature=0.3):  # noqa: ANN001
        return LLMResponse(content="wait")


class _Agent:
    agent_id = "did:civ:devnet:test-agent"


def _implementation_task() -> dict:
    return {
        "task_id": "task-1",
        "required_capability": "implementation",
        "input": {
            "instruction": "Produce a Beta delta artifact. H3 remains blocked.",
            "alpha_output": '{"result": "Alpha plan text"}',
            "expected_artifact_name": "DELTA.md",
            "boundary": "H3 remains blocked; no production authorization.",
        },
    }


def _review_task() -> dict:
    return {
        "task_id": "task-2",
        "required_capability": "review",
        "input": {
            "instruction": "Review L1 vs production readiness.",
            "beta_output": "Beta artifact",
            "boundary": "H3 remains blocked; production readiness must fail.",
        },
    }


def test_contract_is_inactive_for_plain_tasks() -> None:
    contract = build_task_contract({"task_id": "plain", "input": {"description": "hello"}})

    assert contract.active is False
    assert verify_task_delivery({"task_id": "plain"}, "anything").passed is True


def test_implementation_contract_requires_delta_sections_and_blocks_replay() -> None:
    replay = verify_task_delivery(_implementation_task(), "Alpha plan text")

    assert replay.passed is False
    assert "output replays upstream content" in replay.failure_reasons

    missing = verify_task_delivery(_implementation_task(), "H3 remains blocked.")

    assert missing.passed is False
    assert "missing required section: 变更摘要" in missing.failure_reasons
    assert "missing required section: 与上游不同之处" in missing.failure_reasons

    ok = verify_task_delivery(
        _implementation_task(),
        "## 变更摘要\n生成 delta。\n## 与上游不同之处\n不是复述。\n## H.3\nH.3 remains blocked.",
    )

    assert ok.passed is True


def test_h3_contract_blocks_positive_production_authorization_claims() -> None:
    report = verify_task_delivery(
        _review_task(),
        "## 通过/不通过\nProduction readiness passed.\n## 问题清单\n- none",
    )

    assert report.passed is False
    assert "output makes a positive H3/production authorization claim" in report.failure_reasons


def test_h3_contract_does_not_treat_through_as_pass_status() -> None:
    report = verify_task_delivery(
        _implementation_task(),
        (
            "## 变更摘要\n"
            "通过增加边界说明，生产路径继续保持隔离。\n"
            "## 与上游不同之处\n"
            "不是复述。\n"
            "## H3\n"
            "H.3 remains blocked."
        ),
    )

    assert report.passed is True


def test_h3_contract_blocks_chinese_pass_status_claims() -> None:
    report = verify_task_delivery(
        _review_task(),
        "## 通过/不通过\nH3 已通过生产准入。\n## 问题清单\n- none",
    )

    assert report.passed is False
    assert "output makes a positive H3/production authorization claim" in report.failure_reasons


def test_review_contract_requires_issue_list() -> None:
    report = verify_task_delivery(
        _review_task(),
        "## 通过/不通过\nL1 通过，production readiness 不通过。",
    )

    assert report.passed is False
    assert "review output lacks an issue list" in report.failure_reasons


def test_review_contract_accepts_split_verdict_wording() -> None:
    report = verify_task_delivery(
        _review_task(),
        "## Verdict\nL1 controlled pilot passed. Production readiness failed.\n## Findings\n- H3 remains blocked.",
    )

    assert report.passed is True


def test_loop_rewrites_invalid_task_execute_to_pool_fail() -> None:
    loop = CognitiveLoop(_Agent(), llm=_WaitLLM(), tools=ToolRegistry())
    ctx = TickContext()
    ctx.briefing = {"active_tasks": [_implementation_task()]}
    ctx.decision = Decision(
        action="task_execute",
        params={"task_id": "task-1", "output": "Alpha plan text", "success": True},
    )

    loop._enforce_delivery_contract(ctx)

    assert ctx.decision.action == "pool_fail"
    assert ctx.decision.params == {"task_id": "task-1"}
    assert ctx.briefing["delivery_contract_violations"][0]["passed"] is False
