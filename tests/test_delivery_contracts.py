from __future__ import annotations

from civitasos_runtime.delivery_contracts import build_task_contract, verify_task_delivery
from civitasos_runtime.loop import (
    CognitiveLoop,
    _fallback_output_text,
    _repair_contract_output_if_safe,
)
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


def _planning_task() -> dict:
    return {
        "task_id": "task-alpha",
        "required_capability": "planning",
        "input": {
            "instruction": "Create the L1 pilot plan. H3 remains blocked.",
            "expected_artifact_name": "alpha_l1_pilot_plan.md",
            "boundary": "H3 remains blocked; no production authorization.",
            "delivery_contract": {
                "required_sections": ["任务边界", "执行计划", "H3"],
                "forbid_upstream_replay": False,
                "h3_must_remain_blocked": True,
            },
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
    assert any("Rewrite as a delta" in suggestion for suggestion in replay.repair_suggestions)

    missing = verify_task_delivery(_implementation_task(), "H3 remains blocked.")

    assert missing.passed is False
    assert "missing required section: 变更摘要" in missing.failure_reasons
    assert "missing required section: 与上游不同之处" in missing.failure_reasons
    assert "Add a dedicated section named `变更摘要` with task-specific content." in missing.repair_suggestions

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
    assert any("blocked/no-authorization boundary" in suggestion for suggestion in report.repair_suggestions)


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


def test_h3_contract_accepts_blocked_state_with_architecture_review_wording() -> None:
    report = verify_task_delivery(
        _implementation_task(),
        (
            "## 变更摘要\n"
            "任务边界限定于受控环境内部执行。执行计划包含安全审计：通过第三方红队渗透测试。\n"
            "## 与上游不同之处\n"
            "新增明确的阻断边界，不开放生产执行。\n"
            "## H3\n"
            "H.3 模块保持永久阻断状态：未达成生产就绪标准；未启用运行时执行权限；"
            "未开放接收写入接口；通过架构评审委员会决议保持阻断。"
        ),
    )

    assert report.passed is True


def test_h3_contract_accepts_chinese_explicit_prohibition_wording() -> None:
    report = verify_task_delivery(
        _planning_task(),
        (
            "## 任务边界\n"
            "L1 受控试点仅限内部测试环境运行。该试点不涉及任何生产环境部署，"
            "不得进行外部数据 mutation 操作，不得触发任何生产就绪性验证流程。\n"
            "## 执行计划\n"
            "1. 采用双因素认证机制确保操作可追溯性。\n"
            "## H3\n"
            "H.3 模块保持阻断状态；无生产就绪性；无生产运行时执行权限；无生产接收写入能力。"
        ),
    )

    assert report.passed is True


def test_h3_contract_ignores_unrelated_pass_status_in_dense_single_line_output() -> None:
    report = verify_task_delivery(
        _planning_task(),
        (
            "## 任务边界 L1 受控试点只处理内部测试数据。 "
            "## 执行计划 完成输入校验，通过分步验证确认内部结果。 "
            "## H3 H.3 remains blocked; no production readiness; "
            "no production runtime execution; no production receipt writes."
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


def test_contract_fallback_output_is_delivery_contract_shaped() -> None:
    output = _fallback_output_text("", "task-1", _implementation_task())
    report = verify_task_delivery(_implementation_task(), output)

    assert "auto-deliver" not in output
    assert "## 变更摘要" in output
    assert "## 与上游不同之处" in output
    assert "## H3" in output
    assert report.passed is True


def test_contract_fallback_output_covers_explicit_planning_sections() -> None:
    output = _fallback_output_text("alpha_l1_pilot_plan.md", "task-alpha", _planning_task())
    report = verify_task_delivery(_planning_task(), output)

    assert "## 任务边界" in output
    assert "## 执行计划" in output
    assert "## H3" in output
    assert report.passed is True


def test_contract_fallback_omits_raw_unstructured_source_for_active_contracts() -> None:
    output = _fallback_output_text(
        "生产环境节点可以通过哈希验证进入执行路径。",
        "task-alpha",
        _planning_task(),
    )
    report = verify_task_delivery(_planning_task(), output)

    assert "生产环境节点可以通过哈希验证" not in output
    assert "raw content is omitted" in output
    assert report.passed is True


def test_review_contract_fallback_avoids_positive_authorization_terms() -> None:
    output = _fallback_output_text("", "task-2", _review_task())
    report = verify_task_delivery(_review_task(), output)

    assert "unsafe production authorization" not in output
    assert "不进入生产授权" not in output
    assert report.passed is True


def test_shape_only_contract_repair_replaces_thin_tool_output() -> None:
    repaired = _repair_contract_output_if_safe(
        "alpha_l1_pilot_plan.md",
        _planning_task(),
        "task-alpha",
    )

    assert repaired != "alpha_l1_pilot_plan.md"
    assert verify_task_delivery(_planning_task(), repaired).passed is True


def test_contract_repair_does_not_mask_positive_h3_claims() -> None:
    unsafe = "H3 已通过生产准入。"

    repaired = _repair_contract_output_if_safe(unsafe, _planning_task(), "task-alpha")
    report = verify_task_delivery(_planning_task(), repaired)

    assert repaired == unsafe
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
    assert ctx.decision.params["task_id"] == "task-1"
    assert "_repair_suggestions" in ctx.decision.params
    assert any("Rewrite as a delta" in suggestion for suggestion in ctx.decision.params["_repair_suggestions"])
    assert ctx.briefing["delivery_contract_violations"][0]["passed"] is False
    assert ctx.briefing["delivery_contract_violations"][0]["repair_suggestions"]
