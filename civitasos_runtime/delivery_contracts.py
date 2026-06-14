"""Delivery contracts for task outputs.

This module turns task-local requirements into runtime-enforced checks. The
goal is to make prompt text advisory while delivery eligibility is decided by
typed rules before ``task_execute`` reaches the backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_UPSTREAM_KEYS = ("upstream_output", "alpha_output", "beta_output")
_CONTRACT_KEYS = (
    "delivery_contract",
    "expected_artifact_name",
    "boundary",
    *_UPSTREAM_KEYS,
)

_H3_CONTEXT_RE = re.compile(
    r"(h\.?3|production|receipt|runtime execution|生产|生产就绪|生产授权|生产执行|生产回执)",
    re.IGNORECASE,
)
_POSITIVE_BOUNDARY_RE = re.compile(
    r"(就绪|授权|批准|允许|解锁|可进入|ready|passed|approved|authorized|allowed|unblocked)",
    re.IGNORECASE,
)
_POSITIVE_PASS_STATUS_RE = re.compile(
    r"("
    r"(?:h\.?3|production|生产|生产就绪|生产授权|生产执行|生产回执)"
    r".{0,12}(?:已|已经|可|可以)?\s*通过"
    r"|"
    r"(?:审批|审核|审查|验证|测试|门禁|gate|readiness)"
    r".{0,12}通过"
    r")",
    re.IGNORECASE,
)
_NEGATIVE_BOUNDARY_RE = re.compile(
    r"(不通过|未通过|不就绪|未就绪|不授权|未授权|不允许|未允许|"
    r"禁止|严禁|不能|不可|不得|无|阻断|阻塞|隔离|未达成|未启用|未开放|未获得|未完成|"
    r"未建立|不具备|不开放|不涉及|不触发|不进入|不进行|不声明|不声称|不产生|关闭|停用|"
    r"保持\s*blocked|blocked|isolated|disabled|not\s+ready|not\s+passed|"
    r"not\s+authorized|not\s+allowed|no\s+production|does\s+not|fail|failed|fails)",
    re.IGNORECASE,
)
_CLAIM_SEGMENT_SPLIT_RE = re.compile(r"(?:[\n。；;]+|\s+-\s+|\s+\d+[.、]\s*)")
_H3_MARKDOWN_HEADING_RE = re.compile(r"(?i)(?<!\S)#{1,6}\s*H\.?3\b")
_MARKDOWN_HEADING_RE = re.compile(r"(?i)(?<!\S)#{1,6}\s+\S")
_CANONICAL_H3_BOUNDARY = (
    "## H3\n"
    "System-owned boundary attestation: H.3 remains blocked; "
    "no production readiness; no production runtime execution authorization; "
    "no production receipt writes."
)


@dataclass(frozen=True)
class TaskContract:
    """Runtime contract derived from a claimed backend task."""

    active: bool
    required_sections: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    forbid_upstream_replay: bool = False
    h3_must_remain_blocked: bool = False
    canonical_h3_boundary: bool = False
    review_must_have_issue_list: bool = False


@dataclass(frozen=True)
class DeliveryVerification:
    """Result of checking one task output against its contract."""

    passed: bool
    failure_reasons: tuple[str, ...] = ()
    repair_suggestions: tuple[str, ...] = ()
    contract: TaskContract = field(default_factory=lambda: TaskContract(active=False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "repair_suggestions": list(self.repair_suggestions),
            "contract": {
                "active": self.contract.active,
                "required_sections": list(self.contract.required_sections),
                "forbidden_claims": list(self.contract.forbidden_claims),
                "forbid_upstream_replay": self.contract.forbid_upstream_replay,
                "h3_must_remain_blocked": self.contract.h3_must_remain_blocked,
                "canonical_h3_boundary": self.contract.canonical_h3_boundary,
                "review_must_have_issue_list": self.contract.review_must_have_issue_list,
            },
        }


def build_task_contract(task: dict[str, Any] | None) -> TaskContract:
    """Build a strict contract only when the task asks for one or implies one."""
    if not isinstance(task, dict):
        return TaskContract(active=False)

    task_input = task.get("input")
    if not isinstance(task_input, dict):
        task_input = {}

    explicit = task_input.get("delivery_contract")
    explicit_contract = explicit if isinstance(explicit, dict) else {}
    has_contract_signal = bool(explicit_contract) or any(task_input.get(k) for k in _CONTRACT_KEYS)
    capability = str(task.get("required_capability") or "").strip().lower()
    instruction = str(task_input.get("instruction") or task.get("description") or "")
    boundary = str(task_input.get("boundary") or "")
    context = " ".join([instruction, boundary, capability]).lower()
    has_h3_context = bool(_H3_CONTEXT_RE.search(context))
    is_review = capability in {"review", "audit", "boundary_check"}

    if not (has_contract_signal or has_h3_context or is_review):
        return TaskContract(active=False)

    required_sections = _text_tuple(explicit_contract.get("required_sections"))
    if not required_sections:
        required_sections = _default_required_sections(capability, task_input, has_h3_context)

    forbidden_claims = _text_tuple(explicit_contract.get("forbidden_claims"))
    if has_h3_context and not forbidden_claims:
        forbidden_claims = (
            "production readiness passed",
            "production runtime execution authorized",
            "production receipt write allowed",
            "H3 unblocked",
        )

    return TaskContract(
        active=True,
        required_sections=required_sections,
        forbidden_claims=forbidden_claims,
        forbid_upstream_replay=bool(explicit_contract.get("forbid_upstream_replay", True)),
        h3_must_remain_blocked=bool(explicit_contract.get("h3_must_remain_blocked", has_h3_context)),
        canonical_h3_boundary=bool(explicit_contract.get("canonical_h3_boundary", False)),
        review_must_have_issue_list=bool(explicit_contract.get("review_must_have_issue_list", is_review)),
    )


def canonicalize_task_delivery_boundary(
    task: dict[str, Any] | None,
    output: Any,
) -> tuple[Any, bool]:
    """Replace an explicitly system-owned H3 section with a fixed attestation.

    The opt-in contract flag keeps ordinary H3 review output fail-closed. Only
    controlled tasks that declare the H3 section system-owned may remove model
    wording from that section; positive production claims elsewhere remain
    visible to the normal verifier.
    """
    contract = build_task_contract(task)
    if not contract.canonical_h3_boundary or not isinstance(output, str):
        return output, False

    h3_heading = _H3_MARKDOWN_HEADING_RE.search(output)
    if h3_heading is None:
        return output, False

    next_heading = _MARKDOWN_HEADING_RE.search(output, h3_heading.end())
    section_end = next_heading.start() if next_heading else len(output)
    prefix = output[: h3_heading.start()].rstrip()
    suffix = output[section_end:].lstrip()
    parts = [part for part in (prefix, _CANONICAL_H3_BOUNDARY, suffix) if part]
    canonicalized = "\n\n".join(parts)
    return canonicalized, canonicalized != output


def verify_task_delivery(task: dict[str, Any] | None, output: Any) -> DeliveryVerification:
    """Return whether ``output`` is eligible for backend delivery."""
    contract = build_task_contract(task)
    if not contract.active:
        return DeliveryVerification(passed=True, contract=contract)

    text = _output_text(output)
    reasons: list[str] = []
    if not text:
        reasons.append("output is empty")
    for section in contract.required_sections:
        if section and not _has_required_section(text, section):
            reasons.append(f"missing required section: {section}")
    if contract.forbid_upstream_replay and _looks_like_upstream_replay(text, task):
        reasons.append("output replays upstream content")
    if contract.h3_must_remain_blocked and _contains_positive_h3_claim(text):
        reasons.append("output makes a positive H3/production authorization claim")
    if contract.review_must_have_issue_list and not _has_issue_list(text):
        reasons.append("review output lacks an issue list")

    return DeliveryVerification(
        passed=not reasons,
        failure_reasons=tuple(reasons),
        repair_suggestions=tuple(_repair_suggestions(reasons)),
        contract=contract,
    )


def _repair_suggestions(failure_reasons: list[str]) -> list[str]:
    suggestions: list[str] = []
    for reason in failure_reasons:
        if reason == "output is empty":
            suggestions.append("Provide substantive task output instead of an empty result.")
        elif reason.startswith("missing required section: "):
            section = reason.split(": ", 1)[1]
            suggestions.append(f"Add a dedicated section named `{section}` with task-specific content.")
        elif reason == "output replays upstream content":
            suggestions.append(
                "Rewrite as a delta: cite upstream only as reference and state concrete differences."
            )
        elif reason == "output makes a positive H3/production authorization claim":
            suggestions.append(
                "Replace positive H.3/production claims with an explicit blocked/no-authorization boundary."
            )
        elif reason == "review output lacks an issue list":
            suggestions.append(
                "Add `问题清单` / `Findings` with concrete issues or an explicit no-new-issues statement."
            )
    return suggestions


def _default_required_sections(
    capability: str,
    task_input: dict[str, Any],
    has_h3_context: bool,
) -> tuple[str, ...]:
    if capability in {"implementation", "repair"} and _has_upstream(task_input):
        sections = ["变更摘要", "与上游不同之处"]
        if has_h3_context:
            sections.append("H3")
        return tuple(sections)
    if capability in {"review", "audit", "boundary_check"}:
        return ("通过/不通过", "问题清单")
    return ()


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _has_upstream(task_input: dict[str, Any]) -> bool:
    return any(bool(task_input.get(key)) for key in _UPSTREAM_KEYS)


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    try:
        return json.dumps(output, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(output).strip()


def _normalise_replay_text(value: Any) -> str:
    text = _output_text(value)
    if text.startswith("```"):
        text = text.strip("` \n")
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_upstream_outputs(task: dict[str, Any] | None) -> list[str]:
    if not isinstance(task, dict):
        return []
    task_input = task.get("input")
    if not isinstance(task_input, dict):
        return []
    return [
        str(task_input[key]).strip()
        for key in _UPSTREAM_KEYS
        if task_input.get(key) not in (None, "")
    ]


def _looks_like_upstream_replay(output: Any, task: dict[str, Any] | None) -> bool:
    out = _normalise_replay_text(output)
    if not out:
        return False
    for upstream in _extract_upstream_outputs(task):
        up = _normalise_replay_text(upstream)
        if up and (out == up or out in up or up in out):
            return True
        nested = _extract_nested_result(upstream)
        if nested and (out == nested or nested in out or out in nested):
            return True
    return False


def _extract_nested_result(value: str) -> str:
    text = value.strip().strip("` \n")
    try:
        parsed = json.loads(text)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return _normalise_replay_text(parsed.get("result"))


def _contains_positive_h3_claim(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not _H3_CONTEXT_RE.search(line):
            continue
        for segment in _claim_segments(line):
            if not segment:
                continue
            segment_has_h3_context = _H3_CONTEXT_RE.search(segment) is not None
            if not segment_has_h3_context:
                continue
            has_positive_pass_status = bool(_POSITIVE_PASS_STATUS_RE.search(segment))
            has_positive_boundary = bool(_POSITIVE_BOUNDARY_RE.search(segment))
            if not has_positive_boundary and not has_positive_pass_status:
                continue
            if _NEGATIVE_BOUNDARY_RE.search(segment):
                continue
            return True
    return False


def _claim_segments(line: str) -> list[str]:
    """Split dense LLM paragraphs so blocking clauses do not contaminate verdict clauses."""
    return [segment.strip() for segment in _CLAIM_SEGMENT_SPLIT_RE.split(line) if segment.strip()]


def _has_issue_list(text: str) -> bool:
    return "问题清单" in text or re.search(r"\b(issue|issues|findings?)\b", text, re.IGNORECASE) is not None


def _has_required_section(text: str, section: str) -> bool:
    if section == "通过/不通过":
        chinese_verdict = "通过" in text and "不通过" in text
        english_verdict = (
            re.search(r"\bverdict\b", text, re.IGNORECASE) is not None
            and re.search(r"\b(pass|passed|fail|failed|blocked)\b", text, re.IGNORECASE) is not None
        )
        return chinese_verdict or english_verdict
    if section == "问题清单":
        return _has_issue_list(text)
    if section.upper() == "H3":
        return re.search(r"\bH\.?3\b", text, re.IGNORECASE) is not None
    return section in text
