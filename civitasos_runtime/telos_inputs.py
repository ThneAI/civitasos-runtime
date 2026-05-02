"""Input extraction helpers for H.1 Telos alignment."""
from __future__ import annotations

from typing import Any


def first_task(briefing: dict[str, Any]) -> dict[str, Any]:
    for key in ("active_tasks", "pool_tasks", "tasks", "opportunities"):
        value = briefing.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    return item
    task = briefing.get("task") or briefing.get("benchmark_task")
    return task if isinstance(task, dict) else {}


def extract_explicit_telos(
    briefing: dict[str, Any],
    task: dict[str, Any],
) -> tuple[str, str]:
    for source, source_mapping in (("task.telos", task), ("briefing.telos", briefing)):
        value = task_text(source_mapping, "telos", "objective", "desired_outcome", "result_telos")
        if value:
            return value, source
    metadata = briefing.get("metadata")
    if isinstance(metadata, dict):
        value = task_text(metadata, "telos", "objective", "desired_outcome")
        if value:
            return value, "metadata.telos"
    task_input = task.get("input") if isinstance(task, dict) else None
    if isinstance(task_input, dict):
        value = task_text(task_input, "telos", "objective", "desired_outcome")
        if value:
            return value, "task.input.telos"
    return "", ""


def task_text(source_mapping: dict[str, Any], *keys: str) -> str:
    if not isinstance(source_mapping, dict):
        return ""
    for key in keys:
        value = source_mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def task_list(source_mapping: dict[str, Any], key: str) -> list[str]:
    if not isinstance(source_mapping, dict):
        return []
    value = source_mapping.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def list_from_mapping(source_mapping: dict[str, Any], key: str) -> list[str]:
    value = source_mapping.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def looks_risky(briefing: dict[str, Any], task: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in (
            briefing.get("briefing"),
            briefing.get("description"),
            task.get("briefing") if isinstance(task, dict) else "",
            task.get("description") if isinstance(task, dict) else "",
        )
        if value
    ).lower()
    return any(
        token in text
        for token in ("transfer", "payment", "支付", "转账", "delete", "mutation")
    )


def find_verification_level(payload: Any) -> str:
    if isinstance(payload, dict):
        value = payload.get("verification_level")
        if isinstance(value, str) and value in {"baseline", "elevated", "strict"}:
            return value
        for child in payload.values():
            found = find_verification_level(child)
            if found:
                return found
    if isinstance(payload, list):
        for child in payload:
            found = find_verification_level(child)
            if found:
                return found
    return ""


def has_hint(value: str, hints: tuple[str, ...]) -> bool:
    return any(hint in value for hint in hints)


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}