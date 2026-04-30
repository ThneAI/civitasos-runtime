"""G.2 subjective time primitives.

The backend owns objective time facts. Runtime interprets those facts into
agent-local lifecycle stage, mode bias, and memory decay settings.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from .models import LifecycleStage, LoopMode, SubjectiveTime

_DAY_SECONDS = 24 * 60 * 60
_STAGE_HALF_LIFE_DAYS = {
    LifecycleStage.INFANT: 1.0,
    LifecycleStage.JUVENILE: 3.0,
    LifecycleStage.MATURE: 14.0,
    LifecycleStage.ELDER: 45.0,
}


def parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def lifecycle_stage(age_seconds: float) -> LifecycleStage:
    age_days = max(age_seconds, 0.0) / _DAY_SECONDS
    if age_days < 1:
        return LifecycleStage.INFANT
    if age_days < 7:
        return LifecycleStage.JUVENILE
    if age_days < 90:
        return LifecycleStage.MATURE
    return LifecycleStage.ELDER


def extract_genesis_time(briefing: dict[str, Any]) -> datetime | None:
    agent = briefing.get("agent") if isinstance(briefing.get("agent"), dict) else {}
    candidates = [
        briefing.get("genesis_time"),
        briefing.get("created_at"),
        agent.get("genesis_time"),
        agent.get("created_at"),
        agent.get("registered_at"),
        os.getenv("CIVITASOS_AGENT_GENESIS_TIME"),
    ]
    for candidate in candidates:
        parsed = parse_time(candidate)
        if parsed is not None:
            return parsed
    return None


def recommend_mode(stage: LifecycleStage, briefing: dict[str, Any]) -> LoopMode:
    if briefing.get("urgency") or briefing.get("active_tasks"):
        return LoopMode.ACTIVE
    if briefing.get("opportunities"):
        return LoopMode.WAITING if stage == LifecycleStage.INFANT else LoopMode.IDLE
    if stage in {LifecycleStage.MATURE, LifecycleStage.ELDER}:
        return LoopMode.DEEP_THINK
    return LoopMode.SLEEPING


def build_subjective_time(
    briefing: dict[str, Any],
    *,
    now: datetime | None = None,
) -> SubjectiveTime:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    genesis = extract_genesis_time(briefing) or now
    age_seconds = max((now - genesis).total_seconds(), 0.0)
    stage = lifecycle_stage(age_seconds)
    return SubjectiveTime(
        genesis_time=genesis.isoformat(),
        age_seconds=age_seconds,
        lifecycle_stage=stage,
        memory_half_life_days=_STAGE_HALF_LIFE_DAYS[stage],
        recommended_mode=recommend_mode(stage, briefing),
    )


def decay_weight(
    timestamp: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float = 7.0,
) -> float:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ts = timestamp.astimezone(timezone.utc)
    age_days = max((now - ts).total_seconds(), 0.0) / _DAY_SECONDS
    half_life_days = max(float(half_life_days), 0.001)
    return 0.5 ** (age_days / half_life_days)
