"""Data models for civitas-runtime — distilled from CivitasOS ontology."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecisionSource(str, Enum):
    RULES = "rules"
    LLM = "llm"
    HYBRID = "hybrid"


class TickPhase(str, Enum):
    PERCEIVE = "perceive"
    RECALL = "recall"
    DECIDE = "decide"
    CONSCIENCE = "conscience"
    ACT = "act"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    REMEMBER = "remember"


class LoopMode(str, Enum):
    ACTIVE = "active"         # 5-15s interval
    IDLE = "idle"             # 30-60s
    SLEEPING = "sleeping"     # 5-10min
    EVENT = "event"           # instant wakeup


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """Output of the Decide phase."""
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    source: DecisionSource = DecisionSource.RULES


@dataclass
class Evaluation:
    """Output of the Evaluate phase."""
    success: bool
    outcome: Any = None
    cost: float = 0.0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class ConscienceVerdict:
    """Result of a Conscience check on a Decision."""
    allowed: bool
    reason: str
    suggestion: str = ""


@dataclass
class EnergyState:
    """Snapshot of the Agent's economic energy — distilled from CivitasOS economics."""
    balance: float = 1000.0
    staked: float = 100.0
    risk_score: float = 0.0
    gas_base_fee: float = 1.0
    potential: float = 50.0
    balance_cap: float = 10000.0
    reputation: float = 0.5


@dataclass
class TickContext:
    """Per-tick data flowing through the cognitive loop."""
    tick_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    briefing: dict[str, Any] = field(default_factory=dict)
    memories: dict[str, Any] = field(default_factory=dict)
    decision: Decision | None = None
    conscience_verdict: ConscienceVerdict | None = None
    action_result: Any = None
    evaluation: Evaluation | None = None
    reflection: str | None = None
    phase: TickPhase = TickPhase.PERCEIVE


# ---------------------------------------------------------------------------
# Tool definition model
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """Schema for a tool available to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    requires_conscience: bool = False
    estimated_cost: float = 0.0


# ---------------------------------------------------------------------------
# LLM response models
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
