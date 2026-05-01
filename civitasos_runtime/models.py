"""Data models for civitasos-runtime — distilled from CivitasOS ontology."""

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
    EXPECT = "expect"
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
    WAITING = "waiting"       # cautious observation window
    DEEP_THINK = "deep_think" # reflective low-frequency cognition
    EVENT = "event"           # instant wakeup


class LifecycleStage(str, Enum):
    INFANT = "infant"
    JUVENILE = "juvenile"
    MATURE = "mature"
    ELDER = "elder"


class ExpectationDomain(str, Enum):
    SURVIVAL = "survival"
    ECONOMIC = "economic"
    REPUTATION = "reputation"
    RELATION = "relation"
    TASK = "task"
    GOVERNANCE = "governance"
    CONSTITUTIONAL = "constitutional"


class ExpectationStateKind(str, Enum):
    PREDICTED = "predicted"
    DESIRED = "desired"
    NORMATIVE = "normative"


class ExpectationUpdateRule(str, Enum):
    EMA = "EMA"
    PRECISION_WEIGHTED_DELTA = "precision_weighted_delta"
    SLOW_TRAIT_DRIFT = "slow_trait_drift"
    RELATION_MATRIX_UPDATE = "relation_matrix_update"
    DECAY = "decay"
    GOVERNANCE_TRIGGER = "governance_trigger"
    GOVERNED_REVISION = "governed_revision"


class ExpectationLifecycleState(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    VIOLATED = "violated"
    REVISED = "revised"
    DECAYED = "decayed"
    INSTITUTIONALIZED = "institutionalized"
    DEPRECATED = "deprecated"


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
class PendingThresholdChange:
    """Queued conscience threshold change awaiting governance approval (Fix 3: 治理门控)."""
    param_name: str
    current_value: float
    proposed_value: float
    reason: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EnergyState:
    """Snapshot of the Agent's economic energy — distilled from CivitasOS economics."""
    balance: float = 500.0
    staked: float = 0.0
    risk_score: float = 0.0
    gas_base_fee: float = 1.0
    potential: float = 50.0
    balance_cap: float = 10000.0
    reputation: float = 0.5
    # Fix 1+2: 观 + R2R data flowing into decision pipeline
    aspect_gap: float = 0.0
    peer_trust_avg: float = 0.5
    active_relations: int = 0


@dataclass
class SubjectiveTime:
    """Agent-local interpretation of objective genesis time."""
    genesis_time: str | None
    age_seconds: float
    lifecycle_stage: LifecycleStage
    memory_half_life_days: float
    recommended_mode: LoopMode


@dataclass
class ExpectationTrace:
    """A single reality/expectation comparison for H.0 observability."""
    domain: ExpectationDomain
    state_kind: ExpectationStateKind
    expected_value: float | None = None
    actual_value: float | None = None
    surprise_score: float | None = None
    precision: float | None = None
    valence: str = ""
    stake: float | None = None
    source_event_id: str = ""
    lifecycle_state: ExpectationLifecycleState = ExpectationLifecycleState.ACTIVE


@dataclass
class DriveTrace:
    """Drive is action pressure, not permission."""
    domain: ExpectationDomain
    drive_score: float | None = None
    actionability: float | None = None
    action_bias: str = ""
    constitution_verdict: str = ""


@dataclass
class ExpectationUpdate:
    """Auditable candidate or applied update for Identity-owned IEM state."""
    target: str
    parameter_name: str
    old_value: float | str | None = None
    new_value: float | str | None = None
    rule: ExpectationUpdateRule = ExpectationUpdateRule.PRECISION_WEIGHTED_DELTA
    reason_event: str = ""
    update_params: dict[str, Any] = field(default_factory=dict)
    constitution_verdict: str = ""
    local_update_blocked: bool = False


@dataclass
class RelationExpectationVector:
    """Minimal directed relation expectation vector for H.0-B."""
    expected_trust: float = 0.72
    expected_delivery_quality: float = 0.72
    expected_cooperation: float = 0.70
    expected_betrayal_risk: float = 0.12
    expected_repair_probability: float = 0.55
    precision: float = 0.35


@dataclass
class DirectedRelationExpectation:
    """Expectation owned by one identity about one counterparty relation edge."""
    relation_id: str
    from_identity: str
    to_identity: str
    state_kind: ExpectationStateKind = ExpectationStateKind.PREDICTED
    expectation: RelationExpectationVector = field(default_factory=RelationExpectationVector)
    source_event_ids: list[str] = field(default_factory=list)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TickContext:
    """Per-tick data flowing through the cognitive loop."""
    tick_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    briefing: dict[str, Any] = field(default_factory=dict)
    memories: dict[str, Any] = field(default_factory=dict)
    expectations: dict[str, Any] = field(default_factory=dict)
    surprise: dict[str, Any] = field(default_factory=dict)
    drive: dict[str, Any] = field(default_factory=dict)
    action_bias: dict[str, Any] = field(default_factory=dict)
    expectation_updates: list[ExpectationUpdate] = field(default_factory=list)
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
