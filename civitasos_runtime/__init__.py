"""civitasos-runtime — CivitasOS Agent Runtime, distilled from CivitasOS social mechanisms.

    from civitasos_runtime import AgentRunner

    runner = AgentRunner(
        base_url="http://node1:8099",
        name="WorkerBot",
        capabilities=["translation"],
        llm="openai:gpt-4o",
    )
    await runner.start()

For embedding mode (single tick):

    from civitasos_runtime import CognitiveLoop, Conscience, Energy, ToolRegistry
"""

from .conscience import Conscience
from .energy import Energy
from .gateway import CallRecord, CivitasGateway, GatewayConfig, Ledger
from .llm import (
    AnthropicAdapter,
    LiteLLMAdapter,
    LLMAdapter,
    OpenAIAdapter,
    create_llm,
)
from .loop import CognitiveLoop
from .memory import HybridMemory, LocalMemory
from .models import (
    ConscienceVerdict,
    Decision,
    DecisionSource,
    EnergyState,
    Evaluation,
    LifecycleStage,
    LLMResponse,
    LoopMode,
    PendingThresholdChange,
    DirectedRelationExpectation,
    SubjectiveTime,
    RelationExpectationVector,
    TickContext,
    TickPhase,
    ToolCall,
    ToolDef,
)
from .rules import RulesEngine
from .runner import AgentRunner
from .tools import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    # Main entry point
    "AgentRunner",
    # Gateway
    "CivitasGateway",
    "GatewayConfig",
    "CallRecord",
    "Ledger",
    # Core modules
    "CognitiveLoop",
    "Conscience",
    "Energy",
    "RulesEngine",
    "ToolRegistry",
    # Memory
    "HybridMemory",
    "LocalMemory",
    # LLM
    "LLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LiteLLMAdapter",
    "create_llm",
    # Data models
    "TickContext",
    "Decision",
    "DecisionSource",
    "Evaluation",
    "ConscienceVerdict",
    "EnergyState",
    "PendingThresholdChange",
    "DirectedRelationExpectation",
    "RelationExpectationVector",
    "LifecycleStage",
    "SubjectiveTime",
    "LoopMode",
    "TickPhase",
    "ToolDef",
    "ToolCall",
    "LLMResponse",
]
