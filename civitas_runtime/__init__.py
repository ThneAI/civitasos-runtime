"""civitas-runtime — CivitasOS Agent Runtime, distilled from CivitasOS social mechanisms.

    from civitas_runtime import AgentRunner

    runner = AgentRunner(
        base_url="http://node1:8099",
        name="WorkerBot",
        capabilities=["translation"],
        llm="openai:gpt-4o",
    )
    await runner.start()

For embedding mode (single tick):

    from civitas_runtime import CognitiveLoop, Conscience, Energy, ToolRegistry
"""

from .conscience import Conscience
from .energy import Energy
from .llm import (
    AnthropicAdapter,
    LiteLLMAdapter,
    LLMAdapter,
    OpenAIAdapter,
    create_llm,
)
from .loop import CognitiveLoop
from .models import (
    ConscienceVerdict,
    Decision,
    DecisionSource,
    EnergyState,
    Evaluation,
    LLMResponse,
    LoopMode,
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
    # Core modules
    "CognitiveLoop",
    "Conscience",
    "Energy",
    "RulesEngine",
    "ToolRegistry",
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
    "LoopMode",
    "TickPhase",
    "ToolDef",
    "ToolCall",
    "LLMResponse",
]
