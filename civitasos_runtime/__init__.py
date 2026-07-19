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
from .atomic_checkpoint import AtomicCheckpointStore
from .checkpoint_models import DomainSnapshot, REQUIRED_DOMAINS
from .checkpoint_adapters import (
    backend_projection_snapshots,
    capture_runtime_snapshots,
    identity_snapshot,
    restore_runtime_snapshots,
)
from .checkpoint_capture import (
    AtomicCheckpointCapture,
    BackendCheckpointClient,
    verify_signer_possession,
)
from .checkpoint_restore import (
    AtomicCheckpointRestore,
    CheckpointRestoreIncomplete,
)
from .checkpoint_observer import (
    CheckpointRestoreMilestone,
    CheckpointRestoreObserver,
    emit_checkpoint_restore_milestone,
)
from .checkpoint_runtime import (
    CheckpointTickBlocked,
    RuntimeRestoreIntentStore,
    RuntimeTickLatch,
)
from .energy import Energy
from .gateway import CallRecord, CivitasGateway, GatewayConfig, Ledger
from .iem_anchor import build_iem_anchor, genesis_iem_state, iem_state_hash, iem_update_log_hash
from .identity_expectation import apply_identity_expectation_traces, apply_iem_updates_to_state
from .llm import (
    AnthropicAdapter,
    LiteLLMAdapter,
    LLMAdapter,
    OpenAIAdapter,
    create_llm,
)
from .loop import CognitiveLoop
from .memory import HybridMemory, LocalMemory
from .mentorship import (
    AdviceProvider,
    BackendMentorshipAdviceProvider,
    BackendMentorshipClient,
)
from .models import (
    ConscienceVerdict,
    Decision,
    DecisionSource,
    EnergyState,
    Evaluation,
    IEMVersionAnchor,
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
    "AtomicCheckpointStore",
    "DomainSnapshot",
    "REQUIRED_DOMAINS",
    "capture_runtime_snapshots",
    "backend_projection_snapshots",
    "identity_snapshot",
    "restore_runtime_snapshots",
    "AtomicCheckpointCapture",
    "BackendCheckpointClient",
    "verify_signer_possession",
    "AtomicCheckpointRestore",
    "CheckpointRestoreIncomplete",
    "CheckpointRestoreMilestone",
    "CheckpointRestoreObserver",
    "emit_checkpoint_restore_milestone",
    "CheckpointTickBlocked",
    "RuntimeRestoreIntentStore",
    "RuntimeTickLatch",
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
    "AdviceProvider",
    "BackendMentorshipAdviceProvider",
    "BackendMentorshipClient",
    "build_iem_anchor",
    "genesis_iem_state",
    "iem_state_hash",
    "iem_update_log_hash",
    "apply_identity_expectation_traces",
    "apply_iem_updates_to_state",
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
    "IEMVersionAnchor",
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
