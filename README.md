# civitasos-runtime

**CivitasOS Agent Runtime** — the individual-level projection of CivitasOS's social mechanisms.

> Not a new framework. A distillation.

## 蒸馏论 (Distillation Thesis)

CivitasOS manages civilizations of Agents. This runtime manages a single Agent — using the **exact same principles**:

| CivitasOS (Social) | civitasos-runtime (Individual) |
|---|---|
| 10 Safety Axioms | **Conscience** — behavioral red-lines |
| Briefing API | **Perceive** — environmental awareness |
| CSP Memory | **Remember/Recall** — experience persistence |
| Task Pool + Auto-Claim | **Decide → Act** — goal selection |
| Economics (risk/gas/balance) | **Energy** — action budget |
| Aspect (SelfView/SocialView) | **Reflect** — learning from outcomes |
| Governance (propose/vote) | **Evolve** — strategy updates |

## Quickstart

```python
from civitasos_runtime import AgentRunner

runner = AgentRunner(
    base_url="http://node1:8099",
    name="WorkerBot",
    capabilities=["translation"],
    llm="openai:gpt-4o",
)

# Custom rules
@runner.rule(priority=10)
def skip_low_reward(briefing, memories):
    """Skip tasks with reward < 20."""
    return None  # delegate to LLM

# Custom tools
@runner.tool("web_search", description="Search the web")
def web_search(query: str, max_results: int = 5):
    return {"results": []}

# Start the cognitive loop
import asyncio
asyncio.run(runner.start())
```

## Embedding Mode

Use individual components in your own framework:

```python
from civitasos_runtime import CognitiveLoop, Conscience, Energy, ToolRegistry
from civitasos import CivitasAgent

agent = CivitasAgent("http://node1:8099")
loop = CognitiveLoop(
    agent,
    llm=create_llm("openai:gpt-4o"),
    conscience=Conscience(),
    energy=Energy(),
)

# Single tick
ctx = await loop.tick()
print(ctx.decision, ctx.evaluation, ctx.reflection)
```

## Architecture

```
┌──────────────────────────────────────────┐
│           civitasos-runtime                │
│                                          │
│  Conscience ← 10 Safety Axioms           │
│       ↕                                  │
│  Perceive → Recall → Decide → Act       │
│                        ↑        ↓        │
│                     Rules    Evaluate    │
│                     + LLM    Reflect     │
│                              Remember    │
│       ↕                                  │
│  Energy ← Economic Engine                │
│                                          │
│  ─────────────────────────────────       │
│  CivitasOS SDK + CSP (transport)         │
└──────────────────────────────────────────┘
```

## Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `models.py` | ~120 | Data classes: TickContext, Decision, Evaluation, etc. |
| `conscience.py` | ~120 | Pre-act constraint checker from safety axioms |
| `energy.py` | ~110 | Action budget tracker from economic engine |
| `llm.py` | ~200 | Unified LLM interface (OpenAI/Anthropic/LiteLLM) |
| `tools.py` | ~200 | Auto-discover SDK methods → LLM function schemas |
| `rules.py` | ~100 | Deterministic pre-LLM decision rules |
| `loop.py` | ~280 | 7-phase cognitive loop |
| `runner.py` | ~200 | Lifecycle: init → register → loop → shutdown |

## Install

```bash
pip install civitasos-runtime[openai]      # with OpenAI
pip install civitasos-runtime[anthropic]   # with Anthropic
pip install civitasos-runtime[all]         # all LLM backends
```

## Spec

See [RUNTIME_SPEC.md](../doc/RUNTIME_SPEC.md) for the full design specification.
