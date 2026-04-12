"""AgentRunner — lifecycle manager: init → register → loop → shutdown.

Provides the top-level ``runner.start()`` entry point and graceful exit.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any, Callable

from .conscience import Conscience
from .energy import Energy
from .llm import LLMAdapter, create_llm
from .loop import CognitiveLoop
from .models import ConscienceVerdict, Decision, EnergyState, TickContext
from .rules import RulesEngine, RuleFn
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRunner:
    """Full-lifecycle Agent manager — the main entry point for civitas-runtime.

    Usage::

        runner = AgentRunner(
            base_url="http://node1:8099",
            name="AlphaTrader",
            capabilities=["trading"],
            llm="openai:gpt-4o",
        )
        await runner.start()
    """

    def __init__(
        self,
        base_url: str | list[str] = "http://localhost:8099",
        name: str = "CivitasAgent",
        capabilities: list[str] | None = None,
        llm: str | LLMAdapter = "openai:gpt-4o",
        *,
        cognitive_provider: dict[str, Any] | None = None,
        stake: int = 100,
        heartbeat_interval: int = 60,
        llm_kwargs: dict[str, Any] | None = None,
        conscience_config: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url
        self._name = name
        self._capabilities = capabilities or []
        self._stake = stake
        self._heartbeat_interval = heartbeat_interval
        self._csp = cognitive_provider
        self._shutting_down = False

        # Lazy init — set up in start()
        self._agent: Any = None
        self._loop: CognitiveLoop | None = None
        self._tools: ToolRegistry | None = None
        self._rules = RulesEngine()
        self._conscience = Conscience(**(conscience_config or {}))
        self._energy = Energy()

        # LLM adapter
        if isinstance(llm, str):
            self._llm = create_llm(llm, **(llm_kwargs or {}))
        else:
            self._llm = llm

        # Custom tools (registered before start)
        self._custom_tools: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []

    # -- Decorator APIs (pre-start registration) ----------------------------

    def rule(self, priority: int = 50, name: str = "") -> Callable[[RuleFn], RuleFn]:
        """Decorator to register a custom decision rule."""
        return self._rules.rule(priority=priority, name=name)

    def tool(
        self,
        name: str,
        description: str = "",
        requires_conscience: bool = False,
        estimated_cost: float = 0.0,
    ) -> Callable:
        """Decorator to register a custom tool callable by the LLM."""
        def decorator(fn: Callable) -> Callable:
            self._custom_tools.append((name, fn, {
                "description": description,
                "requires_conscience": requires_conscience,
                "estimated_cost": estimated_cost,
            }))
            return fn
        return decorator

    def conscience_check(self, fn: Callable) -> Callable:
        """Decorator to register a custom conscience check."""
        self._conscience.add_check(fn)
        return fn

    def on_reflect(self, fn: Callable) -> Callable:
        """Decorator to register a post-reflect callback."""
        # Will be forwarded to loop after init
        self._on_reflect_fn = fn
        return fn

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Initialize, register, and run the cognitive loop until shutdown."""
        logger.info("AgentRunner starting: %s", self._name)

        # 1. Create SDK agent
        self._agent = self._create_agent()

        # 2. Generate keys + register
        self._agent.generate_keys()
        await self._register()

        # 3. Build tool registry
        self._tools = ToolRegistry(self._agent)
        for tool_name, fn, kwargs in self._custom_tools:
            self._tools.register(tool_name, fn, **kwargs)

        # 4. Build cognitive loop
        self._loop = CognitiveLoop(
            self._agent,
            llm=self._llm,
            conscience=self._conscience,
            energy=self._energy,
            rules=self._rules,
            tools=self._tools,
            agent_name=self._name,
            capabilities=self._capabilities,
        )

        # Forward on_reflect if registered
        if hasattr(self, "_on_reflect_fn"):
            self._loop.on_reflect(self._on_reflect_fn)

        # 5. Install signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.ensure_future(self.stop())
            )

        # 6. Start heartbeat + cognitive loop
        logger.info("Agent %s registered, starting cognitive loop", self._name)
        await asyncio.gather(
            self._heartbeat_loop(),
            self._cognitive_loop(),
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("AgentRunner shutting down: %s", self._name)

        # Save shutdown state
        if self._agent:
            try:
                self._agent.remember("shutdown_state", {
                    "tick_count": self._loop.tick_count if self._loop else 0,
                    "mode": self._loop.mode.value if self._loop else "unknown",
                    "clean_shutdown": True,
                })
            except Exception:
                pass

    # -- Internal loops ------------------------------------------------------

    async def _cognitive_loop(self) -> None:
        """Main loop: tick → sleep → repeat."""
        assert self._loop is not None

        # Check for crash recovery
        await self._recover()

        while not self._shutting_down:
            try:
                ctx = await self._loop.tick()
                logger.info(
                    "Tick #%d: action=%s success=%s mode=%s",
                    self._loop.tick_count,
                    ctx.decision.action if ctx.decision else "none",
                    ctx.evaluation.success if ctx.evaluation else "n/a",
                    self._loop.mode.value,
                )
            except Exception:
                logger.exception("Cognitive loop tick failed")

            # Sleep based on current mode
            interval = self._loop.interval
            if interval > 0 and not self._shutting_down:
                await asyncio.sleep(interval)

        logger.info("Cognitive loop exited after %d ticks", self._loop.tick_count)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the CivitasOS node."""
        while not self._shutting_down:
            try:
                if hasattr(self._agent, "heartbeat"):
                    self._agent.heartbeat()
            except Exception:
                logger.debug("Heartbeat failed")
            await asyncio.sleep(self._heartbeat_interval)

    async def _recover(self) -> None:
        """Check for crash recovery state from previous run."""
        if not self._agent:
            return
        try:
            state = self._agent.recall("shutdown_state")
            if state and not state.get("clean_shutdown"):
                logger.warning(
                    "Recovering from unclean shutdown (last tick_count=%s)",
                    state.get("tick_count"),
                )
                # Check active tasks
                try:
                    briefing = self._agent.briefing()
                    active = briefing.get("active_tasks", [])
                    if active:
                        logger.info(
                            "Found %d active tasks from previous run", len(active)
                        )
                except Exception:
                    pass
        except Exception:
            pass  # No saved state — fresh start

    async def _register(self) -> None:
        """Register the agent on the CivitasOS network."""
        try:
            self._agent.a2a_quickstart(
                name=self._name,
                endpoint="",
                description=f"Autonomous agent: {', '.join(self._capabilities)}",
            )
            logger.info("Agent registered via a2a_quickstart")
        except Exception:
            logger.warning("a2a_quickstart failed, trying register()")
            try:
                self._agent.register(
                    agent_id=self._name.lower().replace(" ", "_"),
                    name=self._name,
                    capabilities=self._capabilities,
                    stake=self._stake,
                )
                logger.info("Agent registered via register()")
            except Exception:
                logger.exception("Agent registration failed entirely")
                raise

    def _create_agent(self) -> Any:
        """Create the CivitasOS SDK agent instance."""
        try:
            from civitasos import CivitasAgent  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "CivitasOS SDK required: pip install civitasos"
            ) from exc

        kwargs: dict[str, Any] = {"base_url": self._base_url}
        if self._csp:
            kwargs["cognitive_provider"] = self._csp
        return CivitasAgent(**kwargs)

    # -- Accessors -----------------------------------------------------------

    @property
    def agent(self) -> Any:
        """The underlying CivitasAgent SDK instance."""
        return self._agent

    @property
    def loop(self) -> CognitiveLoop | None:
        return self._loop

    @property
    def tools(self) -> ToolRegistry | None:
        return self._tools

    @property
    def is_running(self) -> bool:
        return not self._shutting_down and self._loop is not None
