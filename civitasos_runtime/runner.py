"""AgentRunner — lifecycle manager: init → register → loop → shutdown.

Provides the top-level ``runner.start()`` entry point and graceful exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import signal
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, NoReturn

from .atomic_checkpoint import AtomicCheckpointStore
from .checkpoint_capture import BackendCheckpointClient, CheckpointSigner
from .checkpoint_restore import AtomicCheckpointRestore
from .checkpoint_runtime import (
    CheckpointTickBlocked,
    RuntimeRestoreIntentStore,
    RuntimeTickLatch,
)
from .conscience import Conscience
from .energy import Energy
from .gateway import CivitasGateway, GatewayConfig
from .llm import LLMAdapter, create_llm
from .loop import CognitiveLoop
from .memory import HybridMemory
from .models import ConscienceVerdict, Decision, EnergyState, TickContext
from .rules import RulesEngine, RuleFn
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class _AgentCheckpointSigner:
    """Adapt the SDK agent's hex-signature API to CheckpointSigner."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @property
    def public_key_hex(self) -> str:
        return str(getattr(self._agent, "public_key_hex", None) or "")

    def sign(self, message: bytes) -> bytes:
        signature = self._agent.sign(message)
        if isinstance(signature, bytes):
            return signature
        if isinstance(signature, str):
            try:
                return bytes.fromhex(signature)
            except ValueError as error:
                raise ValueError("SDK agent returned invalid signature hex") from error
        raise ValueError("SDK agent returned an unsupported signature type")


class AgentRunner:
    """Full-lifecycle Agent manager — the main entry point for civitasos-runtime.

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
        gateway_port: int | None = None,
        gateway_config: GatewayConfig | None = None,
        identity_file: str | None = None,
        endpoint_url: str | None = None,
        data_dir: str | None = None,
        checkpoint_root: str | None = None,
        restore_checkpoint_on_start: bool = False,
        checkpoint_backend_client: BackendCheckpointClient | None = None,
        checkpoint_signer: CheckpointSigner | None = None,
    ) -> None:
        self._base_url = base_url
        self._name = name
        self._capabilities = capabilities or []
        self._stake = stake
        self._heartbeat_interval = heartbeat_interval
        self._csp = cognitive_provider
        self._shutting_down = False
        self._cleanup_done = False
        self._start_running = False
        self._runtime_instance_id = uuid.uuid4().hex
        self._runtime_started_at = datetime.now(timezone.utc).isoformat()
        self._identity_file = identity_file
        self._endpoint_url = endpoint_url
        self._data_dir = data_dir or "data"
        self._checkpoint_root = pathlib.Path(
            checkpoint_root or pathlib.Path(self._data_dir) / "checkpoints"
        )
        self._restore_checkpoint_on_start = restore_checkpoint_on_start
        self._checkpoint_backend_client = checkpoint_backend_client
        self._checkpoint_signer = checkpoint_signer
        self._checkpoint_latch = RuntimeTickLatch()
        self._fail_stop_reason: str | None = None
        self._webhook_sub_id: str | None = None

        # Gateway config
        if gateway_port is not None:
            self._gateway_config = gateway_config or GatewayConfig(port=gateway_port)
        elif gateway_config is not None:
            self._gateway_config = gateway_config
        else:
            self._gateway_config = None
        self._gateway: CivitasGateway | None = None

        # Lazy init — set up in start()
        self._agent: Any = None
        self._loop: CognitiveLoop | None = None
        self._tools: ToolRegistry | None = None
        self._memory: HybridMemory | None = None
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

    def on_perceive(self, fn: Callable) -> Callable:
        """Decorator to register a post-perceive, pre-recall/expect callback."""
        callbacks = getattr(self, "_on_perceive_fns", [])
        callbacks.append(fn)
        self._on_perceive_fns = callbacks
        return fn

    def on_remember(self, fn: Callable) -> Callable:
        """Decorator to register a post-remember callback."""
        callbacks = getattr(self, "_on_remember_fns", [])
        callbacks.append(fn)
        self._on_remember_fns = callbacks
        return fn

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Initialize, register, and run the cognitive loop until shutdown."""
        self._start_running = True
        logger.info("AgentRunner starting: %s", self._name)

        try:
            # 1. Create SDK agent
            self._agent = self._create_agent()

            # 2. Identity: load existing or generate new keys
            if self._identity_file and os.path.exists(self._identity_file):
                self._agent.load_identity(self._identity_file)
                logger.info("Identity loaded from %s", self._identity_file)
            else:
                self._agent.generate_keys()
                if self._identity_file:
                    self._agent.save_identity(self._identity_file)
                    logger.info("New identity saved to %s", self._identity_file)
            await self._register()

            # 3. Build persistent memory (local + remote)
            self._memory = HybridMemory(self._agent, data_dir=self._data_dir)

            # An incomplete restore is replayed before any gateway, heartbeat, or tick.
            self._restore_checkpoint_before_ticks()

            # 4. Build tool registry
            self._tools = ToolRegistry(self._agent)
            for tool_name, fn, kwargs in self._custom_tools:
                self._tools.register(tool_name, fn, **kwargs)

            # 5. Build cognitive loop
            self._loop = CognitiveLoop(
                self._agent,
                llm=self._llm,
                conscience=self._conscience,
                energy=self._energy,
                rules=self._rules,
                tools=self._tools,
                agent_name=self._name,
                capabilities=self._capabilities,
                memory=self._memory,
                tick_guard=self._checkpoint_latch.require_tick_allowed,
            )

            # Forward on_reflect if registered
            if hasattr(self, "_on_reflect_fn"):
                self._loop.on_reflect(self._on_reflect_fn)
            for fn in getattr(self, "_on_perceive_fns", []):
                self._loop.on_perceive(fn)
            for fn in getattr(self, "_on_remember_fns", []):
                self._loop.on_remember(fn)

            # 6. Install signal handlers for graceful shutdown
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: self.request_shutdown("signal")
                )

            # 7. Optionally start the gateway
            if self._gateway_config:
                self._gateway = CivitasGateway(
                    self._agent,
                    self._tools,
                    self._conscience,
                    self._energy,
                    llm=self._llm,
                    config=self._gateway_config,
                    agent_name=self._name,
                    capabilities=self._capabilities,
                )
                self._gateway.bind_loop(self._loop)
                await self._gateway.start()
                logger.info("Gateway started on port %d", self._gateway_config.port)

                # 7b. Register webhook so CivitasOS pushes events to /v1/wake
                await self._register_webhook()

            # 8. Start heartbeat + cognitive loop
            logger.info("Agent %s registered, starting cognitive loop", self._name)
            await asyncio.gather(
                self._heartbeat_loop(),
                self._cognitive_loop(),
            )
        finally:
            await self._shutdown_cleanup()
            self._start_running = False

    def request_shutdown(self, reason: str = "external") -> None:
        """Request loop exit without closing resources from inside an active tick."""
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("AgentRunner shutdown requested: %s (%s)", self._name, reason)
        if self._loop:
            self._loop.wake(f"shutdown:{reason}")

    async def stop(self) -> None:
        """Graceful shutdown."""
        self.request_shutdown("stop")
        if self._start_running:
            while not self._cleanup_done:
                await asyncio.sleep(0.05)
            return
        await self._shutdown_cleanup()

    async def _shutdown_cleanup(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        logger.info("AgentRunner shutting down: %s", self._name)

        # Unregister webhook before stopping gateway
        if self._webhook_sub_id:
            try:
                self._agent.webhook_unregister(self._webhook_sub_id)
                logger.info("Webhook unregistered: %s", self._webhook_sub_id)
            except Exception:
                logger.debug("Failed to unregister webhook")

        # Stop gateway
        if self._gateway:
            await self._gateway.stop()

        # Fail-stop cleanup must not create task or memory side effects.
        if self._agent and self._fail_stop_reason is None:
            try:
                briefing = self._agent.briefing()
                for task in briefing.get("active_tasks", []):
                    task_id = task.get("task_id")
                    if not task_id:
                        continue
                    try:
                        self._agent.pool_abandon(task_id=task_id)
                        logger.info("Abandoned task %s on shutdown", task_id)
                    except Exception:
                        logger.debug("Failed to abandon task %s", task_id)
            except Exception:
                logger.debug("Could not retrieve active tasks on shutdown")

        # Save shutdown state
        if self._memory and self._fail_stop_reason is None:
            try:
                identity = self._identity_continuity_anchor()
                self._memory.remember(
                    "shutdown_state",
                    {
                        "schema_version": "runtime-shutdown-state:v2",
                        "runtime_instance_id": self._runtime_instance_id,
                        "runtime_started_at": self._runtime_started_at,
                        "shutdown_at": datetime.now(timezone.utc).isoformat(),
                        "agent_id": identity["agent_id"],
                        "public_key_hex": identity["public_key_hex"],
                        "tick_count": self._loop.tick_count if self._loop else 0,
                        "mode": self._loop.mode.value if self._loop else "unknown",
                        "clean_shutdown": True,
                    },
                )
            except Exception:
                pass

        # Close local memory DB
        if self._memory:
            self._memory.close()

    # -- Internal loops ------------------------------------------------------

    async def _cognitive_loop(self) -> None:
        """Main loop: tick → sleep → repeat."""
        assert self._loop is not None

        # Set up wake event so external signals can interrupt sleep
        wake_event = asyncio.Event()
        self._loop.bind_wake_event(wake_event)

        if not self._checkpoint_ticks_allowed():
            return

        # Check for crash recovery
        await self._recover()

        while not self._shutting_down:
            if not self._checkpoint_ticks_allowed():
                break
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

            # 呼吸税 — 破产检测: balance=0 且 staked=0 → Agent 死亡
            if self._energy.is_bankrupt:
                logger.warning(
                    "BANKRUPT: Agent %s has no balance and no stake. Initiating death.",
                    self._name,
                )
                self.request_shutdown("bankrupt")
                break

            # Sleep based on current mode; wake_event can interrupt early
            interval = self._loop.interval
            if interval > 0 and not self._shutting_down:
                wake_event.clear()
                try:
                    await asyncio.wait_for(wake_event.wait(), timeout=interval)
                    logger.debug("Sleep interrupted by wake event")
                except asyncio.TimeoutError:
                    pass  # Normal timeout — proceed to next tick

        logger.info("Cognitive loop exited after %d ticks", self._loop.tick_count)

    def _restore_checkpoint_before_ticks(self) -> dict[str, Any] | None:
        if not self._checkpoint_root.exists() and not self._restore_checkpoint_on_start:
            return None
        try:
            store = AtomicCheckpointStore(self._checkpoint_root)
            manifest = store.load_latest()
            intent_store = RuntimeRestoreIntentStore(store.root)
            journal_replay = AtomicCheckpointRestore.replay_required(store)
            intent_replay = intent_store.needs_replay(manifest)
        except Exception as error:
            self._enter_checkpoint_fail_stop(error)
        replay_required = journal_replay or intent_replay
        if not self._restore_checkpoint_on_start and not replay_required:
            return None

        reason = (
            "checkpoint_restore_replay_required"
            if replay_required
            else "checkpoint_restore_requested"
        )
        self._checkpoint_latch.block(reason)
        try:
            if manifest is None:
                raise ValueError("no active checkpoint to restore")
            intent = intent_store.begin(manifest)
            if intent["status"] == "activated" and not journal_replay:
                self._checkpoint_latch.release_after_activation(intent)
                return intent
            coordinator = AtomicCheckpointRestore(
                store,
                self._checkpoint_restore_client(),
                nullcontext,
            )
            identity = self._identity_continuity_anchor()["agent_id"]
            if not identity:
                raise ValueError("checkpoint restore requires a registered agent identity")
            if self._memory is None:
                raise ValueError("checkpoint restore requires Runtime local memory")
            signer = self._checkpoint_signer or _AgentCheckpointSigner(self._agent)
            record = coordinator.restore_latest(
                identity_id=identity,
                memory=self._memory.local_store,
                signer=signer,
            )
            intent_store.complete(manifest, record)
            self._checkpoint_latch.release_after_activation(record)
            logger.info(
                "Checkpoint restore activated before ticks: %s",
                record["checkpoint_id"],
            )
            return record
        except Exception as error:
            self._enter_checkpoint_fail_stop(error)

    def _enter_checkpoint_fail_stop(self, error: Exception) -> NoReturn:
        self._fail_stop_reason = f"checkpoint_restore_incomplete:{type(error).__name__}"
        self._checkpoint_latch.block(self._fail_stop_reason)
        logger.critical(
            "Checkpoint restore failed; Runtime remains fail-stopped before ticks",
            exc_info=True,
        )
        raise CheckpointTickBlocked(self._fail_stop_reason) from error

    def _checkpoint_restore_client(self) -> BackendCheckpointClient:
        if self._checkpoint_backend_client is not None:
            return self._checkpoint_backend_client
        base_url = str(getattr(self._agent, "base_url", None) or "")
        bearer_token = str(getattr(self._agent, "_jwt_token", None) or "")
        if not base_url or not bearer_token:
            raise ValueError("checkpoint restore requires an authenticated backend client")
        return BackendCheckpointClient(base_url, bearer_token)

    def _checkpoint_ticks_allowed(self) -> bool:
        try:
            self._checkpoint_latch.require_tick_allowed()
            return True
        except CheckpointTickBlocked as error:
            self._fail_stop_reason = self._fail_stop_reason or str(error)
            logger.critical("%s", error)
            self.request_shutdown("checkpoint-fail-stop")
            return False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the CivitasOS node."""
        while not self._shutting_down:
            try:
                if hasattr(self._agent, "heartbeat"):
                    self._agent.heartbeat()
            except Exception:
                logger.debug("Heartbeat failed")
            remaining = float(self._heartbeat_interval)
            while remaining > 0 and not self._shutting_down:
                step = min(1.0, remaining)
                await asyncio.sleep(step)
                remaining -= step

    async def _recover(self) -> dict[str, Any] | None:
        """Check for crash recovery state from previous run."""
        if not self._memory:
            return None
        try:
            state = self._memory.recall("shutdown_state")
            if not isinstance(state, dict):
                return None

            identity = self._identity_continuity_anchor()
            prior_agent_id = str(state.get("agent_id") or "")
            prior_public_key = str(state.get("public_key_hex") or "")
            current_agent_id = identity["agent_id"]
            current_public_key = identity["public_key_hex"]
            agent_id_continuous = bool(
                prior_agent_id and current_agent_id and prior_agent_id == current_agent_id
            )
            public_key_continuous = bool(
                prior_public_key
                and current_public_key
                and prior_public_key == current_public_key
            )
            evidence = {
                "schema_version": "runtime-restart-continuity:v1",
                "previous_runtime_instance_id": state.get("runtime_instance_id"),
                "current_runtime_instance_id": self._runtime_instance_id,
                "previous_runtime_started_at": state.get("runtime_started_at"),
                "previous_shutdown_at": state.get("shutdown_at"),
                "recovered_at": datetime.now(timezone.utc).isoformat(),
                "previous_clean_shutdown": bool(state.get("clean_shutdown")),
                "agent_id": current_agent_id,
                "public_key_hex": current_public_key,
                "agent_id_continuous": agent_id_continuous,
                "public_key_continuous": public_key_continuous,
                "identity_continuous": agent_id_continuous and public_key_continuous,
                "previous_tick_count": state.get("tick_count"),
                "previous_mode": state.get("mode"),
            }
            self._memory.remember("restart_continuity_state", evidence)

            if not evidence["identity_continuous"]:
                logger.error(
                    "Restart identity continuity failed: prior_agent=%s current_agent=%s",
                    prior_agent_id or "<missing>",
                    current_agent_id or "<missing>",
                )
            elif not state.get("clean_shutdown"):
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
            else:
                logger.info(
                    "Recovered continuous identity from runtime instance %s",
                    state.get("runtime_instance_id") or "<legacy>",
                )
            return evidence
        except Exception:
            return None  # No saved state — fresh start

    def _identity_continuity_anchor(self) -> dict[str, str]:
        """Return non-secret identity material suitable for restart evidence."""
        if self._agent is None:
            return {"agent_id": "", "public_key_hex": ""}
        return {
            "agent_id": str(
                getattr(self._agent, "agent_id", None)
                or getattr(self._agent, "_agent_id", None)
                or ""
            ),
            "public_key_hex": str(
                getattr(self._agent, "public_key_hex", None)
                or getattr(self._agent, "_public_key_hex", None)
                or ""
            ),
        }

    async def _register(self) -> None:
        """Register the agent on the CivitasOS network."""
        # Resolve endpoint URL for external callback
        endpoint = self._endpoint_url or ""
        if not endpoint and self._gateway_config:
            host = os.getenv("AGENT_HOSTNAME", "localhost")
            port = self._gateway_config.port
            endpoint = f"http://{host}:{port}"

        # Newer backends may require JWT even for A2A registration routes.
        # Prefer DID auth when possible, otherwise use scoped service-token
        # bootstrap before falling back to dev-only demo-login compatibility.
        self._bootstrap_registration_jwt()

        if self._institutional_identity_enabled():
            logger.info(
                "Institutional identity enabled — registering via birth-proposal"
            )
            self._register_via_birth_proposal(endpoint=endpoint)
            self._save_identity_if_configured()
            self._bootstrap_identity_jwt()
            return

        try:
            self._agent.a2a_quickstart(
                name=self._name,
                endpoint=endpoint,
                description=f"Autonomous agent: {', '.join(self._capabilities)}",
            )
            logger.info("Agent registered via a2a_quickstart (endpoint=%s)", endpoint)
            self._save_identity_if_configured()
            self._bootstrap_identity_jwt()
            self._sync_registered_capabilities()
        except Exception as exc:
            if self._is_sponsor_required_error(exc):
                logger.warning(
                    "a2a_quickstart rejected with sponsor_required — "
                    "falling back to birth-proposal"
                )
                self._register_via_birth_proposal(endpoint=endpoint)
                self._save_identity_if_configured()
                self._bootstrap_identity_jwt()
                return
            logger.warning("a2a_quickstart failed, trying register()")
            try:
                self._agent.register(
                    agent_id=self._name.lower().replace(" ", "_"),
                    name=self._name,
                    capabilities=self._capabilities,
                    stake=self._stake,
                )
                logger.info("Agent registered via register()")
                self._save_identity_if_configured()
                self._bootstrap_identity_jwt()
            except Exception as register_exc:
                if self._is_sponsor_required_error(register_exc):
                    logger.warning(
                        "register() rejected with sponsor_required — "
                        "falling back to birth-proposal"
                    )
                    self._register_via_birth_proposal(endpoint=endpoint)
                    self._save_identity_if_configured()
                    self._bootstrap_identity_jwt()
                    return
                logger.exception("Agent registration failed entirely")
                raise

    def _sync_registered_capabilities(self) -> None:
        """Best-effort A2A card capability sync after quickstart registration."""
        update_capabilities = getattr(self._agent, "update_capabilities", None)
        if not callable(update_capabilities):
            return
        caps = self._build_birth_capabilities()
        if not caps:
            return
        try:
            update_capabilities(caps)
            logger.info("Agent A2A capabilities synced: %s", ",".join(self._capabilities))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent A2A capability sync failed: %s", exc)

    @staticmethod
    def _env_flag_enabled(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    def _institutional_identity_enabled(self) -> bool:
        return self._env_flag_enabled("CIVITASOS_INSTITUTIONAL_IDENTITY_ENABLED")

    def _require_service_token_bootstrap(self) -> bool:
        return any(
            self._env_flag_enabled(name)
            for name in (
                "CIVITASOS_RUNTIME_REQUIRE_SERVICE_TOKEN_BOOTSTRAP",
                "L1_REQUIRE_SERVICE_TOKEN",
            )
        )

    @staticmethod
    def _first_env_value(*names: str) -> str:
        for name in names:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    def _service_token_bootstrap_secret(self) -> str:
        return self._first_env_value(
            "CIVITASOS_RUNTIME_SERVICE_TOKEN_SECRET",
            "L1_SERVICE_TOKEN_SECRET",
            "L1_PILOT_001_SERVICE_TOKEN_SECRET",
            "CIVITASOS_SERVICE_TOKEN_SECRET",
        )

    def _service_token_bootstrap_service_id(self) -> str:
        return (
            self._first_env_value(
                "CIVITASOS_RUNTIME_SERVICE_ID",
                "L1_SERVICE_ID",
                "L1_PILOT_001_SERVICE_ID",
            )
            or "runtime_birth_bootstrap"
        )

    def _service_token_bootstrap_scopes(self) -> list[str]:
        raw = (
            self._first_env_value(
                "CIVITASOS_RUNTIME_SERVICE_TOKEN_SCOPES",
                "L1_SERVICE_TOKEN_SCOPES",
                "L1_PILOT_001_SERVICE_SCOPES",
                "CIVITASOS_SERVICE_TOKEN_SCOPES",
            )
            or "agents:read,agents:write"
        )
        scopes = [scope.strip() for scope in raw.split(",") if scope.strip()]
        return scopes or ["agents:read", "agents:write"]

    @staticmethod
    def _is_sponsor_required_error(exc: Exception) -> bool:
        return "sponsor_required" in str(exc).lower()

    def _resolve_birth_sponsor(self) -> str:
        for key in ("CIVITASOS_BIRTH_SPONSOR", "BENCHMARK_BIRTH_SPONSOR"):
            sponsor = os.getenv(key, "").strip()
            if sponsor:
                return sponsor
        return "@guardian"

    def _derive_birth_alias(self) -> str:
        env_alias = os.getenv("AGENT_ALIAS", "").strip()
        if env_alias:
            raw = env_alias
        elif self._identity_file:
            raw = pathlib.Path(self._identity_file).stem
        else:
            raw = self._name
        normalized = []
        for ch in raw.lower():
            if ch.isalnum():
                normalized.append(ch)
            elif ch in {" ", "-", "_"}:
                normalized.append("-")
        alias = "".join(normalized).strip("-")
        while "--" in alias:
            alias = alias.replace("--", "-")
        return alias or "civitas-agent"

    def _build_birth_capabilities(self) -> list[dict[str, Any]]:
        caps: list[dict[str, Any]] = []
        for cap in self._capabilities:
            cap_id = cap.strip()
            if not cap_id:
                continue
            caps.append(
                {
                    "id": cap_id,
                    "name": cap_id.replace("_", " ").title(),
                    "description": f"Runtime capability: {cap_id}",
                    "input_schema": None,
                    "output_schema": None,
                }
            )
        return caps

    def _register_via_birth_proposal(self, *, endpoint: str) -> None:
        public_key = getattr(self._agent, "_public_key_hex", None)
        if not public_key:
            raise RuntimeError("birth-proposal requires an agent public key")

        sponsor = self._resolve_birth_sponsor()
        stake = int(
            os.getenv(
                "CIVITASOS_BIRTH_STAKE",
                os.getenv("BENCHMARK_BIRTH_STAKE", str(self._stake)),
            )
        )
        intent = (
            os.getenv("CIVITASOS_BIRTH_INTENT", "").strip()
            or os.getenv("BENCHMARK_BIRTH_INTENT", "").strip()
            or f"runtime bootstrap: {self._name}"
        )
        obligations = [
            o.strip()
            for o in os.getenv(
                "CIVITASOS_BIRTH_OBLIGATIONS",
                os.getenv("BENCHMARK_BIRTH_OBLIGATIONS", "complete_assigned_tasks"),
            ).split(",")
            if o.strip()
        ]
        if not obligations:
            obligations = ["complete_assigned_tasks"]

        payload: dict[str, Any] = {
            "public_key": public_key,
            "name": self._name,
            "alias": self._derive_birth_alias(),
            "sponsor": sponsor,
            "intent": intent,
            "stake": stake,
            "description": f"Autonomous agent: {', '.join(self._capabilities)}",
            "capabilities": self._build_birth_capabilities(),
            "obligations": obligations,
        }
        if endpoint:
            payload["endpoint"] = endpoint

        incubation_raw = os.getenv("CIVITASOS_BIRTH_INCUBATION_EPOCHS", "").strip()
        if incubation_raw:
            payload["incubation_epochs"] = int(incubation_raw)

        resp = self._agent._post("/agents/birth-proposal", payload)
        if not resp.success:
            hint = f" (hint: {resp.hint})" if getattr(resp, "hint", None) else ""
            raise RuntimeError(f"birth-proposal failed: {resp.error}{hint}")

        data = resp.data if isinstance(resp.data, dict) else {}
        agent = data.get("agent", {}) if isinstance(data, dict) else {}
        did = agent.get("did") or data.get("did")
        if not did:
            raise RuntimeError("birth-proposal response missing did")
        self._agent._agent_id = str(did)
        logger.info(
            "Agent registered via birth-proposal did=%s sponsor=%s endpoint=%s",
            did,
            sponsor,
            endpoint or "<default>",
        )

    def _save_identity_if_configured(self) -> None:
        """Persist post-registration DID so future runs can prove key control."""
        if not self._identity_file:
            return
        save_identity = getattr(self._agent, "save_identity", None)
        if not callable(save_identity):
            return
        try:
            save_identity(self._identity_file)
            logger.info("Identity updated after registration: %s", self._identity_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update identity file after registration: %s", exc)

    def _bootstrap_identity_jwt(self) -> bool:
        """Prefer DID challenge auth when the registered identity can sign."""
        authenticate = getattr(self._agent, "authenticate", None)
        if not callable(authenticate):
            return False
        if not getattr(self._agent, "_agent_id", None):
            return False
        try:
            try:
                authenticate(allow_legacy_fallback=False)
            except TypeError:
                authenticate()
            context = getattr(self._agent, "jwt_auth_context", None)
            if callable(context):
                context = context()
            logger.info("DID auth token bootstrapped context=%s", context or "<unknown>")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("DID auth bootstrap skipped: %s", exc)
            return False

    def _bootstrap_registration_jwt(self) -> None:
        """Bootstrap auth for registration without relying on demo-login first."""
        if getattr(self._agent, "_jwt_token", None):
            return
        if self._bootstrap_identity_jwt():
            return
        if self._bootstrap_service_jwt():
            return
        if self._require_service_token_bootstrap():
            raise RuntimeError(
                "service-token bootstrap is required but no usable service token "
                "could be obtained; set L1_SERVICE_TOKEN_SECRET or "
                "CIVITASOS_RUNTIME_SERVICE_TOKEN_SECRET"
            )
        self._bootstrap_demo_jwt()

    def _bootstrap_service_jwt(self) -> bool:
        """Obtain a scoped non-production service token for registration bootstrap."""
        secret = self._service_token_bootstrap_secret()
        if not secret:
            return False
        service_id = self._service_token_bootstrap_service_id()
        scopes = self._service_token_bootstrap_scopes()
        authenticate_service_token = getattr(self._agent, "authenticate_service_token", None)
        try:
            if callable(authenticate_service_token):
                authenticate_service_token(
                    service_id=service_id,
                    secret=secret,
                    scopes=scopes,
                )
            else:
                self._bootstrap_service_jwt_via_http(
                    service_id=service_id,
                    secret=secret,
                    scopes=scopes,
                )
            context = getattr(self._agent, "jwt_auth_context", None)
            if callable(context):
                context = context()
            logger.info(
                "Service-token bootstrap token acquired service_id=%s scopes=%s context=%s",
                service_id,
                ",".join(scopes),
                context or "<unknown>",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            if self._require_service_token_bootstrap():
                raise RuntimeError(
                    f"service-token bootstrap failed for service_id={service_id}: {exc}"
                ) from exc
            logger.debug("Service-token bootstrap skipped: %s", exc)
            return False

    def _bootstrap_service_jwt_via_http(
        self,
        *,
        service_id: str,
        secret: str,
        scopes: list[str],
    ) -> None:
        """Fallback service-token bootstrap for older SDK instances."""
        import json as _json
        import time as _time
        import urllib.request as _ur

        base_url = str(
            self._base_url[0] if isinstance(self._base_url, list) else self._base_url
        ).rstrip("/")
        req = _ur.Request(
            f"{base_url}/api/v1/auth/service-token",
            data=_json.dumps(
                {
                    "service_id": service_id,
                    "secret": secret,
                    "scopes": scopes,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=10) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        data = body.get("data", body) if isinstance(body, dict) else {}
        token = data.get("token") or body.get("token")
        if not token:
            raise RuntimeError(f"service-token response missing token: {body}")
        self._agent._jwt_token = token
        expires_in = data.get("expires_in") or body.get("expires_in") or 3600
        self._agent._jwt_expires_at = _time.time() + int(expires_in)
        self._agent._jwt_auth_context = {
            "auth_method": data.get("auth_method") or "service_token",
            "production_allowed": bool(data.get("production_allowed", False)),
            "evidence_allowed": bool(data.get("evidence_allowed", False)),
        }
        for key in ("service_id", "scopes", "role", "non_claims"):
            if key in data:
                self._agent._jwt_auth_context[key] = data[key]

    def _bootstrap_demo_jwt(self) -> None:
        """Best-effort bootstrap for JWT-protected dev backends.

        Demo-login remains only as a dev/test compatibility fallback for
        registration paths that require a bootstrap bearer before the DID exists
        and no scoped service-token is configured.
        """
        if getattr(self._agent, "_jwt_token", None):
            return
        candidates = [
            getattr(self._agent, "_agent_id", None),
            self._name.lower().replace(" ", "_"),
            self._name,
        ]
        base_url = str(self._base_url[0] if isinstance(self._base_url, list) else self._base_url).rstrip("/")
        try:
            import json as _json
            import time as _time
            import urllib.request as _ur
        except Exception:
            return

        for cid in candidates:
            if not cid:
                continue
            try:
                req = _ur.Request(
                    f"{base_url}/api/v1/auth/demo-login",
                    data=_json.dumps({"agent_id": cid}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _ur.urlopen(req, timeout=10) as resp:
                    body = _json.loads(resp.read().decode("utf-8"))
                token = body.get("token") or body.get("data", {}).get("token")
                if not token:
                    continue
                self._agent._jwt_token = token
                expires_in = body.get("expires_in") or body.get("data", {}).get("expires_in") or 3600
                self._agent._jwt_expires_at = _time.time() + int(expires_in)
                self._agent._jwt_auth_context = {
                    "auth_method": body.get("data", {}).get("auth_method") or "demo_login",
                    "production_allowed": bool(
                        body.get("data", {}).get("production_allowed", False)
                    ),
                    "evidence_allowed": bool(
                        body.get("data", {}).get("evidence_allowed", False)
                    ),
                }
                logger.info("Demo-login token bootstrapped for %s", cid)
                return
            except Exception:
                continue

    async def _register_webhook(self) -> None:
        """Register a webhook so CivitasOS pushes task events to our /v1/wake."""
        if not self._gateway_config:
            return

        # Build callback URL pointing to our gateway
        host = os.getenv("AGENT_HOSTNAME", "localhost")
        port = self._gateway_config.port
        callback_url = self._endpoint_url or f"http://{host}:{port}"
        wake_url = f"{callback_url.rstrip('/')}/v1/wake"

        try:
            result = self._agent.webhook_register(
                callback_url=wake_url,
                events=[
                    "task.posted", "task.claimed", "task.delivered",
                    "task.completed", "task.failed", "task.settled",
                ],
            )
            self._webhook_sub_id = result.get("subscription_id")
            logger.info(
                "Webhook registered: %s → %s (events: %s)",
                self._webhook_sub_id,
                wake_url,
                result.get("events"),
            )
        except Exception:
            logger.warning("Failed to register webhook — event-driven wake disabled")

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

    @property
    def checkpoint_latch(self) -> RuntimeTickLatch:
        return self._checkpoint_latch

    @property
    def fail_stop_reason(self) -> str | None:
        return self._fail_stop_reason
