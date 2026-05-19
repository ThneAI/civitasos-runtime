"""CivitasGateway — dual-mode access to CivitasOS.

Two modes serving different callers, both economically metered:

Autonomous Mode (POST /v1/tools/{name}):
    For humans & expert agents who understand CivitasOS.
    Fine-grained, per-tool access. Each call:
    auth → balance check → conscience → execute → meter.

Delegate Mode (POST /v1/delegate):
    For external agents without CivitasOS domain knowledge.
    Submit natural-language intent → Runtime reasons → executes → returns.
    Higher cost (includes LLM reasoning surcharge).

Economic Model:
    Every call is authenticated by caller DID and metered.
    Costs are tracked in a Ledger and settled periodically
    via the CivitasOS economic system (task settlement or transfer).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .conscience import Conscience
from .energy import Energy
from .llm import LLMAdapter
from .models import Decision, DecisionSource, Evaluation, TickContext
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Ledger
# ---------------------------------------------------------------------------

@dataclass
class GatewayConfig:
    """Configuration for the CivitasGateway server."""
    host: str = "0.0.0.0"
    port: int = 8300
    delegate_surcharge: float = 2.0   # extra gas for LLM reasoning in delegate mode
    max_delegate_steps: int = 10      # max sequential actions per delegate request
    require_auth: bool = True         # enforce X-Civitas-DID header
    wake_callback_secret: str | None = None
    wake_callback_issuer: str = "civitasos-backend"
    wake_signature_tolerance_secs: int = 300


@dataclass
class CallRecord:
    """Single metered call for economic settlement."""
    caller_did: str
    mode: str            # "autonomous" | "delegate"
    action: str
    gas_cost: float
    timestamp: float
    success: bool
    tick_id: str = ""


class Ledger:
    """In-memory call ledger — tracks gas owed per caller for settlement."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._debts: dict[str, float] = {}  # did → total owed

    def record(self, rec: CallRecord) -> None:
        self._records.append(rec)
        self._debts[rec.caller_did] = self._debts.get(rec.caller_did, 0) + rec.gas_cost

    def debt(self, did: str) -> float:
        return self._debts.get(did, 0)

    def settle(self, did: str) -> float:
        """Mark a DID as settled. Returns amount cleared."""
        return self._debts.pop(did, 0)

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": len(self._records),
            "total_gas": sum(r.gas_cost for r in self._records),
            "outstanding": dict(self._debts),
            "recent": [
                {
                    "caller": r.caller_did[:20] + "...",
                    "mode": r.mode,
                    "action": r.action,
                    "gas": r.gas_cost,
                    "ok": r.success,
                }
                for r in self._records[-20:]
            ],
        }


def build_wake_signature(
    secret: str,
    timestamp: str,
    raw_body: bytes,
    *,
    issuer: str = "civitasos-backend",
) -> str:
    """Build backend-compatible HMAC signature for wake callbacks."""
    signed_payload = (
        issuer.encode("utf-8")
        + b"."
        + timestamp.encode("utf-8")
        + b"."
        + raw_body
    )
    digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def validate_wake_signature(
    secret: str | None,
    headers: Any,
    raw_body: bytes,
    *,
    tolerance_secs: int = 300,
    expected_issuer: str = "civitasos-backend",
    now: float | None = None,
) -> tuple[bool, str]:
    """Validate optional backend wake callback HMAC.

    If no secret is configured, wake callbacks remain compatibility-accepted.
    When a secret is configured, missing, stale, or mismatched signatures fail.
    """
    if not secret:
        return True, "signature_not_required"

    issuer = headers.get("X-Civitas-Webhook-Issuer", "")
    timestamp = headers.get("X-Civitas-Webhook-Timestamp", "")
    signature = headers.get("X-Civitas-Webhook-Signature", "")
    if not issuer or not timestamp or not signature:
        return False, "missing_signature_headers"
    if issuer != expected_issuer:
        return False, "invalid_signature_issuer"
    try:
        observed = int(timestamp)
    except ValueError:
        return False, "invalid_signature_timestamp"
    current = int(now if now is not None else time.time())
    if abs(current - observed) > tolerance_secs:
        return False, "stale_signature_timestamp"

    expected = build_wake_signature(secret, timestamp, raw_body, issuer=issuer)
    if not hmac.compare_digest(signature, expected):
        return False, "signature_mismatch"
    return True, "signature_valid"


def build_wake_audit_context(headers: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Extract stable audit metadata from a wake callback without trusting it as auth."""
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    task_id = body.get("task_id") or data.get("task_id")
    return {
        "schema_version": "civitasos-wake-audit-context:v1",
        "issuer": headers.get("X-Civitas-Webhook-Issuer", ""),
        "event": body.get("event", "unknown"),
        "task_id": task_id,
        "subscription_id": body.get("subscription_id"),
        "signature_present": bool(headers.get("X-Civitas-Webhook-Signature", "")),
    }


def build_wake_event_record(headers: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Build a structured event record for the next cognitive tick."""
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    audit = build_wake_audit_context(headers, body)
    return {
        "schema_version": "civitasos-wake-event-record:v1",
        "event": audit["event"],
        "task_id": audit["task_id"],
        "agent_id": body.get("agent_id") or data.get("agent_id"),
        "subscription_id": audit["subscription_id"],
        "issuer": audit["issuer"],
        "backend_timestamp": body.get("timestamp"),
        "signature_present": audit["signature_present"],
        "data": data,
    }


# ---------------------------------------------------------------------------
# Delegate system prompt
# ---------------------------------------------------------------------------

_DELEGATE_TEMPLATE = """\
你是 CivitasOS 的领域专家。一个外部 Agent 委托你执行任务。

你的身份:
- Runtime 专家: {name}
- 信誉: {reputation:.2f}
- 可用余额: {balance:.1f} CIV

你拥有 {tool_count} 个 CivitasOS 工具，涵盖:
- 任务池: 发布、发现、认领、完成任务
- 社交网络: R2R关系、信任发现、信号发送
- 经济: 质押、转账、gas 市场
- 治理: 提案、投票
- 记忆: 存储、回忆、遗忘

委托方预算: {budget:.1f} CIV

准则:
1. 精准执行委托意图，不做多余操作
2. 选择成本最低的有效方案
3. 如果意图不明确或无法执行，回复 "wait" 并说明原因
4. 你的良知约束依然生效 — 不执行违反安全公理的操作
"""


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class CivitasGateway:
    """Dual-mode HTTP gateway for CivitasOS.

    Usage::

        gateway = CivitasGateway(agent, tools, conscience, energy, llm=llm)
        await gateway.start()   # blocks, serving on :8300
    """

    def __init__(
        self,
        agent: Any,
        tools: ToolRegistry,
        conscience: Conscience,
        energy: Energy,
        llm: LLMAdapter | None = None,
        *,
        config: GatewayConfig | None = None,
        agent_name: str = "",
        capabilities: list[str] | None = None,
    ) -> None:
        self._agent = agent
        self._tools = tools
        self._conscience = conscience
        self._energy = energy
        self._llm = llm
        self._config = config or GatewayConfig()
        if self._config.wake_callback_secret is None:
            self._config.wake_callback_secret = (
                os.getenv("CIVITASOS_WAKE_CALLBACK_SECRET", "").strip() or None
            )
        self._ledger = Ledger()
        self._name = agent_name or "CivitasRuntime"
        self._capabilities = capabilities or []
        self._cognitive_loop: Any = None  # bound via bind_loop()

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    def bind_loop(self, loop: Any) -> None:
        """Bind the CognitiveLoop so /v1/wake can call loop.wake()."""
        self._cognitive_loop = loop

    # -- Server lifecycle ---------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP gateway (blocks until shutdown)."""
        try:
            from aiohttp import web
        except ImportError as exc:
            raise ImportError(
                "Gateway requires aiohttp: pip install 'civitasos-runtime[gateway]'"
            ) from exc

        app = web.Application()
        app.router.add_get("/v1/tools", self._handle_list_tools)
        app.router.add_post("/v1/tools/{name}", self._handle_tool_call)
        app.router.add_post("/v1/delegate", self._handle_delegate)
        app.router.add_post("/", self._handle_wake)
        app.router.add_post("/v1/wake", self._handle_wake)
        app.router.add_get("/v1/status", self._handle_status)
        app.router.add_get("/v1/ledger", self._handle_ledger)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        logger.info(
            "CivitasGateway listening on %s:%d  [autonomous + delegate]",
            self._config.host,
            self._config.port,
        )
        # Keep running (caller owns the event loop)
        self._runner = runner

    async def stop(self) -> None:
        if hasattr(self, "_runner"):
            await self._runner.cleanup()

    # -- Auth ---------------------------------------------------------------

    def _authenticate(self, request: Any) -> str:
        """Extract caller DID from request header."""
        did = request.headers.get("X-Civitas-DID", "")
        if self._config.require_auth and not did:
            from aiohttp import web
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "Missing X-Civitas-DID header"}),
                content_type="application/json",
            )
        return did or "anonymous"

    def _get_caller_balance(self, did: str) -> float:
        """Query caller's CivitasOS balance."""
        try:
            account = self._agent.get_account(did)
            return float(account.get("balance", 0))
        except Exception:
            # Fallback: can't verify — allow but log
            logger.debug("Could not query balance for %s", did)
            return float("inf")

    # ======================================================================
    # WEBHOOK WAKE — event-driven activation
    # ======================================================================

    async def _handle_wake(self, request: Any) -> Any:
        """POST /v1/wake — receive webhook from CivitasOS, wake the agent.

        Body (from CivitasOS dispatch_a2a_webhook)::

            {
                "event": "task.posted",
                "subscription_id": "a2a-wh-...",
                "agent_id": "...",
                "timestamp": "...",
                "data": { ... }
            }
        """
        from aiohttp import web

        raw_body = await request.read()
        ok, reason = validate_wake_signature(
            self._config.wake_callback_secret,
            request.headers,
            raw_body,
            tolerance_secs=self._config.wake_signature_tolerance_secs,
            expected_issuer=self._config.wake_callback_issuer,
        )
        if not ok:
            logger.warning("WAKE rejected: %s", reason)
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "invalid wake signature", "reason": reason}),
                content_type="application/json",
            )

        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            body = {}

        event_type = body.get("event", "unknown")
        audit_context = build_wake_audit_context(request.headers, body)
        event_record = build_wake_event_record(request.headers, body)
        logger.info(
            "WAKE received: event=%s task_id=%s issuer=%s data_keys=%s",
            event_type,
            audit_context.get("task_id"),
            audit_context.get("issuer"),
            list(body.get("data", {}).keys()),
        )

        if self._cognitive_loop is not None:
            record_wake_event = getattr(self._cognitive_loop, "record_wake_event", None)
            if callable(record_wake_event):
                record_wake_event(event_record)
            self._cognitive_loop.wake(reason=event_type)
        else:
            logger.warning("Wake received but no cognitive loop bound")

        return web.json_response({"accepted": True, "event": event_type})

    # ======================================================================
    # AUTONOMOUS MODE — fine-grained tool access
    # ======================================================================

    async def _handle_list_tools(self, request: Any) -> Any:
        """GET /v1/tools — catalog with schemas, categories, and gas costs."""
        from aiohttp import web

        category = request.query.get("category")
        tools_list = self._tools.list_tools()
        if category:
            tools_list = [t for t in tools_list if t.category == category]

        # Group by category for overview
        categories: dict[str, int] = {}
        items = []
        for t in tools_list:
            categories[t.category] = categories.get(t.category, 0) + 1
            items.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "category": t.category,
                "gas_cost": t.estimated_cost,
                "requires_conscience": t.requires_conscience,
            })

        return web.json_response({
            "tools": items,
            "count": len(items),
            "categories": categories,
        })

    async def _handle_tool_call(self, request: Any) -> Any:
        """POST /v1/tools/{name} — Autonomous: execute one tool, debit gas.

        Headers: X-Civitas-DID
        Body: {"params": {...}}
        """
        from aiohttp import web

        did = self._authenticate(request)
        name = request.match_info["name"]

        try:
            body = await request.json()
        except Exception:
            body = {}
        params = body.get("params", {})

        # Find tool
        tool = self._tools.get(name)
        if not tool:
            return web.json_response(
                {"error": f"Tool '{name}' not found", "hint": "GET /v1/tools to list"},
                status=404,
            )

        # Estimate cost
        gas = self._energy.estimate_cost(name)

        # Check caller balance
        balance = self._get_caller_balance(did)
        if balance < gas:
            return web.json_response({
                "error": "insufficient_balance",
                "balance": balance,
                "required": gas,
                "tool": name,
            }, status=402)

        # Conscience check
        decision = Decision(action=name, params=params, reasoning="autonomous_call")
        if tool.requires_conscience:
            verdict = self._conscience.check(
                decision,
                self._energy.state,
                context={"agent_id": did, "active_task_count": 0},
            )
            if not verdict.allowed:
                return web.json_response({
                    "error": "conscience_denied",
                    "reason": verdict.reason,
                    "suggestion": verdict.suggestion,
                }, status=403)

        # Execute
        t0 = time.monotonic()
        try:
            result = await self._tools.aexecute(name, params)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            actual_cost = self._energy.debit(name)

            # Record in ledger
            self._ledger.record(CallRecord(
                caller_did=did,
                mode="autonomous",
                action=name,
                gas_cost=actual_cost,
                timestamp=time.time(),
                success=True,
            ))

            return web.json_response({
                "success": True,
                "result": _safe_json(result),
                "gas_used": actual_cost,
                "duration_ms": elapsed_ms,
            })

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._ledger.record(CallRecord(
                caller_did=did,
                mode="autonomous",
                action=name,
                gas_cost=0,
                timestamp=time.time(),
                success=False,
            ))
            return web.json_response({
                "success": False,
                "error": str(exc),
                "duration_ms": elapsed_ms,
            }, status=500)

    # ======================================================================
    # DELEGATE MODE — intent-level expert proxy
    # ======================================================================

    async def _handle_delegate(self, request: Any) -> Any:
        """POST /v1/delegate — submit intent, runtime executes as expert.

        Headers: X-Civitas-DID
        Body: {"intent": "...", "budget": 50.0, "steps": 1}
        """
        from aiohttp import web

        did = self._authenticate(request)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        intent = body.get("intent", "")
        budget = float(body.get("budget", 50.0))
        max_steps = min(int(body.get("steps", 1)), self._config.max_delegate_steps)

        if not intent:
            return web.json_response({"error": "missing 'intent' field"}, status=400)

        if not self._llm:
            return web.json_response(
                {"error": "delegate mode requires LLM — configure llm in gateway"},
                status=503,
            )

        # Check caller balance
        balance = self._get_caller_balance(did)
        if balance < budget:
            return web.json_response({
                "error": "insufficient_balance",
                "balance": balance,
                "requested_budget": budget,
            }, status=402)

        # Run delegate tick(s)
        results = []
        total_gas = 0.0

        for step in range(max_steps):
            ctx = await self._delegate_tick(intent, did, budget - total_gas)

            gas_for_step = (ctx.evaluation.cost if ctx.evaluation else 0) + self._config.delegate_surcharge
            total_gas += gas_for_step

            # Record in ledger
            self._ledger.record(CallRecord(
                caller_did=did,
                mode="delegate",
                action=ctx.decision.action if ctx.decision else "none",
                gas_cost=gas_for_step,
                timestamp=time.time(),
                success=ctx.evaluation.success if ctx.evaluation else False,
                tick_id=ctx.tick_id,
            ))

            results.append({
                "step": step + 1,
                "tick_id": ctx.tick_id,
                "action": ctx.decision.action if ctx.decision else None,
                "params": ctx.decision.params if ctx.decision else None,
                "reasoning": ctx.decision.reasoning if ctx.decision else None,
                "success": ctx.evaluation.success if ctx.evaluation else False,
                "result": _safe_json(ctx.evaluation.outcome) if ctx.evaluation else None,
                "error": ctx.evaluation.error if ctx.evaluation else None,
                "reflection": ctx.reflection,
                "gas": gas_for_step,
            })

            # Stop conditions
            if ctx.decision and ctx.decision.action == "wait":
                break
            if total_gas >= budget:
                break

        return web.json_response({
            "success": all(r["success"] for r in results if r["action"] != "wait"),
            "steps": results,
            "total_gas": total_gas,
            "budget_remaining": budget - total_gas,
        })

    async def _delegate_tick(
        self,
        intent: str,
        caller_did: str,
        remaining_budget: float,
    ) -> TickContext:
        """Execute one cognitive tick driven by external intent."""
        assert self._llm is not None

        ctx = TickContext()

        # 1. Perceive — still get CivitasOS state for context
        try:
            ctx.briefing = self._agent.briefing()
        except Exception:
            ctx.briefing = {}
        self._energy.refresh(ctx.briefing.get("economics", {}))

        # 2. LLM decides based on intent + CivitasOS state
        energy = self._energy.state
        system = _DELEGATE_TEMPLATE.format(
            name=self._name,
            reputation=energy.reputation,
            balance=energy.balance,
            tool_count=len(self._tools.list_tools()),
            budget=remaining_budget,
        )

        user_msg = (
            f"委托意图:\n{intent}\n\n"
            f"当前CivitasOS状态:\n"
            f"- 活跃任务: {len(ctx.briefing.get('active_tasks', []))}\n"
            f"- 可用机会: {len(ctx.briefing.get('opportunities', []))}\n"
            f"- 余额: {energy.balance:.0f} CIV\n\n"
            f"请选择最合适的工具执行，或回复 wait 说明原因。"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = await self._llm.chat(
                messages,
                tools=self._tools.to_openai_tools() or None,
            )
        except Exception:
            logger.exception("Delegate LLM call failed")
            ctx.decision = Decision(action="wait", reasoning="LLM call failed", source=DecisionSource.LLM)
            return ctx

        # Parse LLM decision
        if response.tool_calls:
            tc = response.tool_calls[0]
            ctx.decision = Decision(
                action=tc.name,
                params=tc.arguments,
                reasoning=response.content or f"Expert chose {tc.name}",
                confidence=0.8,
                source=DecisionSource.LLM,
            )
        elif response.content and "wait" in response.content.lower():
            ctx.decision = Decision(action="wait", reasoning=response.content, source=DecisionSource.LLM)
            return ctx
        else:
            ctx.decision = Decision(
                action="wait",
                reasoning=response.content or "No tool selected",
                source=DecisionSource.LLM,
            )
            return ctx

        # 3. Conscience check
        tool_def = self._tools.get(ctx.decision.action)
        if tool_def and tool_def.requires_conscience:
            verdict = self._conscience.check(
                ctx.decision,
                self._energy.state,
                context={"agent_id": caller_did, "active_task_count": 0},
            )
            ctx.conscience_verdict = verdict
            if not verdict.allowed:
                ctx.evaluation = Evaluation(success=False, error=f"Conscience: {verdict.reason}")
                ctx.reflection = f"BLOCKED: {verdict.reason}. Suggestion: {verdict.suggestion}"
                return ctx

        # 4. Execute
        t0 = time.monotonic()
        try:
            ctx.action_result = await self._tools.aexecute(
                ctx.decision.action, ctx.decision.params,
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            cost = self._energy.debit(ctx.decision.action)
            ctx.evaluation = Evaluation(
                success=True,
                outcome=ctx.action_result,
                cost=cost,
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            ctx.evaluation = Evaluation(success=False, error=str(exc), duration_ms=elapsed)

        # 5. Reflect
        if ctx.evaluation.success:
            ctx.reflection = (
                f"✓ Executed {ctx.decision.action} for delegate. "
                f"Cost {ctx.evaluation.cost:.2f} CIV in {ctx.evaluation.duration_ms}ms."
            )
        else:
            ctx.reflection = (
                f"✗ {ctx.decision.action} failed: {ctx.evaluation.error}. "
                f"Will suggest alternative approach."
            )

        return ctx

    # ======================================================================
    # STATUS & LEDGER
    # ======================================================================

    async def _handle_status(self, request: Any) -> Any:
        """GET /v1/status — runtime status and caller debt (if authed)."""
        from aiohttp import web

        did = request.headers.get("X-Civitas-DID", "")
        energy = self._energy.state

        resp: dict[str, Any] = {
            "runtime": self._name,
            "mode": "autonomous + delegate",
            "tools": len(self._tools.list_tools()),
            "reputation": energy.reputation,
            "balance": energy.balance,
        }

        if did:
            resp["caller_debt"] = self._ledger.debt(did)
            try:
                caller_balance = self._get_caller_balance(did)
                resp["caller_balance"] = caller_balance
            except Exception:
                pass

        return web.json_response(resp)

    async def _handle_ledger(self, request: Any) -> Any:
        """GET /v1/ledger — economic settlement ledger summary."""
        from aiohttp import web
        return web.json_response(self._ledger.summary())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if obj is None:
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
