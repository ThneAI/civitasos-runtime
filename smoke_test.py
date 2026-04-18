#!/usr/bin/env python3
"""Smoke test: run a single tick of the distilled Agent against live CivitasOS.

Usage:
    python smoke_test.py

Requires:
    - CivitasOS backend running on localhost:8099
    - CivitasOS CSP running on localhost:8200
    - openai pip package installed
    - QWEN_API_KEY env var or hardcoded below
"""

import asyncio
import json
import logging
import os
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

# ── Config ──────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("CIVITASOS_URL", "http://localhost:8099")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"

# Local Ollama fallback
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")


async def test_phase_1_sdk():
    """Phase 1: Verify SDK can connect, register, and get briefing."""
    logger.info("=" * 60)
    logger.info("Phase 1: SDK Connectivity")
    logger.info("=" * 60)

    from civitasos import CivitasAgent

    agent = CivitasAgent(base_url=BACKEND_URL)
    agent.generate_keys()
    logger.info("Keys generated, agent_id (pre-register): %s", agent.agent_id)

    # Register
    result = agent.a2a_quickstart(
        name="DistilledAgent-Smoke",
        endpoint="",
        description="Smoke test of civitasos-runtime distilled Agent",
    )
    did = result["agent"]["did"]
    logger.info("Registered: DID=%s, reputation=%.2f, tier=%s",
                did, result["agent"]["reputation"], result["agent"]["trust_tier"])

    # Briefing
    briefing = agent.briefing()
    logger.info("Briefing received: %s", list(briefing.keys()))
    logger.info("  economics: %s", json.dumps(briefing.get("economics", {}), indent=2))
    logger.info("  active_tasks: %d", len(briefing.get("active_tasks", [])))
    logger.info("  opportunities: %d", len(briefing.get("opportunities", [])))
    logger.info("  urgency: %s", briefing.get("urgency"))

    return agent, briefing


async def test_phase_2_tools(agent):
    """Phase 2: Verify ToolRegistry auto-discovery."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Tool Discovery")
    logger.info("=" * 60)

    from civitasos_runtime.tools import ToolRegistry

    registry = ToolRegistry(agent)
    tools = registry.list_tools()
    logger.info("Discovered %d tools", len(tools))

    # Group by category
    categories: dict[str, list[str]] = {}
    for t in tools:
        categories.setdefault(t.category, []).append(t.name)
    for cat, names in sorted(categories.items()):
        logger.info("  [%s] %d tools: %s", cat, len(names), ", ".join(names[:5]))
        if len(names) > 5:
            logger.info("    ... and %d more", len(names) - 5)

    # Check OpenAI tool format
    openai_tools = registry.to_openai_tools()
    logger.info("OpenAI tool schemas: %d definitions", len(openai_tools))
    if openai_tools:
        logger.info("  Example: %s", openai_tools[0]["function"]["name"])

    return registry


async def test_phase_3_conscience():
    """Phase 3: Verify Conscience checks."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: Conscience System")
    logger.info("=" * 60)

    from civitasos_runtime.conscience import Conscience
    from civitasos_runtime.models import Decision, EnergyState

    conscience = Conscience()
    energy = EnergyState(balance=100, balance_cap=500, reputation=0.5)

    # Should allow normal action
    d1 = Decision(action="pool_discover", reasoning="Looking for tasks")
    v1 = conscience.check(d1, energy, context={"agent_id": "test", "active_task_count": 0})
    logger.info("pool_discover: allowed=%s reason=%s", v1.allowed, v1.reason)
    assert v1.allowed, "Should allow pool_discover"

    # Should deny forbidden action
    d2 = Decision(action="delete_audit_log", reasoning="Clean up")
    v2 = conscience.check(d2, energy, context={"agent_id": "test", "active_task_count": 0})
    logger.info("delete_audit_log: allowed=%s reason=%s", v2.allowed, v2.reason)
    assert not v2.allowed, "Should deny delete_audit_log"

    # Should deny self-claim
    d3 = Decision(action="pool_claim", params={"poster_id": "test"}, reasoning="Claim own task")
    v3 = conscience.check(d3, energy, context={"agent_id": "test", "active_task_count": 0})
    logger.info("self-claim: allowed=%s reason=%s", v3.allowed, v3.reason)
    assert not v3.allowed, "Should deny self-claim"

    logger.info("All conscience checks passed ✓")


async def test_phase_4_energy():
    """Phase 4: Verify Energy budget tracking."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 4: Energy Budget")
    logger.info("=" * 60)

    from civitasos_runtime.energy import Energy

    energy = Energy()
    energy.refresh({
        "balance": 150.0,
        "staked": 50.0,
        "balance_cap": 500.0,
        "reputation": 0.6,
        "risk_score": 0.2,
    })

    state = energy.state
    logger.info("Balance: %.1f / %.1f CIV", state.balance, state.balance_cap)
    logger.info("Staked: %.1f CIV", state.staked)
    logger.info("Reputation: %.2f", state.reputation)
    logger.info("Fear level: %s", energy.fear_level)
    logger.info("Greed signal: %s", energy.greed_signal)

    # Test affordability
    can = energy.can_afford("pool_claim")
    logger.info("Can afford pool_claim: %s", can)
    assert can, "Should be able to afford with 150 CIV balance"

    cost = energy.debit("pool_claim")
    logger.info("Debited pool_claim: %.2f CIV, remaining: %.1f", cost, energy.state.balance)
    logger.info("Energy system works ✓")


async def test_phase_5_rules(briefing):
    """Phase 5: Verify Rules engine."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 5: Rules Engine")
    logger.info("=" * 60)

    from civitasos_runtime.rules import RulesEngine

    rules = RulesEngine()
    decision = rules.evaluate(briefing, {})
    logger.info("Rules evaluation against live briefing: %s",
                f"action={decision.action}" if decision else "no rule fired (→ LLM)")

    # Test with urgent briefing
    urgent = {**briefing, "urgency": {"deadline_ms": 1000, "task_id": "urgent-1"}}
    decision2 = rules.evaluate(urgent, {})
    logger.info("Urgent briefing: %s",
                f"action={decision2.action}" if decision2 else "no rule fired")
    logger.info("Rules engine works ✓")


async def test_phase_6_llm():
    """Phase 6: Verify LLM adapter with Qwen or local Ollama."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 6: LLM Adapter")
    logger.info("=" * 60)

    from civitasos_runtime.llm import OpenAIAdapter

    if QWEN_API_KEY:
        logger.info("Using Qwen via DashScope")
        llm = OpenAIAdapter(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        model_name = QWEN_MODEL
    else:
        logger.info("Using local Ollama: %s", OLLAMA_MODEL)
        llm = OpenAIAdapter(
            model=OLLAMA_MODEL,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
        )
        model_name = OLLAMA_MODEL

    # Simple completion test
    messages = [
        {"role": "system", "content": "你是CivitasOS中的一个自主Agent。简短回答。"},
        {"role": "user", "content": "你好,请用一句话介绍你自己。"},
    ]

    t0 = time.time()
    response = await llm.chat(messages)
    elapsed = time.time() - t0

    logger.info("LLM response (%.1fs): %s", elapsed, response.content)
    logger.info("Usage: prompt=%d, completion=%d tokens",
                response.usage.get("prompt_tokens", 0),
                response.usage.get("completion_tokens", 0))
    assert response.content, "LLM should return content"
    logger.info("LLM adapter works ✓")

    return llm


async def test_phase_7_single_tick(agent, llm):
    """Phase 7: Run one full cognitive tick."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 7: FULL COGNITIVE TICK")
    logger.info("=" * 60)

    from civitasos_runtime.conscience import Conscience
    from civitasos_runtime.energy import Energy
    from civitasos_runtime.loop import CognitiveLoop
    from civitasos_runtime.rules import RulesEngine
    from civitasos_runtime.tools import ToolRegistry

    conscience = Conscience()
    energy = Energy()
    rules = RulesEngine()
    tools = ToolRegistry(agent)

    loop = CognitiveLoop(
        agent,
        llm=llm,
        conscience=conscience,
        energy=energy,
        rules=rules,
        tools=tools,
        agent_name="DistilledAgent-Smoke",
        capabilities=["general", "task-execution"],
    )

    # Execute a single tick
    t0 = time.time()
    ctx = await loop.tick()
    elapsed = time.time() - t0

    logger.info("")
    logger.info("─── Tick Result ───")
    logger.info("Tick ID: %s", ctx.tick_id)
    logger.info("Phase reached: %s", ctx.phase.value if ctx.phase else "none")
    logger.info("Decision: action=%s source=%s",
                ctx.decision.action if ctx.decision else "none",
                ctx.decision.source.value if ctx.decision and ctx.decision.source else "none")
    if ctx.decision and ctx.decision.reasoning:
        logger.info("Reasoning: %s", ctx.decision.reasoning[:200])
    if ctx.conscience_verdict:
        logger.info("Conscience: allowed=%s reason=%s",
                    ctx.conscience_verdict.allowed, ctx.conscience_verdict.reason)
    if ctx.evaluation:
        logger.info("Evaluation: success=%s cost=%.2f duration=%dms",
                    ctx.evaluation.success,
                    ctx.evaluation.cost,
                    ctx.evaluation.duration_ms)
        if ctx.evaluation.outcome:
            outcome_str = json.dumps(ctx.evaluation.outcome, ensure_ascii=False, default=str)
            logger.info("Outcome: %s", outcome_str[:300])
    if ctx.reflection:
        logger.info("Reflection: %s", ctx.reflection[:200])
    logger.info("Total tick time: %.2fs", elapsed)
    logger.info("Loop mode after tick: %s", loop.mode.value)

    return ctx


async def main():
    logger.info("🚀 CivitasOS Distilled Agent — Smoke Test")
    logger.info("Backend: %s", BACKEND_URL)
    logger.info("LLM: %s (Ollama fallback: %s)", QWEN_MODEL if QWEN_API_KEY else "N/A", OLLAMA_MODEL)
    logger.info("")

    results: dict[str, str] = {}

    try:
        # Phase 1: SDK
        agent, briefing = await test_phase_1_sdk()
        results["SDK"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 1 FAILED: %s", e)
        results["SDK"] = f"FAIL ✗: {e}"
        return results

    try:
        # Phase 2: Tools
        registry = await test_phase_2_tools(agent)
        results["Tools"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 2 FAILED: %s", e)
        results["Tools"] = f"FAIL ✗: {e}"

    try:
        # Phase 3: Conscience
        await test_phase_3_conscience()
        results["Conscience"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 3 FAILED: %s", e)
        results["Conscience"] = f"FAIL ✗: {e}"

    try:
        # Phase 4: Energy
        await test_phase_4_energy()
        results["Energy"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 4 FAILED: %s", e)
        results["Energy"] = f"FAIL ✗: {e}"

    try:
        # Phase 5: Rules
        await test_phase_5_rules(briefing)
        results["Rules"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 5 FAILED: %s", e)
        results["Rules"] = f"FAIL ✗: {e}"

    try:
        # Phase 6: LLM
        llm = await test_phase_6_llm()
        results["LLM"] = "PASS ✓"
    except Exception as e:
        logger.error("Phase 6 FAILED: %s", e)
        results["LLM"] = f"FAIL ✗: {e}"
        llm = None

    if llm:
        try:
            # Phase 7: Full tick
            ctx = await test_phase_7_single_tick(agent, llm)
            action = ctx.decision.action if ctx.decision else "none"
            results["CognitiveTick"] = f"PASS ✓ (action={action})"
        except Exception as e:
            logger.error("Phase 7 FAILED: %s", e)
            results["CognitiveTick"] = f"FAIL ✗: {e}"

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for name, status in results.items():
        icon = "✓" if "PASS" in status else "✗"
        logger.info("  %s  %s: %s", icon, name, status)
        if "FAIL" in status:
            all_pass = False

    logger.info("")
    if all_pass:
        logger.info("ALL PHASES PASSED — Agent is operational! 🎉")
    else:
        logger.info("Some phases failed — check logs above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
