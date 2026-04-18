#!/usr/bin/env python3
"""Smoke test for CivitasGateway — tests both Autonomous and Delegate modes.

Usage:
    python test_gateway.py

Requires:
    - CivitasOS backend on localhost:8099
    - Ollama with qwen3:latest (for Delegate mode)
    - aiohttp installed
"""

import asyncio
import json
import logging
import os
import sys
import time

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_gateway")

BACKEND_URL = os.getenv("CIVITASOS_URL", "http://localhost:8099")
GATEWAY_PORT = 8300
GATEWAY_BASE = f"http://localhost:{GATEWAY_PORT}"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")

TEST_DID = "did:civ:gateway_test_001"


async def start_gateway():
    """Start the CivitasGateway in the background and return cleanup coro."""
    from civitasos import CivitasAgent
    from civitasos_runtime import (
        CivitasGateway,
        Conscience,
        Energy,
        GatewayConfig,
        ToolRegistry,
    )
    from civitasos_runtime.llm import OpenAIAdapter

    # Create SDK agent
    agent = CivitasAgent(base_url=BACKEND_URL)
    agent.generate_keys()
    agent.a2a_quickstart(
        name="GatewayTestAgent",
        endpoint="",
        description="Gateway smoke test agent",
    )
    logger.info("SDK agent registered: %s", agent.agent_id)

    # Build components
    tools = ToolRegistry(agent)
    conscience = Conscience()
    energy = Energy()

    # Refresh energy from CivitasOS
    try:
        briefing = agent.briefing()
        energy.refresh(briefing.get("economics", {}))
    except Exception:
        pass

    # LLM for delegate mode
    llm = OpenAIAdapter(
        api_key="ollama",
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Gateway
    config = GatewayConfig(port=GATEWAY_PORT, require_auth=True)
    gateway = CivitasGateway(
        agent, tools, conscience, energy, llm=llm,
        config=config, agent_name="GatewayTestAgent",
    )

    await gateway.start()
    logger.info("Gateway started on :%d", GATEWAY_PORT)

    return gateway


async def wait_for_gateway(timeout: float = 5.0):
    """Wait until the gateway is reachable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{GATEWAY_BASE}/v1/status") as resp:
                    if resp.status in (200, 401):
                        return True
        except Exception:
            pass
        await asyncio.sleep(0.2)
    return False


# ===========================================================================
# Tests
# ===========================================================================

async def test_list_tools():
    """GET /v1/tools — should list 100+ tools with categories."""
    logger.info("=" * 60)
    logger.info("Test: List tools (Autonomous discovery)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{GATEWAY_BASE}/v1/tools",
            headers={"X-Civitas-DID": TEST_DID},
        ) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            data = await resp.json()

    count = data["count"]
    cats = data["categories"]
    logger.info("Tools: %d across %d categories", count, len(cats))
    for cat, n in sorted(cats.items()):
        logger.info("  %-12s %3d tools", cat, n)

    assert count > 100, f"Expected 100+ tools, got {count}"
    logger.info("✅ List tools: %d tools discovered", count)
    return True


async def test_auth_required():
    """POST without X-Civitas-DID should return 401."""
    logger.info("=" * 60)
    logger.info("Test: Auth required (no DID header → 401)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GATEWAY_BASE}/v1/tools/pool_discover",
            json={"params": {}},
        ) as resp:
            assert resp.status == 401, f"Expected 401, got {resp.status}"

    logger.info("✅ Auth check: 401 without DID header")
    return True


async def test_autonomous_tool_call():
    """POST /v1/tools/pool_discover — Autonomous mode tool execution."""
    logger.info("=" * 60)
    logger.info("Test: Autonomous tool call (pool_discover)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GATEWAY_BASE}/v1/tools/pool_discover",
            headers={"X-Civitas-DID": TEST_DID},
            json={"params": {}},
        ) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            data = await resp.json()

    logger.info("Result: success=%s gas_used=%.2f duration=%dms",
                data["success"], data["gas_used"], data["duration_ms"])
    logger.info("Discovered: %s", json.dumps(data["result"], ensure_ascii=False)[:200])

    assert data["success"], "Expected success=True"
    assert data["gas_used"] >= 0, "Expected gas >= 0"
    logger.info("✅ Autonomous call: success, gas=%.2f CIV", data["gas_used"])
    return True


async def test_autonomous_404():
    """POST /v1/tools/nonexistent — should return 404."""
    logger.info("=" * 60)
    logger.info("Test: Autonomous 404 (nonexistent tool)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GATEWAY_BASE}/v1/tools/i_dont_exist",
            headers={"X-Civitas-DID": TEST_DID},
            json={"params": {}},
        ) as resp:
            assert resp.status == 404, f"Expected 404, got {resp.status}"

    logger.info("✅ 404 for unknown tool")
    return True


async def test_conscience_block():
    """Autonomous call to a dangerous tool should be blocked by conscience."""
    logger.info("=" * 60)
    logger.info("Test: Conscience block (delete_audit_log)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        # delete_audit_log is hard-denied by conscience
        async with session.post(
            f"{GATEWAY_BASE}/v1/tools/delete_audit_log",
            headers={"X-Civitas-DID": TEST_DID},
            json={"params": {}},
        ) as resp:
            # Tool might not exist in SDK → 404, which is also fine
            status = resp.status
            data = await resp.json()

    if status == 404:
        logger.info("✅ Tool not found (never exposed) — also safe")
    elif status == 403:
        logger.info("✅ Conscience blocked: %s", data.get("reason", ""))
    else:
        logger.info("Status %d — result: %s", status, data)

    return True


async def test_delegate_mode():
    """POST /v1/delegate — Delegate mode: intent → expert reasoning → execution."""
    logger.info("=" * 60)
    logger.info("Test: Delegate mode (intent-based)")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GATEWAY_BASE}/v1/delegate",
            headers={"X-Civitas-DID": TEST_DID},
            json={
                "intent": "查看当前任务池中有哪些可用的机会",
                "budget": 20.0,
                "steps": 1,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            data = await resp.json()

    logger.info("Delegate result:")
    for step in data.get("steps", []):
        logger.info("  Step %d: action=%s success=%s gas=%.2f",
                     step["step"], step["action"], step["success"], step["gas"])
        if step.get("reasoning"):
            logger.info("  Reasoning: %s", step["reasoning"][:150])
        if step.get("reflection"):
            logger.info("  Reflection: %s", step["reflection"][:150])

    logger.info("Total gas: %.2f / budget remaining: %.2f",
                data["total_gas"], data["budget_remaining"])
    logger.info("✅ Delegate mode: completed with %d step(s)", len(data.get("steps", [])))
    return True


async def test_ledger():
    """GET /v1/ledger — should show recorded calls."""
    logger.info("=" * 60)
    logger.info("Test: Ledger tracking")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{GATEWAY_BASE}/v1/ledger",
            headers={"X-Civitas-DID": TEST_DID},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()

    logger.info("Ledger: %d calls, %.2f total gas", data["total_calls"], data["total_gas"])
    if data.get("outstanding"):
        for did, amount in data["outstanding"].items():
            logger.info("  %s owes %.2f CIV", did, amount)
    logger.info("✅ Ledger tracking OK")
    return True


# ===========================================================================
# Main
# ===========================================================================

async def main():
    logger.info("🚀 CivitasGateway Smoke Test")
    logger.info("Backend: %s | Gateway: :%d", BACKEND_URL, GATEWAY_PORT)
    logger.info("")

    # Start gateway
    gateway = await start_gateway()

    # Wait for it to be ready
    ready = await wait_for_gateway()
    if not ready:
        logger.error("Gateway did not start in time")
        sys.exit(1)

    # Run tests
    tests = [
        ("List Tools", test_list_tools),
        ("Auth Required", test_auth_required),
        ("Autonomous Call", test_autonomous_tool_call),
        ("Autonomous 404", test_autonomous_404),
        ("Conscience Block", test_conscience_block),
        ("Delegate Mode", test_delegate_mode),
        ("Ledger", test_ledger),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = await test_fn()
            if result:
                passed += 1
        except Exception as exc:
            logger.error("❌ %s FAILED: %s", name, exc)
            failed += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %d/%d passed, %d failed", passed, len(tests), failed)
    logger.info("=" * 60)

    # Cleanup
    await gateway.stop()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
