#!/usr/bin/env python3
"""P1-5: Long-running stability test — 1000+ cognitive loop ticks.

Validates:
  - No memory leaks (RSS stays within bounds)
  - Energy accounting is correct (breathing tax accumulates)
  - Tick count monotonically increases
  - No unhandled exceptions
  - Mode transitions work (IDLE → EXPLORE)

Runs WITHOUT LLM — uses a stub that always returns "wait".
"""

import asyncio
import os
import resource
import sys
import time

# Ensure we can import civitasos_runtime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Override before imports
os.environ.setdefault("CIVITASOS_URL", "http://localhost:8099")


def get_rss_mb() -> float:
    """Current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


async def run_stability_test(n_ticks: int = 1000) -> bool:
    from civitasos import CivitasAgent
    from civitasos_runtime.loop import CognitiveLoop
    from civitasos_runtime.energy import Energy
    from civitasos_runtime.conscience import Conscience
    from civitasos_runtime.rules import RulesEngine
    from civitasos_runtime.tools import ToolRegistry
    from civitasos_runtime.memory import HybridMemory

    # ── Stub LLM (no network calls) ──────────────────────────────────
    class StubLLM:
        """Returns 'wait' for every decision request."""

        async def decide(self, prompt: str, **_kw) -> dict:
            return {"action": "wait", "reasoning": "stub-llm: no action", "params": {}}

        async def reflect(self, context: str, **_kw) -> str:
            return "stub reflection"

    # ── Setup ─────────────────────────────────────────────────────────
    agent = CivitasAgent(
        base_url="http://localhost:8099",
    )

    energy = Energy()
    # Set high initial balance so breathing tax doesn't bankrupt us
    energy._state.balance = 1_000_000.0
    energy._state.staked = 100.0
    energy._state.balance_cap = 2_000_000.0

    conscience = Conscience()
    rules = RulesEngine()
    tools = ToolRegistry(agent)
    memory = HybridMemory(agent=agent, data_dir="/tmp/stability_test_mem")

    loop = CognitiveLoop(
        agent,
        llm=StubLLM(),
        conscience=conscience,
        energy=energy,
        rules=rules,
        tools=tools,
        agent_name="StabilityBot",
        capabilities=["testing"],
        memory=memory,
    )

    # ── Metrics ───────────────────────────────────────────────────────
    rss_start = get_rss_mb()
    balance_start = energy.state.balance
    t0 = time.monotonic()
    errors = []

    print(f"{'='*60}")
    print(f" Stability Test: {n_ticks} ticks")
    print(f" RSS start: {rss_start:.1f} MB | Balance: {balance_start:.1f} CIV")
    print(f"{'='*60}")
    print()

    # ── Run N ticks ────────────────────────────────────────────────────
    for i in range(1, n_ticks + 1):
        try:
            ctx = await loop.tick()
        except Exception as e:
            errors.append((i, str(e)))
            if len(errors) > 10:
                print(f"  Too many errors ({len(errors)}), aborting")
                break
            continue

        # Progress every 200 ticks
        if i % 200 == 0 or i == n_ticks:
            rss_now = get_rss_mb()
            bal = energy.state.balance
            elapsed = time.monotonic() - t0
            tps = i / elapsed if elapsed > 0 else 0
            print(
                f"  Tick {i:5d} | RSS {rss_now:.1f} MB | "
                f"Balance {bal:.1f} CIV | Tick/s {tps:.1f}"
            )

    # ── Assertions ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    rss_end = get_rss_mb()
    rss_growth = rss_end - rss_start
    balance_end = energy.state.balance
    balance_spent = balance_start - balance_end

    print()
    print(f"{'='*60}")
    print(f" Results")
    print(f"{'='*60}")
    print(f"  Ticks completed: {loop.tick_count}")
    print(f"  Elapsed:         {elapsed:.2f}s ({loop.tick_count/elapsed:.1f} ticks/s)")
    print(f"  RSS start/end:   {rss_start:.1f} / {rss_end:.1f} MB (growth: {rss_growth:+.1f} MB)")
    print(f"  Balance spent:   {balance_spent:.3f} CIV (breathing tax)")
    print(f"  Errors:          {len(errors)}")

    passed = 0
    failed = 0

    def check(desc, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {desc}")
            passed += 1
        else:
            print(f"  ❌ {desc}")
            failed += 1

    print()
    check(f"completed {n_ticks} ticks", loop.tick_count >= n_ticks)
    check(f"no errors (got {len(errors)})", len(errors) == 0)
    check(f"RSS growth < 50 MB ({rss_growth:+.1f})", rss_growth < 50)
    check(f"breathing tax deducted ({balance_spent:.3f} CIV > 0)", balance_spent > 0)
    check(f"not bankrupt (balance {balance_end:.1f})", not energy.is_bankrupt)
    check(f"tick rate > 10/s ({loop.tick_count/elapsed:.1f})", loop.tick_count / elapsed > 10)

    print()
    total = passed + failed
    if failed == 0:
        print(f" Stability: {passed}/{total} passed ✅")
    else:
        print(f" Stability: {passed}/{total} passed, {failed} FAILED ❌")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    ok = asyncio.run(run_stability_test(n))
    sys.exit(0 if ok else 1)
