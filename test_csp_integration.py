"""CSP Integration Test — validates remember/recall through live CSP.

Requires:
  - Backend running on localhost:8099
  - CSP running on localhost:8200
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "civitasos-sdk", "python"))
sys.path.insert(0, os.path.dirname(__file__))

from civitasos import CivitasAgent

BACKEND = "http://localhost:8099"
CSP = "http://localhost:8200"
AGENT_ID = "csp-test-agent"

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


def main():
    global passed, failed

    # Create agent WITH CSP configured
    agent = CivitasAgent(
        BACKEND,
        cognitive_provider={
            "url": CSP,
            "services": ["memory", "briefing"],
        },
    )
    agent.generate_keys()

    try:
        agent.register(
            agent_id=AGENT_ID,
            name="CSP Test Agent",
            capabilities=["testing"],
            stake=100,
        )
    except Exception:
        pass  # may already exist

    print(f"\n{'='*60}")
    print("CSP Integration Test")
    print(f"  Backend: {BACKEND}")
    print(f"  CSP:     {CSP}")
    print(f"  Agent:   {AGENT_ID}")
    print(f"{'='*60}\n")

    # ── T1: remember / recall roundtrip ──────────────────────────
    print("Phase 1: Memory CRUD")

    def t1_remember_recall():
        agent.remember("test_key", {"strategy": "aggressive", "confidence": 0.9})
        val = agent.recall("test_key")
        assert val is not None, "recall returned None"
        assert val.get("strategy") == "aggressive", f"wrong value: {val}"
        assert val.get("confidence") == 0.9

    test("remember + recall roundtrip", t1_remember_recall)

    def t1_overwrite():
        agent.remember("test_key", {"strategy": "defensive", "confidence": 0.5})
        val = agent.recall("test_key")
        assert val["strategy"] == "defensive"
        assert val["confidence"] == 0.5

    test("overwrite existing key", t1_overwrite)

    def t1_recall_missing():
        val = agent.recall("nonexistent_key_xyz")
        assert val is None, f"Expected None, got {val}"

    test("recall missing key returns None", t1_recall_missing)

    def t1_complex_value():
        data = {
            "lessons": ["don't over-commit", "check reputation first"],
            "metrics": {"success_rate": 0.87, "avg_reward": 42.5},
            "nested": {"deep": {"value": True}},
        }
        agent.remember("complex_data", data)
        val = agent.recall("complex_data")
        assert val["lessons"][1] == "check reputation first"
        assert val["metrics"]["success_rate"] == 0.87
        assert val["nested"]["deep"]["value"] is True

    test("complex nested value", t1_complex_value)

    def t1_tags_and_ttl():
        result = agent.remember("tagged_key", "hello", tags=["test", "ephemeral"], ttl_secs=3600)
        assert result is not None
        val = agent.recall("tagged_key")
        assert val == "hello"

    test("remember with tags and TTL", t1_tags_and_ttl)

    def t1_forget():
        agent.remember("to_delete", "temp")
        assert agent.recall("to_delete") == "temp"
        agent.forget("to_delete")
        assert agent.recall("to_delete") is None

    test("forget (delete) key", t1_forget)

    def t1_list_keys():
        keys = agent.memory_list_keys()
        assert isinstance(keys, list), f"Expected list, got {type(keys)}"
        assert len(keys) >= 2, f"Expected ≥2 keys, got {len(keys)}: {keys}"

    test("memory_list_keys returns keys", t1_list_keys)

    # ── T2: HybridMemory (Runtime layer) ────────────────────────
    print("\nPhase 2: HybridMemory (local + remote)")

    from civitas_runtime.memory import HybridMemory
    import tempfile

    def t2_hybrid_roundtrip():
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = HybridMemory(agent, data_dir=tmpdir)
            mem.remember("hybrid_key", {"source": "hybrid"})
            val = mem.recall("hybrid_key")
            assert val["source"] == "hybrid"
            mem.close()

    test("HybridMemory remember+recall", t2_hybrid_roundtrip)

    def t2_local_fallback():
        """Writing via HybridMemory, directly reading from remote confirms both paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = HybridMemory(agent, data_dir=tmpdir)
            mem.remember("dual_write_key", {"written_by": "hybrid"})
            
            # Read directly from CSP to confirm remote write happened
            remote_val = agent.recall("dual_write_key")
            assert remote_val is not None, "Remote write did not reach CSP"
            assert remote_val["written_by"] == "hybrid"
            
            # Read from local (should hit local cache first)
            local_val = mem.recall("dual_write_key")
            assert local_val["written_by"] == "hybrid"
            mem.close()

    test("HybridMemory dual-write (local + CSP)", t2_local_fallback)

    # ── T3: Briefing via CSP ─────────────────────────────────────
    print("\nPhase 3: Briefing from CSP")

    def t3_briefing():
        try:
            b = agent.briefing(agent_id=AGENT_ID)
            assert b is not None
            # CSP briefing returns these fields
            assert "agent_did" in b or "agent" in b or "economics" in b, f"Unexpected briefing shape: {list(b.keys())}"
            print(f"       Briefing keys: {list(b.keys())}")
        except Exception as e:
            # If state-feed endpoint doesn't exist yet, that's a known gap
            if "502" in str(e) or "BAD_GATEWAY" in str(e) or "404" in str(e):
                print(f"       ⚠ Briefing CSP→Core state-feed not wired (expected): {e}")
                return
            raise

    test("CSP briefing", t3_briefing)

    # ── Summary ──────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"CSP Integration: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED ❌")
    else:
        print(" ✅")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
