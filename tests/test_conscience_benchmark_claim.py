from __future__ import annotations

from civitasos_runtime.conscience import Conscience
from civitasos_runtime.models import Decision, EnergyState


def _high_gap_ctx() -> dict[str, float]:
    return {"aspect_gap": 0.8}


def test_benchmark_target_claim_allowed_under_high_aspect_gap(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_TASK_ID", "R01_happy_01")
    conscience = Conscience()
    decision = Decision(
        action="pool_claim",
        params={"task_id": "backend-task-1", "_benchmark_target_claim": True},
    )

    verdict = conscience.check(decision, EnergyState(), context=_high_gap_ctx())
    assert verdict.allowed is True


def test_non_benchmark_claim_still_blocked_under_high_aspect_gap(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_TASK_ID", raising=False)
    monkeypatch.delenv("BENCHMARK_BACKEND_TASK_ID", raising=False)
    conscience = Conscience()
    decision = Decision(
        action="pool_claim",
        params={"task_id": "backend-task-1", "_benchmark_target_claim": True},
    )

    verdict = conscience.check(decision, EnergyState(), context=_high_gap_ctx())
    assert verdict.allowed is False
    assert "Aspect gap" in verdict.reason


def test_benchmark_target_execute_allowed_under_high_aspect_gap(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_TASK_ID", "R01_happy_01")
    monkeypatch.setenv("BENCHMARK_BACKEND_TASK_ID", "backend-task-2")
    conscience = Conscience()
    decision = Decision(
        action="task_execute",
        params={"task_id": "backend-task-2", "output": {"ok": True}, "success": True},
    )

    verdict = conscience.check(decision, EnergyState(), context=_high_gap_ctx())
    assert verdict.allowed is True


def test_non_benchmark_execute_still_blocked(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_TASK_ID", raising=False)
    monkeypatch.delenv("BENCHMARK_BACKEND_TASK_ID", raising=False)
    conscience = Conscience()
    decision = Decision(
        action="task_execute",
        params={"task_id": "backend-task-2", "output": {"ok": True}, "success": True},
    )

    verdict = conscience.check(decision, EnergyState(), context=_high_gap_ctx())
    assert verdict.allowed is False
    assert "Aspect gap" in verdict.reason
