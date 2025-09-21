import asyncio
import pytest

from simulation.power_engine import PowerMonteCarloEngine


@pytest.mark.asyncio
async def test_gpu_failure_demotes_to_cpu(monkeypatch):
    engine = PowerMonteCarloEngine()

    # Force GPU available
    engine.gpu_available = True

    async def _fail_gpu(*args, **kwargs):
        raise RuntimeError("Simulated GPU kernel crash")

    monkeypatch.setattr(engine, "_execute_gpu_batch", _fail_gpu)

    # Minimal simulation inputs
    formulas = [("Sheet1", "A1", "=1+1")]
    variables = []

    # Monkeypatch helper functions to avoid external Excel calls
    async def _mock_analyze_file(*a, **k):
        return {
            "formulas": {"Sheet1": {"A1": "=1+1"}},
            "sheet_data": {"Sheet1": {}},
            "complexity": {"total_cells": 1, "formula_cells": 1, "sparsity": 0.0},
        }

    monkeypatch.setattr(engine, "_analyze_file", _mock_analyze_file)
    monkeypatch.setattr(engine, "_optimize_sparse_ranges", lambda f, s: formulas)
    monkeypatch.setattr(engine, "_create_execution_plan", lambda f, c: {
        "mode": "streaming",
        "chunk_size": 1,
        "use_gpu": True,
        "formulas": formulas,
    })

    # Bypass constants loading
    from excel_parser import service as excel_service  # type: ignore
    monkeypatch.setattr(excel_service, "get_constants_for_file", lambda *a, **k: {})

    # Run
    result, errs, sens = await engine.run_simulation(
        file_path="dummy.xlsx",
        file_id="file123",
        target_cell="A1",
        variables=variables,
        iterations=1,
    )

    assert engine.gpu_available is False, "GPU should have been disabled after crash"
    assert result[0] == 2, "Formula should evaluate correctly on CPU even after GPU crash" 