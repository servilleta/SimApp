#!/usr/bin/env python
"""Load-testing harness for Power Engine heterogeneous scaling.

Run this script locally to benchmark formula throughput on:
    • CUDA GPUs       – when CuPy is available.
    • oneAPI/SYCL GPUs – when dpctl detects devices.
    • Remote GPUs     – when NIM_REMOTE_GPUS/NIM_SERVER_ADDR are set.

Results are printed to console and saved as CSV under benchmarks/results_<timestamp>.csv.

Usage::
    python benchmarks/load_test_scaling.py --formulas 100000 --range 10

The synthetic workload consists of `=SUM(Ax:Ay)` formulas that stress the
SUM-range GPU path without incurring Excel-file I/O.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import asyncio
import logging

# Ensure project root is on PYTHONPATH when executed as standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports from project modules – avoid heavy deps unless needed
# ---------------------------------------------------------------------------
from backend.simulation.power_engine import PowerMonteCarloEngine  # noqa: E402

logger = logging.getLogger("load_test_scaling")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Workload generator
# ---------------------------------------------------------------------------

def generate_workload(num_formulas: int, range_size: int = 10) -> List[tuple]:
    """Generate a list of synthetic SUM-range formulas.

    Each formula is attached to a unique target cell (Z-column) so that
    results do not collide in the engine's key space.
    """
    sheet = "Sheet1"
    formula_list: list[tuple] = []
    for idx in range(1, num_formulas + 1):
        # Cycle rows to keep indices small – avoids huge cell references
        start_row = ((idx - 1) % 1000) + 1  # 1-based
        end_row = start_row + range_size - 1
        target_cell = f"Z{idx}"
        formula = f"=SUM(A{start_row}:A{end_row})"
        formula_list.append((sheet, target_cell, formula))
    return formula_list


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def reset_rr_iterator(pool):
    """Reset round-robin iterator after manual device filtering."""
    if pool._devices:  # noqa: SLF001 – internal field access accepted in bench code
        pool._rr_iter = itertools.cycle(pool._devices)
    else:
        pool._rr_iter = None


def run_benchmark(device_type: str | None, formulas: int, range_size: int) -> Dict[str, Any] | None:
    """Run a single benchmark pass on the requested *device_type*.

    Args:
        device_type: "cuda", "oneapi", "remote", or None for mixed/auto.
        formulas:    Number of formulas in synthetic workload.
        range_size:  Number of rows in each SUM range.
    Returns:
        Dictionary with metrics, or None if the device type is unavailable.
    """
    # Respect user environment – do not create remote stubs automatically
    if device_type == "remote" and not os.getenv("NIM_REMOTE_GPUS"):
        logger.info("[SKIP] remote benchmark requested but NIM_REMOTE_GPUS not configured.")
        return None

    engine = PowerMonteCarloEngine(config={"chunk_size": 2048})

    # Filter the device pool when a specific type is requested
    if device_type:
        engine.device_pool._devices = [d for d in engine.device_pool._devices if d.type == device_type]  # noqa: SLF001
        reset_rr_iterator(engine.device_pool)
        if not engine.device_pool._devices:
            logger.info("[SKIP] %s device not available", device_type)
            return None

    # Pre-warm GPU compilation to exclude one-time overhead
    _ = engine._compile_gpu_kernel("SUM_RANGE")

    # Generate workload
    formula_list = generate_workload(formulas, range_size)
    iteration_values: Dict[tuple, float] = {}

    # Run batch (async)
    async def _run() -> Dict[tuple, float]:
        return await engine._execute_gpu_batch(formula_list, iteration_values, {})

    t0 = time.perf_counter()
    results_dict = asyncio.run(_run())
    elapsed = time.perf_counter() - t0

    throughput = len(results_dict) / elapsed if elapsed else 0.0

    return {
        "device_type": device_type or "mixed",
        "formulas": len(results_dict),
        "elapsed_s": round(elapsed, 4),
        "throughput_fps": round(throughput, 2),
        "gpu_kernels": engine.metrics.get("gpu_kernels_launched", 0),
        "gpu_available": engine.gpu_available,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Power Engine scaling benchmark")
    parser.add_argument("--formulas", type=int, default=50000, help="Number of formulas in workload")
    parser.add_argument("--range", dest="range_size", type=int, default=10, help="Rows per SUM range")
    parser.add_argument("--csv", action="store_true", help="Write results_<timestamp>.csv under benchmarks/")

    args = parser.parse_args(argv)
    scenarios = [None, "cuda", "oneapi", "remote"]  # None = mixed / auto

    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        bench = run_benchmark(sc, args.formulas, args.range_size)
        if bench:
            rows.append(bench)

    # Console output
    if not rows:
        print("No accelerators detected – all benchmarks skipped.")
        return 0

    header = ["device_type", "formulas", "elapsed_s", "throughput_fps", "gpu_kernels", "gpu_available"]
    col_widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in header}
    print("\nPower Engine Scaling Benchmarks\n" + "-" * 40)
    print(" ".join(h.ljust(col_widths[h]) for h in header))
    for r in rows:
        print(" ".join(str(r[h]).ljust(col_widths[h]) for h in header))

    # Optional CSV output
    if args.csv:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "benchmarks" / f"results_{ts}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli()) 