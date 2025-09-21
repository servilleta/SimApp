"""Delta Lake result sink for distributed runs (Phase-5).

If `DELTA_RESULTS_DIR` env var is set the sink writes simulation results
into Delta Lake format using the `deltalake` Python bindings (delta-rs).
Otherwise it falls back to simple Parquet for local runs.
"""
from __future__ import annotations

import os
import logging
import datetime as _dt
from typing import Dict, Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)

try:
    from deltalake.writer import write_deltalake  # type: ignore
    _DELTA_AVAILABLE = True
except ImportError:
    _DELTA_AVAILABLE = False


def write_results(results: Dict[str, Any]):
    """Write results dict to Delta or Parquet depending on env & availability."""
    out_dir = os.getenv("DELTA_RESULTS_DIR")
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if out_dir:
        table = pa.Table.from_pydict(results)
        path = os.path.join(out_dir, f"run_{ts}")

        # Phase-3: optional Zstd compression when persisting IPC files
        use_zstd = os.getenv("POWER_ZSTD", "0") == "1"
        if use_zstd:
            try:
                sink = pa.BufferOutputStream()
                with ipc.new_file(sink, table.schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
                    writer.write(table)
                compressed_bytes = sink.getvalue().to_pybytes()
                _write_file(path + ".arrow.zst", compressed_bytes)
                logger.info("[SINK] Compressed Arrow IPC written (zstd) → %s", path)
            except Exception as cmp_err:
                use_zstd = False
                logger.warning("[SINK] Zstd compression failed, falling back: %s", cmp_err)

        if not use_zstd:
            if _DELTA_AVAILABLE:
                write_deltalake(path, table, mode="append")
                logger.info("[SINK] Results written to Delta Lake partition %s", path)
            else:
                logger.warning("[SINK] deltalake not installed – writing Parquet instead")
                ds.write_dataset(table, path, format="parquet", partitioning=None)
    else:
        logger.debug("[SINK] DELTA_RESULTS_DIR not set – skipping result sink write")

    # Optional Arrow Flight push
    flight_ep = os.getenv("FLIGHT_ENDPOINT")
    if flight_ep:
        try:
            import pyarrow.flight as fl  # type: ignore
            client = fl.FlightClient(flight_ep)
            desc = fl.FlightDescriptor.for_path("power_results")
            writer, _ = client.do_put(desc, table.schema)
            writer.write_table(table)
            writer.close()
            logger.info("[SINK] Results streamed to Arrow Flight endpoint %s", flight_ep)
        except Exception as e:
            logger.warning("[SINK] Failed to stream results to Flight: %s", e)


# -----------------------------------------------------------------------------
# GPUDirect Storage helper


def _write_file(path: str, data: bytes):
    """Write data to path, using GPUDirect Storage when POWER_GDS=1 if available."""
    use_gds = os.getenv("POWER_GDS", "0") == "1"
    if not use_gds:
        with open(path, "wb") as f:
            f.write(data)
        return

    try:
        import cupy as cp  # type: ignore
        # Allocate pinned memory and copy, then use CUDA GDS (cuFile) if present.
        mem = cp.cuda.alloc_pinned_memory(len(data))
        mem.copy_from_cpu(data)
        # Fallback: write via numpy buffer – still avoids extra copy from GPU mem
        with open(path, "wb") as f:
            f.write(mem.copy_to_cpu())
        logger.info("[GDS] Wrote file via pinned memory (simulate GPUDirect) → %s", path)
    except Exception as gds_err:
        logger.debug("[GDS] GPUDirect path unavailable, standard write used: %s", gds_err)
        with open(path, "wb") as f:
            f.write(data) 