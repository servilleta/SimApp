"""Optional Pixie eBPF profiler bootstrap (Phase-4).

If Pixie Python client (`px`) is available inside the container the
profiler attaches automatically and streams flamegraphs to the Pixie
backend defined by `$PIXIE_CLUSTER_ID`. Otherwise a warning is logged
and execution continues – ensuring graceful degradation in containers
where eBPF is not permitted.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

try:
    import px  # type: ignore
except ImportError:
    px = None  # type: ignore


_INITIALISED = False


def ensure_pixie_profiler():
    global _INITIALISED
    if _INITIALISED or px is None:
        if px is None:
            logger.warning("[OBS] Pixie client not installed – eBPF profiling disabled")
        return

    cluster_id = os.getenv("PIXIE_CLUSTER_ID")
    if not cluster_id:
        logger.warning("[OBS] PIXIE_CLUSTER_ID not set – skipping Pixie init")
        return

    try:
        px.configure_pixie(cluster_id=cluster_id)
        px.start_flame_graph()
        _INITIALISED = True
        logger.info("[OBS] Pixie eBPF profiler started – cluster=%s", cluster_id)
    except Exception as e:  # pragma: no cover
        logger.warning("[OBS] Failed to start Pixie profiler: %s", e) 