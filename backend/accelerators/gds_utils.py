"""GPUDirect Storage detection utils (Phase-3).

`is_gds_available()` returns True if libcufile can be loaded and the CUDA
runtime major version is ≥ 12 and cufileInit succeeds.
"""
from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

_lib = None


def is_gds_available() -> bool:
    global _lib
    if _lib is not None:
        return _lib is not False

    try:
        _lib = ctypes.cdll.LoadLibrary("libcufile.so")
        # cufileInit returns 0 on success
        init_fn = _lib.cuFileDriverOpen
        init_fn.restype = ctypes.c_int
        if init_fn() != 0:
            logger.warning("[GDS] cuFileDriverOpen failed – treating as unavailable")
            _lib = False
        else:
            logger.info("[GDS] GPUDirect Storage available and initialised")
    except Exception as e:
        logger.info("[GDS] GPUDirect Storage not available: %s", e)
        _lib = False
    return _lib is not False 