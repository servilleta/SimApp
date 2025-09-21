"""DevicePool – heterogeneous device scheduler (Phase-5).

Abstracts:
    • Local NVIDIA GPUs via CuPy.
    • Local SYCL / oneAPI devices (Intel/AMD) via dpctl.
    • Remote NVIDIA GPUs via NVIDIA NIM gRPC (simple round-robin stub).

`allocate()` returns a context manager that yields a *device handle* with
`.type` in {"cuda", "oneapi", "remote"} and `.index` (int or str).

If no accelerators are found the pool is empty and the caller should
fallback to CPU.
"""
from __future__ import annotations

import os
import itertools
from contextlib import contextmanager
from typing import List, Iterator, Any
import logging

logger = logging.getLogger(__name__)


class _Device:
    def __init__(self, dev_type: str, index: Any):
        self.type = dev_type  # "cuda" | "oneapi" | "remote"
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Device {self.type}:{self.index}>"


class DevicePool:
    def __init__(self):
        self._devices: List[_Device] = []
        self._rr_iter: Iterator[_Device] | None = None
        self._discover_local_cuda()
        self._discover_oneapi()
        self._discover_remote_nim()

        if self._devices:
            self._rr_iter = itertools.cycle(self._devices)
            logger.info("[DEVICE_POOL] Initialised with devices: %s", self._devices)
        else:
            logger.warning("[DEVICE_POOL] No accelerators detected – pool is empty")

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def _discover_local_cuda(self):
        try:
            import cupy as cp  # type: ignore

            for dev_id in range(cp.cuda.runtime.getDeviceCount()):
                self._devices.append(_Device("cuda", dev_id))
        except Exception:
            pass

    def _discover_oneapi(self):
        try:
            import dpctl  # type: ignore

            gpus = dpctl.get_devices(device_type="gpu")
            for i, d in enumerate(gpus):
                self._devices.append(_Device("oneapi", i))
        except Exception:
            pass

    def _discover_remote_nim(self):
        remote_cfg = os.getenv("NIM_REMOTE_GPUS")  # e.g. "gpu0,gpu1"
        if remote_cfg:
            for name in remote_cfg.split(","):
                name = name.strip()
                if name:
                    self._devices.append(_Device("remote", name))

    # ------------------------------------------------------------------
    @contextmanager
    def allocate(self):
        """Round-robin allocate device context.

        For *cuda* devices we push/pop CUDA context via CuPy; for oneAPI we
        enter dpctl device context; remote devices are a no-op (handled in
        gRPC stubs elsewhere).
        """
        if not self._rr_iter:
            # Yield None to indicate CPU fallback
            yield None
            return

        dev = next(self._rr_iter)
        if dev.type == "cuda":
            import cupy as cp  # type: ignore

            with cp.cuda.Device(dev.index):
                yield dev
        elif dev.type == "oneapi":
            import dpctl  # type: ignore

            with dpctl.device_context(dpctl.get_devices("gpu")[dev.index]):
                yield dev
        else:
            # remote – context mgmt later
            yield dev

    # ------------------------------------------------------------------
    def has_accelerators(self) -> bool:
        return bool(self._devices) 