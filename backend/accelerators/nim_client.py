"""Minimal NVIDIA NIM gRPC client stub (Phase-5).

This is a placeholder that defines the API but performs no remote call
unless `NIM_SERVER_ADDR` is provided.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict

import grpc  # type: ignore

logger = logging.getLogger(__name__)

# Protobuf stubs would be generated; we mock minimal interface here
class RemoteKernelStub:  # pragma: no cover
    def __init__(self, channel):
        self._channel = channel

    def RunKernel(self, request):  # type: ignore
        # Not implemented – placeholder
        raise NotImplementedError


def run_kernel_remote(payload: Dict[str, Any]) -> Any:
    addr = os.getenv("NIM_SERVER_ADDR")
    if not addr:
        # Local fallback – compute simple SUM payload if provided
        if payload.get("op") == "SUM_RANGE":
            values = payload.get("values", [])
            return sum(values)
        logger.warning("[NIM] NIM_SERVER_ADDR not set – remote execution disabled")
        return None

    channel = grpc.insecure_channel(addr)
    stub = RemoteKernelStub(channel)
    # Would create protobuf request here
    logger.info("[NIM] Sending kernel request to %s", addr)
    try:
        response = stub.RunKernel(payload)  # type: ignore
        return response
    except Exception as e:
        logger.warning("[NIM] Remote kernel failed: %s", e)
        return None 