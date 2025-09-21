from __future__ import annotations

"""Triton kernel generation utilities for Power Engine (Phase-1).

The kernels are compiled lazily so importing this module does **not** require
Triton to be installed in all environments.  All public helpers return `None`
when Triton is unavailable – the caller should then fall back to the legacy
CuPy/PTX path.
"""

from typing import Optional, Any
import os
import logging

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception as e:  # pragma: no cover – runtime optional
    logger.debug("[TRITON] Unavailable: %s", e)
    _TRITON_OK = False


def build_sum_kernel(use_amp: bool = False) -> Optional[Any]:
    """Compile a Triton kernel that reduces a 1-D array to a scalar sum."""
    if not _TRITON_OK:
        return None

    BLOCK = 1024

    @triton.jit
    def sum_kernel(x_ptr, n, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc = tl.sum(x, axis=0)
        # First thread of each block writes partial sum
        if tl.static_all(tl.arange(0, BLOCK_SIZE) == 0):
            tl.atomic_add(out_ptr, 0, acc)

    def launcher(x, out, stream):
        import torch  # Triton requires torch tensors for launch config
        n = x.numel()
        grid = ( (n + BLOCK - 1) // BLOCK, )
        sum_kernel[grid](x, n, out, BLOCK_SIZE=BLOCK, stream=stream)

    return launcher


def build_arithmetic_kernel(op: str, use_amp: bool = False) -> Optional[Any]:
    """Compile a simple element-wise arithmetic kernel.

    `op` must be one of '+', '-', '*', '/'.
    """
    if not _TRITON_OK:
        return None

    if op not in {'+', '-', '*', '/'}:
        raise ValueError("Unsupported op")

    @triton.jit
    def arith_kernel(a_ptr, b_ptr, n, out_ptr):
        idx = tl.program_id(axis=0) * 1024 + tl.arange(0, 1024)
        mask = idx < n
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        if op == '+':
            res = a + b
        elif op == '-':
            res = a - b
        elif op == '*':
            res = a * b
        else:
            res = tl.where(b != 0, a / b, 0.0)
        tl.store(out_ptr + idx, res, mask=mask)

    return arith_kernel


def build_all_arith_kernels(use_amp: bool = False) -> dict[str, Any]:
    """Return a dictionary with launchers for '+', '-', '*', '/'.

    The function silently returns an empty dict if Triton is not available.
    """
    if not _TRITON_OK:
        return {}

    kernels: dict[str, Any] = {}
    for _op in ['+', '-', '*', '/']:
        k = build_arithmetic_kernel(_op, use_amp)
        if k:
            kernels[_op] = k
    return kernels 