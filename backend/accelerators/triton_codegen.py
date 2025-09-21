"""Triton 3.1 kernel code-gen for Power Engine (Phase-1).

Currently supports two kernel templates:
1. sum_range_kernel – vectorised reduction
2. arithmetic_kernel – element-wise +,-,*,/

`compile_kernel(kind, amp)` returns a callable Triton kernel or None if
Triton isn't available or compilation fails.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore

__all__ = ["compile_kernel"]


def _kernel_sum_range(amp: bool):
    """Triton kernel that reduces `n` float values to a single scalar."""

    @triton.jit
    def _sum_kernel(x_ptr, out_ptr, n):  # type: ignore
        pid = tl.program_id(axis=0)
        BLOCK = 1024
        start = pid * BLOCK
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        acc = tl.reduce(x, axis=0)
        if tl.thread_idx(axis=0) == 0:
            tl.atomic_add(out_ptr, 0, acc)

    return _sum_kernel


def _kernel_arithmetic(amp: bool):
    @triton.jit
    def _arith_kernel(a_ptr, b_ptr, out_ptr, n, op: tl.constexpr):  # type: ignore
        pid = tl.program_id(axis=0)
        BLOCK = 1024
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if op == 0:
            c = a + b
        elif op == 1:
            c = a - b
        elif op == 2:
            c = a * b
        else:
            c = tl.where(b != 0, a / b, 0)
        tl.store(out_ptr + offs, c, mask=mask)

    return _arith_kernel


_KIND_TABLE = {
    "SUM_RANGE": _kernel_sum_range,
    "ARITHMETIC": _kernel_arithmetic,
}


def compile_kernel(kind: str, amp: bool = False):
    """Return compiled Triton kernel or None if unavailable."""
    if triton is None:
        logger.warning("[TRITON] Triton not installed – skipping code-gen")
        return None

    fn_builder = _KIND_TABLE.get(kind)
    if not fn_builder:
        logger.warning("[TRITON] Unknown kernel kind %s", kind)
        return None

    try:
        kernel = fn_builder(amp)
        logger.info("[TRITON] Compiled kernel %s (AMP=%s)", kind, amp)
        return kernel
    except Exception as e:  # pragma: no cover
        logger.warning("[TRITON] Failed to compile %s: %s", kind, e)
        return None 