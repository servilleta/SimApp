"""TVM Unity code-generation utilities (Phase-1).

`compile_kernel(kind)` lowers simple numpy-style ops to CUDA or SYCL using
TVM.  Currently supports:
    – sum_range (reduce)
    – arithmetic (add/sub/mul/div)
Falls back to None if TVM isn't available or compilation fails.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tvm
    from tvm import te, auto_scheduler
except ImportError:  # pragma: no cover
    tvm = None  # type: ignore

__all__ = ["compile_kernel"]


def compile_kernel(kind: str):
    if tvm is None:
        logger.warning("[TVM] TVM not installed – skipping code-gen")
        return None

    if kind == "SUM_RANGE":
        n = te.var("n")
        A = te.placeholder((n,), name="A")
        k = te.reduce_axis((0, n), name="k")
        Out = te.compute((), lambda: te.sum(A[k], axis=k), name="Out")
        s = te.create_schedule(Out.op)
    elif kind == "ARITHMETIC":
        n = te.var("n")
        A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B")
        op = te.var("op")  # 0 add,1 sub,2 mul,3 div
        Out = te.compute((n,), lambda i: te.select(op == 0, A[i] + B[i],
                                                  te.select(op == 1, A[i] - B[i],
                                                            te.select(op == 2, A[i] * B[i],
                                                                      te.if_then_else(B[i] != 0, A[i] / B[i], 0)))),
                          name="Out")
        s = te.create_schedule(Out.op)
    else:
        logger.warning("[TVM] Unknown kernel kind %s", kind)
        return None

    try:
        func = tvm.build(s, [A, B] if kind == "ARITHMETIC" else [A], "cuda")  # assuming CUDA target
        logger.info("[TVM] Compiled %s kernel via TVM", kind)
        return func
    except Exception as e:
        logger.warning("[TVM] Kernel compile failed: %s", e)
        return None 