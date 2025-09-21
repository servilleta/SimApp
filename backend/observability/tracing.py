"""OpenTelemetry tracing bootstrap for Power Engine (Phase-4).

Usage:
    from observability.tracing import get_tracer
    with get_tracer().start_as_current_span("my-span"):
        ...

Configuration is environment-driven (no code changes required in prod):
    OTEL_EXPORTER_OTLP_ENDPOINT  – Tempo/OTLP endpoint (default http://localhost:4317)
    OTEL_SERVICE_NAME            – Service name (default "power_engine")

If OpenTelemetry SDK or exporter is missing the functions become no-ops
so the main code path never breaks.
"""
from __future__ import annotations

import os
from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import Tracer
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,  # type: ignore
    )

    _OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore
    Tracer = Any  # type: ignore
    _OTEL_AVAILABLE = False


_tracer: Tracer | None = None


def _init_tracer() -> Tracer | None:
    if not _OTEL_AVAILABLE:
        logger.warning("[OBS] OpenTelemetry not installed – tracing disabled")
        return None

    if trace.get_tracer_provider() and not isinstance(
        trace.get_tracer_provider(), trace.NoOpTracerProvider
    ):
        # already configured by app
        return trace.get_tracer(__name__)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    service_name = os.getenv("OTEL_SERVICE_NAME", "power_engine")

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)
    logger.info("[OBS] OpenTelemetry tracer initialised – endpoint=%s service=%s", endpoint, service_name)
    return trace.get_tracer(__name__)


def get_tracer() -> Tracer:
    """Return global tracer (NoOpTracer if OpenTelemetry not available)."""
    global _tracer
    if _tracer is None:
        _tracer = _init_tracer() or _create_noop()
    return _tracer


def _create_noop():  # type: ignore
    class _NoOp:
        def start_as_current_span(self, *_a, **_kw):
            from contextlib import nullcontext

            return nullcontext()

    return _NoOp() 