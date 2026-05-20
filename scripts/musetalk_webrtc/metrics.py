"""Prometheus metrics for the MuseTalk WebRTC server.

This module provides production-grade observability by exporting key counters,
histograms, and gauges to a /metrics endpoint in Prometheus exposition format.

Graceful degradation: if `prometheus_client` is not installed, all metric
objects become no-op stubs. The server runs identically — just without
metrics export. This keeps the dependency optional for development/testing.

Metrics exported:
- musetalk_inference_seconds (Histogram): Time per inference batch.
- musetalk_frames_published_total (Counter): Total video frames published.
- musetalk_frames_dropped_total (Counter): Frames dropped due to backpressure.
- musetalk_audio_dropped_ms_total (Counter): Audio ms silently dropped by clamp.
- musetalk_webrtc_sessions_active (Gauge): Currently active WebRTC sessions.
- musetalk_webrtc_offers_total (Counter): Total /offer requests processed.
- musetalk_personaplex_reconnects_total (Counter): Bridge reconnect attempts.
- musetalk_gpu_memory_allocated_bytes (Gauge): GPU memory currently allocated.
- musetalk_engine_jobs_total (Counter): Total inference jobs completed.
- musetalk_circuit_breaker_state (Gauge): TURN circuit breaker state (0=closed, 1=open, 2=half_open).

Usage in server code:
    from .metrics import METRICS
    METRICS.inference_seconds.observe(elapsed)
    METRICS.frames_published.inc()
    METRICS.sessions_active.set(len(self.sessions))

Usage for /metrics endpoint:
    from .metrics import metrics_handler
    app.router.add_get("/metrics", metrics_handler)
"""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class _NoOpMetric:
    """Stub metric that does nothing when prometheus_client is unavailable."""

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
        return self


class _MetricsRegistry:
    """Container for all application metrics.

    If prometheus_client is available, these are real Prometheus metric objects.
    Otherwise, they're no-op stubs that accept the same method calls silently.
    """

    def __init__(self) -> None:
        if PROMETHEUS_AVAILABLE:
            self.inference_seconds = Histogram(
                "musetalk_inference_seconds",
                "Time per inference batch (seconds)",
                buckets=(0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
            )
            self.frames_published = Counter(
                "musetalk_frames_published_total",
                "Total video frames published to WebRTC track",
            )
            self.frames_dropped = Counter(
                "musetalk_frames_dropped_total",
                "Video frames dropped due to queue backpressure",
            )
            self.audio_dropped_ms = Counter(
                "musetalk_audio_dropped_ms_total",
                "Audio milliseconds silently dropped by max_advance clamp",
            )
            self.sessions_active = Gauge(
                "musetalk_webrtc_sessions_active",
                "Currently active WebRTC sessions",
            )
            self.offers_total = Counter(
                "musetalk_webrtc_offers_total",
                "Total /offer requests processed (including idempotent hits)",
            )
            self.personaplex_reconnects = Counter(
                "musetalk_personaplex_reconnects_total",
                "PersonaPlex bridge reconnect attempts",
            )
            self.gpu_memory_allocated = Gauge(
                "musetalk_gpu_memory_allocated_bytes",
                "GPU memory currently allocated (bytes)",
            )
            self.engine_jobs = Counter(
                "musetalk_engine_jobs_total",
                "Total inference jobs completed by the engine",
            )
            self.circuit_breaker_state = Gauge(
                "musetalk_circuit_breaker_state",
                "TURN circuit breaker state: 0=closed, 1=open, 2=half_open",
            )
        else:
            self.inference_seconds = _NoOpMetric()
            self.frames_published = _NoOpMetric()
            self.frames_dropped = _NoOpMetric()
            self.audio_dropped_ms = _NoOpMetric()
            self.sessions_active = _NoOpMetric()
            self.offers_total = _NoOpMetric()
            self.personaplex_reconnects = _NoOpMetric()
            self.gpu_memory_allocated = _NoOpMetric()
            self.engine_jobs = _NoOpMetric()
            self.circuit_breaker_state = _NoOpMetric()


# Singleton instance — import this from anywhere in the codebase.
METRICS = _MetricsRegistry()


async def metrics_handler(_request) -> "web.Response":
    """Serve Prometheus metrics in exposition format.

    Returns 501 if prometheus_client is not installed, so monitoring systems
    get a clear signal that metrics are unavailable (vs. a 404 which could
    mean the route doesn't exist at all).
    """
    from aiohttp import web

    if not PROMETHEUS_AVAILABLE:
        return web.Response(
            status=501,
            text="prometheus_client not installed. Install with: pip install prometheus-client\n",
        )
    body = generate_latest()
    return web.Response(
        body=body,
        content_type=CONTENT_TYPE_LATEST,
    )
