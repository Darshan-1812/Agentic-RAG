"""Small, dependency-free latency and memory benchmark for retrieval callables."""
from __future__ import annotations

import inspect
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


@dataclass
class RetrievalBenchmarkResult:
    """Latency and peak-memory summary for one retrieval callable."""

    iterations: int
    successes: int
    failures: int
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    peak_memory_mb: Optional[float] = None

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "latencies_ms": list(self.latencies_ms),
            "errors": list(self.errors),
        }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def benchmark_retriever(
    retriever: Any,
    query: str,
    *,
    workspace_id: str = "default",
    top_k: int = 5,
    iterations: int = 5,
    warmup: int = 1,
    measure_memory: bool = True,
) -> RetrievalBenchmarkResult:
    """Measure retrieval latency without changing retriever behavior."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    for _ in range(warmup):
        try:
            await _maybe_await(
                retriever(query, workspace_id=workspace_id, top_k=top_k)
            )
        except Exception:  # noqa: BLE001 - warmup failures are reported later
            pass

    started_tracing = False
    if measure_memory and not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracing = True
    if measure_memory:
        tracemalloc.reset_peak()

    latencies: List[float] = []
    errors: List[str] = []
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            try:
                await _maybe_await(
                    retriever(query, workspace_id=workspace_id, top_k=top_k)
                )
            except Exception as exc:  # noqa: BLE001 - benchmark reports failures
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
            else:
                latencies.append((time.perf_counter() - started) * 1000.0)
        peak_memory_mb = None
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / (1024 * 1024)
    finally:
        if started_tracing:
            tracemalloc.stop()

    return RetrievalBenchmarkResult(
        iterations=iterations,
        successes=len(latencies),
        failures=len(errors),
        latencies_ms=latencies,
        errors=errors,
        peak_memory_mb=peak_memory_mb,
    )
