"""Benchmark harness: measure ANN recall against exact search, and measure
real query latency (mean/p50/p95/p99) at varying index sizes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from embed_bench.exact import search as exact_search


@dataclass
class RecallResult:
    k: int
    recall_at_k: float
    n_queries: int


def recall_vs_exact(
    ann_indices: np.ndarray,
    exact_indices: np.ndarray,
    k: int | None = None,
) -> RecallResult:
    """Compare an ANN method's top-k results against exact top-k results.

    Both inputs are (n_queries, k) index arrays; -1 entries (unfilled ANN
    slots) are ignored.
    """
    if ann_indices.shape[0] != exact_indices.shape[0]:
        raise ValueError("ann_indices and exact_indices must have the same number of queries")
    kk = k or exact_indices.shape[1]
    n = ann_indices.shape[0]
    total_hits = 0
    total_possible = 0
    for i in range(n):
        true_set = set(int(v) for v in exact_indices[i, :kk] if v != -1)
        got_set = set(int(v) for v in ann_indices[i, :kk] if v != -1)
        total_hits += len(true_set & got_set)
        total_possible += len(true_set)
    recall = total_hits / total_possible if total_possible else 0.0
    return RecallResult(k=kk, recall_at_k=recall, n_queries=n)


@dataclass
class LatencyResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_queries: int
    raw_ms: List[float] = field(default_factory=list, repr=False)


def measure_latency(search_fn: Callable[[np.ndarray], object], queries: np.ndarray, repeat: int = 1) -> LatencyResult:
    """Measure real per-query wall-clock latency of `search_fn`, called once
    per query (not batched), across `repeat` passes over the query set.
    """
    timings_ms = []
    for _ in range(repeat):
        for i in range(queries.shape[0]):
            q = queries[i : i + 1]
            start = time.perf_counter()
            search_fn(q)
            elapsed = time.perf_counter() - start
            timings_ms.append(elapsed * 1000.0)

    arr = np.array(timings_ms)
    return LatencyResult(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        n_queries=len(arr),
        raw_ms=timings_ms,
    )


@dataclass
class ScalingPoint:
    index_size: int
    latency: LatencyResult
    recall: RecallResult | None = None


def benchmark_at_scale(
    build_index_fn: Callable[[np.ndarray], object],
    search_fn_factory: Callable[[object], Callable[[np.ndarray], np.ndarray]],
    database: np.ndarray,
    queries: np.ndarray,
    sizes: List[int],
    k: int = 10,
    metric: str = "cosine",
    repeat: int = 1,
) -> List[ScalingPoint]:
    """Build an ANN index at each requested database size (a growing prefix of
    `database`), and measure both recall against exact search and query
    latency at that size.
    """
    results = []
    for size in sizes:
        size = min(size, database.shape[0])
        sub_db = database[:size]
        index = build_index_fn(sub_db)
        search_fn = search_fn_factory(index)

        exact_idx, _ = exact_search(queries, sub_db, k=k, metric=metric)

        def batched(q, _search_fn=search_fn):
            idx, _ = _search_fn(q)
            return idx

        ann_all_idx = np.vstack([batched(queries[i : i + 1]) for i in range(queries.shape[0])])
        recall = recall_vs_exact(ann_all_idx, exact_idx, k=k)

        def single_query_search(q, _search_fn=search_fn):
            return _search_fn(q)

        latency = measure_latency(single_query_search, queries, repeat=repeat)
        results.append(ScalingPoint(index_size=size, latency=latency, recall=recall))

    return results
