import numpy as np
import pytest

from embed_bench.benchmark import (
    benchmark_at_scale,
    measure_latency,
    recall_vs_exact,
)
from embed_bench.exact import search as exact_search
from embed_bench.ivf import IVFIndex
from embed_bench.lsh import LSHIndex


def test_recall_vs_exact_identical_indices_is_one():
    exact_idx = np.array([[0, 1, 2], [3, 4, 5]])
    result = recall_vs_exact(exact_idx.copy(), exact_idx)
    assert result.recall_at_k == pytest.approx(1.0)


def test_recall_vs_exact_partial_overlap():
    exact_idx = np.array([[0, 1, 2]])
    ann_idx = np.array([[0, 9, 9]])
    result = recall_vs_exact(ann_idx, exact_idx)
    assert result.recall_at_k == pytest.approx(1 / 3)


def test_recall_vs_exact_ignores_unfilled_slots():
    exact_idx = np.array([[0, 1, 2]])
    ann_idx = np.array([[0, 1, -1]])
    result = recall_vs_exact(ann_idx, exact_idx)
    assert result.recall_at_k == pytest.approx(2 / 3)


def test_recall_vs_exact_shape_mismatch_raises():
    with pytest.raises(ValueError):
        recall_vs_exact(np.zeros((2, 3), dtype=int), np.zeros((3, 3), dtype=int))


def test_measure_latency_reports_positive_percentiles():
    queries = np.random.default_rng(0).normal(size=(20, 4))

    def fake_search(q):
        return np.zeros((q.shape[0], 1))

    result = measure_latency(fake_search, queries)
    assert result.n_queries == 20
    assert result.mean_ms >= 0
    assert result.p50_ms <= result.p95_ms <= result.p99_ms


def test_measure_latency_respects_repeat_count():
    queries = np.random.default_rng(1).normal(size=(5, 3))

    def fake_search(q):
        return None

    result = measure_latency(fake_search, queries, repeat=3)
    assert result.n_queries == 15


def test_benchmark_at_scale_ivf_reports_recall_and_latency():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(4, 5)) * 20
    db = np.vstack([c + rng.normal(scale=0.3, size=(50, 5)) for c in centers])
    queries = db[rng.choice(len(db), size=10, replace=False)]

    def build(sub_db):
        return IVFIndex(n_cells=4, n_probe=4, seed=3).build(sub_db)

    def search_factory(index):
        def fn(q):
            return index.search(q, k=3, metric="l2")
        return fn

    results = benchmark_at_scale(
        build_index_fn=build,
        search_fn_factory=search_factory,
        database=db,
        queries=queries,
        sizes=[50, 100, 200],
        k=3,
        metric="l2",
    )
    assert len(results) == 3
    assert results[-1].index_size == 200
    for point in results:
        assert point.recall.recall_at_k >= 0.0
        assert point.latency.mean_ms >= 0.0


def test_benchmark_at_scale_lsh_recall_reasonable_at_full_size():
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(4, 6))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 15
    db = np.vstack([c + rng.normal(scale=0.1, size=(60, 6)) for c in centers])
    queries = db[rng.choice(len(db), size=15, replace=False)]

    def build(sub_db):
        return LSHIndex(dim=sub_db.shape[1], n_bits=10, n_tables=10, seed=6).build(sub_db)

    def search_factory(index):
        def fn(q):
            return index.search(q, k=3, metric="cosine")
        return fn

    results = benchmark_at_scale(
        build_index_fn=build,
        search_fn_factory=search_factory,
        database=db,
        queries=queries,
        sizes=[len(db)],
        k=3,
        metric="cosine",
    )
    assert results[0].recall.recall_at_k > 0.5
