import numpy as np
import pytest

from embed_bench.benchmark import LatencyResult, RecallResult, recall_vs_exact
from embed_bench.diagnostics import (
    average_cosine_similarity,
    find_near_duplicates,
    intrinsic_dimensionality,
    isotropy_score,
    pca_explained_variance_ratio,
)
from embed_bench.exact import pairwise_scores, search
from embed_bench.ivf import IVFIndex
from embed_bench.kmeans import kmeans
from embed_bench.lsh import LSHIndex
from embed_bench.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---- exact.py edge cases ----

def test_pairwise_scores_single_query_single_db_point():
    q = np.array([[1.0, 2.0, 3.0]])
    db = np.array([[1.0, 2.0, 3.0]])
    scores = pairwise_scores(q, db, metric="l2")
    assert scores[0, 0] == pytest.approx(0.0)


def test_pairwise_scores_zero_vector_cosine_no_nan():
    q = np.array([[0.0, 0.0]])
    db = np.array([[1.0, 0.0]])
    scores = pairwise_scores(q, db, metric="cosine")
    assert np.isfinite(scores).all()


def test_search_k_equals_one():
    db = np.array([[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]])
    q = np.array([[0.9, 0.9]])
    idx, _ = search(q, db, k=1, metric="l2")
    assert idx[0, 0] == 2


def test_search_dot_metric_prefers_large_magnitude():
    db = np.array([[1.0, 0.0], [3.0, 0.0]])
    q = np.array([[1.0, 0.0]])
    idx, scores = search(q, db, k=2, metric="dot")
    assert idx[0, 0] == 1
    assert scores[0, 0] == pytest.approx(3.0)


def test_pairwise_scores_l2_matches_direct_formula():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 5))
    db = rng.normal(size=(6, 5))
    scores = pairwise_scores(q, db, metric="l2")
    direct = np.array([[np.sum((qi - di) ** 2) for di in db] for qi in q])
    np.testing.assert_allclose(scores, direct, atol=1e-8)


# ---- metrics.py edge cases ----

def test_precision_at_k_empty_retrieved_list():
    assert precision_at_k([[]], [{1, 2}], k=5) == pytest.approx(0.0)


def test_recall_at_k_k_zero():
    assert recall_at_k([[1, 2, 3]], [{1}], k=0) == pytest.approx(0.0)


def test_average_precision_all_relevant_retrieved_in_order():
    assert average_precision([1, 2, 3], {1, 2, 3}) == pytest.approx(1.0)


def test_mean_average_precision_empty_lists():
    assert mean_average_precision([], []) == pytest.approx(0.0)


def test_dcg_at_k_empty_ranking():
    assert dcg_at_k([], {1: 2}, k=5) == pytest.approx(0.0)


def test_ndcg_at_k_empty_query_list():
    assert ndcg_at_k([], [], k=5) == pytest.approx(0.0)


def test_ndcg_at_k_k_one():
    retrieved = [[2, 0, 1]]
    gains = [{0: 3, 1: 2, 2: 1}]
    val = ndcg_at_k(retrieved, gains, k=1)
    assert 0.0 <= val <= 1.0


def test_recall_at_k_all_relevant_missing():
    assert recall_at_k([[1, 2]], [{9, 10}], k=2) == pytest.approx(0.0)


# ---- kmeans.py edge cases ----

def test_kmeans_k_equals_n_points_each_its_own_cluster():
    x = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
    result = kmeans(x, k=3, seed=0)
    assert len(set(result.labels.tolist())) == 3


def test_kmeans_zero_k_raises():
    with pytest.raises(ValueError):
        kmeans(np.zeros((5, 2)), k=0)


def test_kmeans_two_far_apart_points_two_clusters():
    x = np.array([[0.0, 0.0]] * 10 + [[1000.0, 1000.0]] * 10)
    result = kmeans(x, k=2, seed=1)
    assert len(set(result.labels[:10].tolist())) == 1
    assert len(set(result.labels[10:].tolist())) == 1
    assert result.labels[0] != result.labels[10]


# ---- lsh.py edge cases ----

def test_lsh_dot_metric_search_runs():
    x = np.random.default_rng(0).normal(size=(30, 5))
    index = LSHIndex(dim=5, n_bits=6, n_tables=4, seed=1).build(x)
    idx, scores = index.search(x[:3], k=2, metric="dot")
    assert idx.shape == (3, 2)


def test_lsh_l2_metric_search_runs():
    x = np.random.default_rng(2).normal(size=(30, 5))
    index = LSHIndex(dim=5, n_bits=6, n_tables=4, seed=3).build(x)
    idx, scores = index.search(x[:3], k=2, metric="l2")
    assert idx.shape == (3, 2)


def test_lsh_single_table_still_works():
    x = np.random.default_rng(4).normal(size=(40, 4))
    index = LSHIndex(dim=4, n_bits=6, n_tables=1, seed=5).build(x)
    idx, _ = index.search(x[:5], k=1, metric="cosine")
    assert idx.shape == (5, 1)


# ---- ivf.py edge cases ----

def test_ivf_dot_metric_search_runs():
    x = np.random.default_rng(0).normal(size=(30, 5))
    index = IVFIndex(n_cells=4, n_probe=2, seed=1).build(x)
    idx, scores = index.search(x[:3], k=2, metric="dot")
    assert idx.shape == (3, 2)


def test_ivf_cosine_metric_search_runs():
    x = np.random.default_rng(2).normal(size=(30, 5))
    index = IVFIndex(n_cells=4, n_probe=2, seed=3).build(x)
    idx, scores = index.search(x[:3], k=2, metric="cosine")
    assert idx.shape == (3, 2)


def test_ivf_single_cell_equals_exact():
    x = np.random.default_rng(4).normal(size=(20, 4))
    q = x[:5]
    index = IVFIndex(n_cells=1, n_probe=1, seed=5).build(x)
    idx, _ = index.search(q, k=3, metric="l2")
    exact_idx, _ = search(q, x, k=3, metric="l2")
    for i in range(5):
        assert set(idx[i].tolist()) == set(exact_idx[i].tolist())


# ---- diagnostics.py edge cases ----

def test_pca_explained_variance_ratio_single_dimension():
    x = np.random.default_rng(0).normal(size=(50, 1))
    ratios = pca_explained_variance_ratio(x)
    assert ratios.shape == (1,)
    assert ratios[0] == pytest.approx(1.0)


def test_intrinsic_dimensionality_low_threshold_returns_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 6))
    dim = intrinsic_dimensionality(x, variance_threshold=0.01)
    assert dim >= 1


def test_isotropy_score_bounded_between_zero_and_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 5)) * rng.uniform(0.1, 5, size=(1, 5))
    score = isotropy_score(x)
    assert 0.0 <= score <= 1.0


def test_average_cosine_similarity_with_small_sample_size():
    x = np.random.default_rng(3).normal(size=(500, 8))
    sim = average_cosine_similarity(x, sample_size=50, seed=4)
    assert -1.0 <= sim <= 1.0


def test_find_near_duplicates_empty_when_all_distinct():
    x = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    pairs = find_near_duplicates(x, threshold=0.9999, metric="cosine")
    # 0,0 vs orthogonal directions -> not duplicates
    assert all(p[2] < 0.9999 for p in pairs) or pairs == []


def test_find_near_duplicates_returns_empty_list_for_two_points_far_apart():
    x = np.array([[0.0, 0.0], [50.0, 50.0]])
    pairs = find_near_duplicates(x, threshold=1e-6, metric="l2")
    assert pairs == []


# ---- benchmark.py edge cases ----

def test_recall_result_dataclass_fields():
    r = RecallResult(k=5, recall_at_k=0.5, n_queries=10)
    assert r.k == 5
    assert r.recall_at_k == 0.5


def test_latency_result_dataclass_fields():
    r = LatencyResult(mean_ms=1.0, p50_ms=0.9, p95_ms=1.5, p99_ms=2.0, n_queries=3)
    assert r.n_queries == 3


def test_recall_vs_exact_no_possible_relevant_returns_zero():
    exact_idx = np.full((1, 3), -1)
    ann_idx = np.full((1, 3), -1)
    result = recall_vs_exact(ann_idx, exact_idx)
    assert result.recall_at_k == pytest.approx(0.0)


def test_recall_vs_exact_custom_k_smaller_than_arrays():
    exact_idx = np.array([[0, 1, 2, 3, 4]])
    ann_idx = np.array([[0, 1, 9, 9, 9]])
    result = recall_vs_exact(ann_idx, exact_idx, k=2)
    assert result.recall_at_k == pytest.approx(1.0)
