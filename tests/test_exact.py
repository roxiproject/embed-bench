import numpy as np
import pytest

from embed_bench.exact import pairwise_scores, search


def test_dot_product_scores_hand_computed():
    q = np.array([[1.0, 0.0]])
    db = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    scores = pairwise_scores(q, db, metric="dot")
    np.testing.assert_allclose(scores, [[1.0, 0.0, 2.0]])


def test_cosine_scores_hand_computed():
    q = np.array([[1.0, 0.0]])
    db = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]])
    scores = pairwise_scores(q, db, metric="cosine")
    np.testing.assert_allclose(scores, [[1.0, 0.0, 1.0, -1.0]], atol=1e-10)


def test_l2_scores_hand_computed():
    q = np.array([[0.0, 0.0]])
    db = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    scores = pairwise_scores(q, db, metric="l2")
    np.testing.assert_allclose(scores, [[25.0, 0.0, 2.0]])


def test_search_returns_best_first_cosine():
    db = np.eye(4)
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    idx, scores = search(q, db, k=2, metric="cosine")
    assert idx[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)


def test_search_returns_best_first_l2():
    db = np.array([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0], [5.0, 5.0]])
    q = np.array([[0.0, 0.0]])
    idx, _ = search(q, db, k=2, metric="l2")
    assert list(idx[0]) == [0, 2]


def test_search_k_larger_than_database_clamped():
    db = np.random.default_rng(0).normal(size=(3, 4))
    q = np.random.default_rng(1).normal(size=(2, 4))
    idx, scores = search(q, db, k=10, metric="l2")
    assert idx.shape == (2, 3)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        pairwise_scores(np.zeros((2, 3)), np.zeros((2, 4)), metric="cosine")


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        pairwise_scores(np.zeros((2, 3)), np.zeros((2, 3)), metric="manhattan")


def test_search_recovers_exact_neighbors_random():
    rng = np.random.default_rng(42)
    db = rng.normal(size=(200, 16))
    q = db[:5] + 1e-6  # queries very close to specific db rows
    idx, _ = search(q, db, k=1, metric="l2")
    assert list(idx[:, 0]) == [0, 1, 2, 3, 4]


def test_search_ordering_is_monotonic_l2():
    rng = np.random.default_rng(7)
    db = rng.normal(size=(50, 8))
    q = rng.normal(size=(3, 8))
    _, scores = search(q, db, k=10, metric="l2")
    for row in scores:
        assert np.all(np.diff(row) >= -1e-9)


def test_search_ordering_is_monotonic_cosine():
    rng = np.random.default_rng(8)
    db = rng.normal(size=(50, 8))
    q = rng.normal(size=(3, 8))
    _, scores = search(q, db, k=10, metric="cosine")
    for row in scores:
        assert np.all(np.diff(row) <= 1e-9)
