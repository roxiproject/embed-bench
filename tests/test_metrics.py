import math

import pytest

from embed_bench.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hand_computed():
    retrieved = [[1, 2, 3, 4, 5]]
    relevant = [{2, 4, 6}]
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_truncates_to_k():
    retrieved = [[1, 2, 3, 4, 5]]
    relevant = [{4, 5}]
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_recall_at_k_averages_across_queries():
    retrieved = [[1, 2], [1, 2]]
    relevant = [{1}, {1, 2, 3}]
    # query1: 1/1 = 1.0, query2: 2/3
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx((1.0 + 2 / 3) / 2)


def test_recall_at_k_skips_empty_relevant():
    retrieved = [[1, 2], [1, 2]]
    relevant = [set(), {1}]
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)


def test_precision_at_k_hand_computed():
    retrieved = [[1, 2, 3, 4, 5]]
    relevant = [{2, 4, 6}]
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)


def test_precision_at_k_smaller_k():
    retrieved = [[1, 2, 3]]
    relevant = [{1, 2}]
    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_average_precision_hand_computed():
    ret = [1, 2, 3, 4, 5]
    rel = {1, 3, 5}
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert average_precision(ret, rel) == pytest.approx(expected)


def test_average_precision_no_relevant_hits():
    assert average_precision([1, 2, 3], {9}) == pytest.approx(0.0)


def test_average_precision_empty_relevant_set():
    assert average_precision([1, 2, 3], set()) == pytest.approx(0.0)


def test_mean_average_precision_across_queries():
    retrieved = [[1, 2, 3], [1, 2, 3]]
    relevant = [{1}, {2}]
    ap1 = 1.0
    ap2 = 1 / 2
    assert mean_average_precision(retrieved, relevant) == pytest.approx((ap1 + ap2) / 2)


def test_dcg_at_k_hand_computed():
    ret = [0, 1, 2]
    gains = {0: 1, 2: 1}
    expected = 1 / math.log2(2) + 1 / math.log2(4)
    assert dcg_at_k(ret, gains, k=3) == pytest.approx(expected)


def test_ndcg_at_k_hand_computed():
    retrieved = [[0, 1, 2]]
    gains = [{0: 1, 2: 1}]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(retrieved, gains, k=3) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_is_one():
    retrieved = [[0, 1, 2]]
    gains = [{0: 3, 1: 2, 2: 1}]
    assert ndcg_at_k(retrieved, gains, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_below_one():
    retrieved = [[2, 1, 0]]
    gains = [{0: 3, 1: 2, 2: 1}]
    assert ndcg_at_k(retrieved, gains, k=3) < 1.0


def test_ndcg_at_k_no_relevance_returns_zero_for_that_query():
    retrieved = [[0, 1, 2]]
    gains = [{}]
    assert ndcg_at_k(retrieved, gains, k=3) == pytest.approx(0.0)


def test_recall_precision_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        recall_at_k([[1, 2]], [], k=2)
    with pytest.raises(ValueError):
        precision_at_k([[1, 2]], [], k=2)
