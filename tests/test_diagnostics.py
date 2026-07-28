import numpy as np
import pytest

from embed_bench.diagnostics import (
    average_cosine_similarity,
    find_near_duplicates,
    intrinsic_dimensionality,
    isotropy_score,
    pca_explained_variance_ratio,
)


def test_pca_variance_ratio_sums_to_one():
    x = np.random.default_rng(0).normal(size=(100, 5))
    ratios = pca_explained_variance_ratio(x)
    assert ratios.sum() == pytest.approx(1.0)


def test_intrinsic_dimensionality_of_line_is_one():
    # points lying exactly on a 1D line embedded in 5D space
    rng = np.random.default_rng(1)
    t = rng.normal(size=(200, 1))
    direction = np.array([[1.0, 2.0, -1.0, 0.5, 3.0]])
    x = t @ direction
    dim = intrinsic_dimensionality(x, variance_threshold=0.95)
    assert dim == 1


def test_intrinsic_dimensionality_of_full_rank_gaussian_is_full():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 8))
    dim = intrinsic_dimensionality(x, variance_threshold=0.99)
    assert dim == 8


def test_intrinsic_dimensionality_of_plane_is_two():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(300, 2))
    basis = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    x = t @ basis
    dim = intrinsic_dimensionality(x, variance_threshold=0.95)
    assert dim == 2


def test_isotropy_score_of_isotropic_gaussian_near_one():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5000, 6))
    score = isotropy_score(x)
    assert score > 0.7


def test_isotropy_score_of_degenerate_line_near_zero():
    rng = np.random.default_rng(5)
    t = rng.normal(size=(300, 1))
    direction = np.array([[1.0, 0.0, 0.0]])
    noise = rng.normal(scale=1e-4, size=(300, 3))
    x = t @ direction + noise
    score = isotropy_score(x)
    assert score < 0.01


def test_average_cosine_similarity_orthogonal_axes_near_zero():
    x = np.eye(6)
    sim = average_cosine_similarity(x, sample_size=None)
    assert abs(sim) < 1e-9


def test_average_cosine_similarity_identical_directions_near_one():
    rng = np.random.default_rng(6)
    base = rng.normal(size=(1, 5))
    x = np.repeat(base, 50, axis=0) * rng.uniform(0.5, 2.0, size=(50, 1))
    sim = average_cosine_similarity(x, sample_size=None)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_find_near_duplicates_detects_identical_vectors():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    pairs = find_near_duplicates(x, threshold=0.999, metric="cosine")
    found = {(p[0], p[1]) for p in pairs}
    assert (0, 1) in found


def test_find_near_duplicates_no_false_positives_for_orthogonal_vectors():
    x = np.eye(5)
    pairs = find_near_duplicates(x, threshold=0.99, metric="cosine")
    assert pairs == []


def test_find_near_duplicates_l2_metric():
    x = np.array([[0.0, 0.0], [0.0001, 0.0001], [10.0, 10.0]])
    pairs = find_near_duplicates(x, threshold=0.01, metric="l2")
    found = {(p[0], p[1]) for p in pairs}
    assert (0, 1) in found
    assert (0, 2) not in found


def test_find_near_duplicates_sorted_best_first_cosine():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    x = np.vstack([x, x[0] + 1e-6, x[1] + 1e-3])  # near-dup of point0 tighter than point1
    pairs = find_near_duplicates(x, threshold=0.9, metric="cosine")
    if len(pairs) >= 2:
        assert pairs[0][2] >= pairs[-1][2]
