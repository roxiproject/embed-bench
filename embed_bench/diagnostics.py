"""Embedding-quality diagnostics that do not require a trained model:
intrinsic dimensionality (PCA explained variance), isotropy/anisotropy, and
duplicate / near-duplicate detection in embedding space.
"""
from __future__ import annotations

import numpy as np

from embed_bench.exact import pairwise_scores


def pca_explained_variance_ratio(x: np.ndarray) -> np.ndarray:
    """Explained-variance ratio of each principal component, descending."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    # SVD on centered data -> singular values relate to eigenvalues of the
    # covariance matrix by eigenvalue_i = s_i^2 / (n - 1).
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s ** 2) / max(x.shape[0] - 1, 1)
    total = eigenvalues.sum()
    if total <= 0:
        return np.zeros_like(eigenvalues)
    return eigenvalues / total


def intrinsic_dimensionality(x: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Smallest number of principal components needed to explain at least
    `variance_threshold` of the total variance."""
    ratios = pca_explained_variance_ratio(x)
    cumulative = np.cumsum(ratios)
    idx = np.searchsorted(cumulative, variance_threshold)
    return int(min(idx + 1, len(ratios)))


def isotropy_score(x: np.ndarray) -> float:
    """Isotropy measured via the ratio of the smallest to largest eigenvalue of
    the (centered) covariance matrix. 1.0 = perfectly isotropic (a sphere),
    values near 0 indicate strong anisotropy (embeddings collapsed onto a
    lower-dimensional cone/subspace, a known embedding-quality pathology).
    """
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s ** 2) / max(x.shape[0] - 1, 1)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    return float(eigenvalues.min() / eigenvalues.max())


def average_cosine_similarity(x: np.ndarray, sample_size: int | None = 2000, seed: int = 0) -> float:
    """Mean pairwise cosine similarity, a common anisotropy diagnostic for
    embedding spaces: values close to 1 mean vectors are clumped in a narrow
    cone (poor isotropy); values near 0 indicate directions are well spread.
    Uses a random sample of pairs for large datasets to stay fully vectorized
    without needing an O(n^2) full matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return 1.0
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        x = x[idx]
        n = sample_size
    sims = pairwise_scores(x, x, metric="cosine")
    mask = ~np.eye(n, dtype=bool)
    return float(sims[mask].mean())


def find_near_duplicates(x: np.ndarray, threshold: float = 0.999, metric: str = "cosine"):
    """Find pairs of near-duplicate vectors (i < j) whose similarity exceeds
    `threshold` (for cosine/dot) or whose distance is below it (for l2).

    Fully vectorized: computes the full (n, n) score matrix once.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    scores = pairwise_scores(x, x, metric=metric)
    iu = np.triu_indices(n, k=1)
    vals = scores[iu]

    if metric == "l2":
        hit_mask = vals <= threshold
    else:
        hit_mask = vals >= threshold

    rows = iu[0][hit_mask]
    cols = iu[1][hit_mask]
    vals = vals[hit_mask]

    order = np.argsort(-vals) if metric != "l2" else np.argsort(vals)
    pairs = [(int(rows[i]), int(cols[i]), float(vals[i])) for i in order]
    return pairs
