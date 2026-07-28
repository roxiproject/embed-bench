"""Brute-force exact nearest-neighbor search used as a ground-truth baseline.

All distance/similarity computations are fully vectorized with numpy -- there is
no Python-level loop over the dataset in the hot path. For n queries and m
database vectors of dimension d, each of the three metrics below computes the
full (n, m) score matrix in a small, fixed number of numpy calls.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_METRICS = ("cosine", "l2", "dot")


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def pairwise_scores(queries: np.ndarray, database: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Return an (n_queries, n_database) score matrix.

    For "cosine" and "dot", higher is more similar. For "l2", lower is closer
    (the matrix contains squared Euclidean distances).
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {SUPPORTED_METRICS}")

    queries = np.asarray(queries, dtype=np.float64)
    database = np.asarray(database, dtype=np.float64)
    if queries.ndim != 2 or database.ndim != 2:
        raise ValueError("queries and database must be 2D arrays")
    if queries.shape[1] != database.shape[1]:
        raise ValueError("queries and database must share the same dimensionality")

    if metric == "dot":
        return queries @ database.T

    if metric == "cosine":
        q = _normalize_rows(queries)
        d = _normalize_rows(database)
        return q @ d.T

    # l2: ||q - d||^2 = ||q||^2 + ||d||^2 - 2 q.d, computed without any
    # per-pair Python loop via broadcasting.
    q_sq = np.sum(queries ** 2, axis=1, keepdims=True)  # (n, 1)
    d_sq = np.sum(database ** 2, axis=1, keepdims=True).T  # (1, m)
    cross = queries @ database.T
    sq_dist = q_sq + d_sq - 2.0 * cross
    return np.maximum(sq_dist, 0.0)


def search(queries: np.ndarray, database: np.ndarray, k: int, metric: str = "cosine"):
    """Exact top-k search.

    Returns (indices, scores), each of shape (n_queries, k). indices[i] are the
    database row indices of the k nearest neighbors of queries[i], ordered from
    best to worst according to the metric.
    """
    scores = pairwise_scores(queries, database, metric=metric)
    n_db = scores.shape[1]
    k = min(k, n_db)

    if metric == "l2":
        # smaller is better
        part = np.argpartition(scores, kth=min(k, n_db - 1) - 1 if k > 0 else 0, axis=1)[:, :k]
    else:
        # larger is better
        part = np.argpartition(-scores, kth=min(k, n_db - 1) - 1 if k > 0 else 0, axis=1)[:, :k]

    row_idx = np.arange(scores.shape[0])[:, None]
    part_scores = scores[row_idx, part]

    if metric == "l2":
        order = np.argsort(part_scores, axis=1)
    else:
        order = np.argsort(-part_scores, axis=1)

    sorted_idx = np.take_along_axis(part, order, axis=1)
    sorted_scores = np.take_along_axis(part_scores, order, axis=1)
    return sorted_idx, sorted_scores
