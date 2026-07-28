"""K-means clustering implemented from scratch with Lloyd's algorithm.

Used as the coarse quantizer for the IVF index, but usable standalone too.
Centroid assignment is fully vectorized; only the outer iteration loop (which
runs a small, bounded number of times) is a Python loop.
"""
from __future__ import annotations

import numpy as np


def _kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=x.dtype)
    first = rng.integers(n)
    centroids[0] = x[first]
    closest_sq_dist = np.sum((x - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq_dist / max(closest_sq_dist.sum(), 1e-12)
        next_idx = rng.choice(n, p=probs)
        centroids[i] = x[next_idx]
        new_sq_dist = np.sum((x - centroids[i]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)
    return centroids


def _assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # (n, k) squared-distance matrix via broadcasting, no per-point loop.
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)
    c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
    cross = x @ centroids.T
    dist = x_sq + c_sq - 2.0 * cross
    return np.argmin(dist, axis=1)


class KMeansResult:
    def __init__(self, centroids: np.ndarray, labels: np.ndarray, n_iter: int, inertia: float):
        self.centroids = centroids
        self.labels = labels
        self.n_iter = n_iter
        self.inertia = inertia


def kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
    n_init: int = 1,
) -> KMeansResult:
    """Run Lloyd's algorithm to convergence (or max_iter), keeping the best of n_init restarts."""
    x = np.asarray(x, dtype=np.float64)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > x.shape[0]:
        raise ValueError("k cannot exceed the number of points")

    rng = np.random.default_rng(seed)
    best = None

    for init in range(n_init):
        centroids = _kmeans_plus_plus_init(x, k, rng)
        labels = _assign(x, centroids)
        n_iter = 0
        for it in range(max_iter):
            n_iter = it + 1
            new_centroids = centroids.copy()
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = x[mask].mean(axis=0)
                # empty cluster: re-seed it at the point farthest from its centroid
                else:
                    x_sq = np.sum(x ** 2, axis=1)
                    c_sq = np.sum(centroids ** 2, axis=1)
                    dist = x_sq + c_sq[labels] - 2.0 * np.sum(x * centroids[labels], axis=1)
                    new_centroids[j] = x[np.argmax(dist)]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            new_labels = _assign(x, centroids)
            converged = shift < tol and np.array_equal(new_labels, labels)
            labels = new_labels
            if converged:
                break

        x_sq = np.sum(x ** 2, axis=1)
        c_sq = np.sum(centroids ** 2, axis=1)
        dist = x_sq + c_sq[labels] - 2.0 * np.sum(x * centroids[labels], axis=1)
        inertia = float(np.sum(np.maximum(dist, 0.0)))

        if best is None or inertia < best.inertia:
            best = KMeansResult(centroids=centroids, labels=labels, n_iter=n_iter, inertia=inertia)

    return best
