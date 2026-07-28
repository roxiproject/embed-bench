"""Inverted-file (IVF) approximate nearest-neighbor index with k-means coarse
quantization. Vectors are assigned to their nearest centroid ("cell"); a
search probes the `n_probe` closest cells and exactly re-ranks their members.
"""
from __future__ import annotations

import numpy as np

from embed_bench.exact import pairwise_scores
from embed_bench.kmeans import kmeans


class IVFIndex:
    def __init__(self, n_cells: int = 16, n_probe: int = 4, seed: int = 0):
        self.n_cells = n_cells
        self.n_probe = n_probe
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.database: np.ndarray | None = None
        self.cell_members: list[np.ndarray] = []

    def build(self, database: np.ndarray) -> "IVFIndex":
        database = np.asarray(database, dtype=np.float64)
        self.database = database
        n_cells = min(self.n_cells, database.shape[0])
        result = kmeans(database, k=n_cells, seed=self.seed, n_init=3)
        self.centroids = result.centroids
        self.cell_members = [
            np.where(result.labels == c)[0] for c in range(n_cells)
        ]
        return self

    def search(self, queries: np.ndarray, k: int, metric: str = "cosine"):
        queries = np.asarray(queries, dtype=np.float64)
        n = queries.shape[0]
        out_idx = np.full((n, k), -1, dtype=np.int64)
        out_scores = np.full((n, k), np.nan, dtype=np.float64)

        # Vectorized coarse assignment: which cells are closest to each query.
        cell_scores = pairwise_scores(queries, self.centroids, metric="l2")
        n_probe = min(self.n_probe, self.centroids.shape[0])
        probe_cells = np.argsort(cell_scores, axis=1)[:, :n_probe]

        for i in range(n):
            cand = np.concatenate([self.cell_members[c] for c in probe_cells[i]])
            if cand.size == 0:
                continue
            cand = np.unique(cand)
            sub_db = self.database[cand]
            scores = pairwise_scores(queries[i : i + 1], sub_db, metric=metric)[0]
            kk = min(k, len(cand))
            if metric == "l2":
                order = np.argsort(scores)[:kk]
            else:
                order = np.argsort(-scores)[:kk]
            out_idx[i, :kk] = cand[order]
            out_scores[i, :kk] = scores[order]

        return out_idx, out_scores
