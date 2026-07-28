"""Locality-sensitive hashing with random hyperplanes (SimHash-style LSH) for
approximate cosine-similarity nearest-neighbor search.

Each hash table projects vectors onto `n_bits` random hyperplanes; the sign of
each projection gives one bit of the hash code. Vectors sharing a hash bucket
in any table are treated as ANN candidates, which are then re-ranked exactly.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np


def _hash_codes(x: np.ndarray, planes: np.ndarray) -> np.ndarray:
    # planes: (n_bits, dim) -> projections: (n, n_bits)
    projections = x @ planes.T
    bits = (projections >= 0).astype(np.uint8)
    # pack bits into an integer per row for fast bucketing, vectorized.
    n_bits = bits.shape[1]
    weights = (1 << np.arange(n_bits)).astype(np.int64)
    return bits @ weights


class LSHIndex:
    """Multi-table random-hyperplane LSH index over a fixed database."""

    def __init__(self, dim: int, n_bits: int = 12, n_tables: int = 6, seed: int = 0):
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables
        rng = np.random.default_rng(seed)
        self.planes: List[np.ndarray] = [rng.normal(size=(n_bits, dim)) for _ in range(n_tables)]
        self.tables: List[dict] = [defaultdict(list) for _ in range(n_tables)]
        self.database: np.ndarray | None = None

    def build(self, database: np.ndarray) -> "LSHIndex":
        database = np.asarray(database, dtype=np.float64)
        self.database = database
        for t in range(self.n_tables):
            codes = _hash_codes(database, self.planes[t])
            table = self.tables[t]
            for i, code in enumerate(codes.tolist()):
                table[code].append(i)
        return self

    def _candidates(self, query: np.ndarray) -> set:
        cand: set = set()
        for t in range(self.n_tables):
            code = int(_hash_codes(query[None, :], self.planes[t])[0])
            cand.update(self.tables[t].get(code, ()))
        return cand

    def search(self, queries: np.ndarray, k: int, metric: str = "cosine"):
        """Return (indices, scores) of shape (n_queries, k), padded with -1 / nan
        when fewer than k candidates were found for a query."""
        from embed_bench.exact import pairwise_scores

        queries = np.asarray(queries, dtype=np.float64)
        n = queries.shape[0]
        out_idx = np.full((n, k), -1, dtype=np.int64)
        out_scores = np.full((n, k), np.nan, dtype=np.float64)

        for i in range(n):
            cand = self._candidates(queries[i])
            if not cand:
                continue
            cand_list = np.array(sorted(cand))
            sub_db = self.database[cand_list]
            scores = pairwise_scores(queries[i : i + 1], sub_db, metric=metric)[0]
            kk = min(k, len(cand_list))
            if metric == "l2":
                order = np.argsort(scores)[:kk]
            else:
                order = np.argsort(-scores)[:kk]
            out_idx[i, :kk] = cand_list[order]
            out_scores[i, :kk] = scores[order]

        return out_idx, out_scores
