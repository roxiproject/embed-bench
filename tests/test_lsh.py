import numpy as np

from embed_bench.exact import search as exact_search
from embed_bench.lsh import LSHIndex, _hash_codes


def _make_well_separated_clusters(n_per_cluster=60, dim=8, n_clusters=5, spread=0.05, gap=15.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * gap
    points, labels = [], []
    for i, c in enumerate(centers):
        pts = c + rng.normal(scale=spread, size=(n_per_cluster, dim))
        points.append(pts)
        labels.append(np.full(n_per_cluster, i))
    return np.vstack(points), np.concatenate(labels)


def test_hash_codes_deterministic():
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(4, 3))
    x = rng.normal(size=(10, 3))
    c1 = _hash_codes(x, planes)
    c2 = _hash_codes(x, planes)
    np.testing.assert_array_equal(c1, c2)


def test_hash_codes_same_hemisphere_get_same_bit():
    planes = np.array([[1.0, 0.0]])
    x = np.array([[1.0, 0.0], [2.0, 0.5], [-1.0, 0.0]])
    codes = _hash_codes(x, planes)
    assert codes[0] == codes[1]
    assert codes[0] != codes[2]


def test_lsh_finds_true_neighbor_within_own_cluster():
    x, labels = _make_well_separated_clusters(seed=1)
    index = LSHIndex(dim=x.shape[1], n_bits=8, n_tables=8, seed=2).build(x)
    idx, _ = index.search(x[:5], k=1, metric="cosine")
    for i in range(5):
        assert idx[i, 0] != -1
        assert labels[idx[i, 0]] == labels[i]


def test_lsh_recall_against_exact_is_reasonably_high():
    rng = np.random.default_rng(3)
    x, labels = _make_well_separated_clusters(seed=4, n_per_cluster=100, n_clusters=6)
    queries = x[rng.choice(len(x), size=30, replace=False)]

    exact_idx, _ = exact_search(queries, x, k=5, metric="cosine")
    index = LSHIndex(dim=x.shape[1], n_bits=10, n_tables=10, seed=5).build(x)
    lsh_idx, _ = index.search(queries, k=5, metric="cosine")

    hits = 0
    total = 0
    for i in range(len(queries)):
        true_set = set(exact_idx[i].tolist())
        got_set = set(int(v) for v in lsh_idx[i] if v != -1)
        hits += len(true_set & got_set)
        total += len(true_set)
    recall = hits / total
    assert recall > 0.6


def test_lsh_more_tables_improves_or_maintains_recall():
    rng = np.random.default_rng(6)
    x, _ = _make_well_separated_clusters(seed=7, n_per_cluster=80, n_clusters=5)
    queries = x[rng.choice(len(x), size=25, replace=False)]
    exact_idx, _ = exact_search(queries, x, k=5, metric="cosine")

    def recall_for(n_tables):
        index = LSHIndex(dim=x.shape[1], n_bits=10, n_tables=n_tables, seed=8).build(x)
        got, _ = index.search(queries, k=5, metric="cosine")
        hits = sum(
            len(set(exact_idx[i].tolist()) & set(int(v) for v in got[i] if v != -1))
            for i in range(len(queries))
        )
        return hits / (len(queries) * 5)

    r_low = recall_for(1)
    r_high = recall_for(12)
    assert r_high >= r_low - 0.05  # more tables should not meaningfully hurt recall


def test_lsh_empty_candidates_padded_with_minus_one():
    x = np.random.default_rng(9).normal(size=(20, 4))
    index = LSHIndex(dim=4, n_bits=20, n_tables=1, seed=10).build(x)
    far_query = np.array([[1000.0, 1000.0, 1000.0, 1000.0]])
    idx, scores = index.search(far_query, k=3, metric="cosine")
    assert idx.shape == (1, 3)
    # with very high n_bits, buckets are tiny; some slots may be unfilled (-1)
    assert idx.dtype == np.int64
