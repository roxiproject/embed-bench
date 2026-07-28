import numpy as np

from embed_bench.exact import search as exact_search
from embed_bench.ivf import IVFIndex


def _make_well_separated_clusters(n_per_cluster=80, dim=6, n_clusters=6, spread=0.2, gap=25.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-gap, gap, size=(n_clusters, dim))
    for i in range(n_clusters):
        centers[i] += i * gap * 4
    points, labels = [], []
    for i, c in enumerate(centers):
        pts = c + rng.normal(scale=spread, size=(n_per_cluster, dim))
        points.append(pts)
        labels.append(np.full(n_per_cluster, i))
    return np.vstack(points), np.concatenate(labels)


def test_ivf_finds_true_neighbor_within_own_cluster():
    x, labels = _make_well_separated_clusters(seed=1)
    index = IVFIndex(n_cells=6, n_probe=2, seed=2).build(x)
    idx, _ = index.search(x[:5], k=1, metric="l2")
    for i in range(5):
        assert idx[i, 0] != -1
        assert labels[idx[i, 0]] == labels[i]


def test_ivf_full_probe_matches_exact_search():
    x, _ = _make_well_separated_clusters(seed=3, n_clusters=4)
    rng = np.random.default_rng(4)
    queries = x[rng.choice(len(x), size=20, replace=False)]

    index = IVFIndex(n_cells=8, n_probe=8, seed=5).build(x)  # probe all cells
    ivf_idx, _ = index.search(queries, k=5, metric="l2")
    exact_idx, _ = exact_search(queries, x, k=5, metric="l2")

    for i in range(len(queries)):
        assert set(ivf_idx[i].tolist()) == set(exact_idx[i].tolist())


def test_ivf_recall_against_exact_is_high_with_reasonable_probe():
    x, _ = _make_well_separated_clusters(seed=6, n_clusters=8, n_per_cluster=60)
    rng = np.random.default_rng(7)
    queries = x[rng.choice(len(x), size=40, replace=False)]

    exact_idx, _ = exact_search(queries, x, k=5, metric="l2")
    index = IVFIndex(n_cells=16, n_probe=6, seed=8).build(x)
    ivf_idx, _ = index.search(queries, k=5, metric="l2")

    hits = sum(
        len(set(exact_idx[i].tolist()) & set(int(v) for v in ivf_idx[i] if v != -1))
        for i in range(len(queries))
    )
    recall = hits / (len(queries) * 5)
    assert recall > 0.8


def test_ivf_more_probes_improves_or_maintains_recall():
    x, _ = _make_well_separated_clusters(seed=9, n_clusters=10, n_per_cluster=50)
    rng = np.random.default_rng(10)
    queries = x[rng.choice(len(x), size=30, replace=False)]
    exact_idx, _ = exact_search(queries, x, k=5, metric="l2")

    def recall_for(n_probe):
        index = IVFIndex(n_cells=20, n_probe=n_probe, seed=11).build(x)
        got, _ = index.search(queries, k=5, metric="l2")
        hits = sum(
            len(set(exact_idx[i].tolist()) & set(int(v) for v in got[i] if v != -1))
            for i in range(len(queries))
        )
        return hits / (len(queries) * 5)

    assert recall_for(20) >= recall_for(1) - 1e-9


def test_ivf_search_output_shape():
    x, _ = _make_well_separated_clusters(seed=12)
    index = IVFIndex(n_cells=6, n_probe=3, seed=13).build(x)
    idx, scores = index.search(x[:7], k=4, metric="cosine")
    assert idx.shape == (7, 4)
    assert scores.shape == (7, 4)


def test_ivf_handles_n_cells_larger_than_needed():
    x = np.random.default_rng(14).normal(size=(10, 3))
    index = IVFIndex(n_cells=50, n_probe=5, seed=15).build(x)
    idx, _ = index.search(x[:2], k=2, metric="l2")
    assert idx.shape == (2, 2)
