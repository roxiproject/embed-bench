import numpy as np

from embed_bench.diagnostics import find_near_duplicates, intrinsic_dimensionality, isotropy_score
from embed_bench.exact import search as exact_search
from embed_bench.ivf import IVFIndex
from embed_bench.kmeans import kmeans
from embed_bench.lsh import LSHIndex
from embed_bench.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k


def _clustered_dataset(seed=0, n_clusters=6, n_per_cluster=80, dim=10):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 12
    points, labels = [], []
    for i, c in enumerate(centers):
        pts = c + rng.normal(scale=0.15, size=(n_per_cluster, dim))
        points.append(pts)
        labels.append(np.full(n_per_cluster, i))
    return np.vstack(points), np.concatenate(labels)


def test_end_to_end_lsh_pipeline_uses_metrics_module():
    db, labels = _clustered_dataset(seed=1)
    rng = np.random.default_rng(2)
    q_idx = rng.choice(len(db), size=20, replace=False)
    queries = db[q_idx]
    query_labels = labels[q_idx]

    exact_idx, _ = exact_search(queries, db, k=10, metric="cosine")
    index = LSHIndex(dim=db.shape[1], n_bits=10, n_tables=10, seed=3).build(db)
    ann_idx, _ = index.search(queries, k=10, metric="cosine")

    # relevance = same cluster as the query
    relevant = [set(np.where(labels == ql)[0].tolist()) for ql in query_labels]
    retrieved = [[int(v) for v in row if v != -1] for row in ann_idx]

    recall = recall_at_k(retrieved, relevant, k=10)
    precision = precision_at_k(retrieved, relevant, k=10)
    # each cluster has 80 members but only the top 10 are retrieved, so recall
    # is capped at 10/80 = 0.125; precision is the more informative signal here.
    assert recall > 0.08
    assert precision > 0.5


def test_end_to_end_ivf_pipeline_with_ndcg():
    db, labels = _clustered_dataset(seed=4, n_clusters=5)
    rng = np.random.default_rng(5)
    q_idx = rng.choice(len(db), size=15, replace=False)
    queries = db[q_idx]
    query_labels = labels[q_idx]

    index = IVFIndex(n_cells=10, n_probe=5, seed=6).build(db)
    ann_idx, _ = index.search(queries, k=10, metric="l2")
    retrieved = [[int(v) for v in row if v != -1] for row in ann_idx]

    gains = []
    for ql in query_labels:
        g = {int(idx): (2 if labels[idx] == ql else 0) for idx in range(len(db))}
        gains.append(g)

    score = ndcg_at_k(retrieved, gains, k=10)
    assert score > 0.3


def test_kmeans_feeds_ivf_and_recovers_structure():
    db, labels = _clustered_dataset(seed=7, n_clusters=4)
    result = kmeans(db, k=4, seed=8, n_init=5)
    # cluster purity check
    mapping = {}
    for c in range(4):
        mask = result.labels == c
        if not np.any(mask):
            continue
        vals, counts = np.unique(labels[mask], return_counts=True)
        mapping[c] = vals[np.argmax(counts)]
    remapped = np.array([mapping[c] for c in result.labels])
    assert np.mean(remapped == labels) > 0.95


def test_diagnostics_pipeline_on_clustered_data():
    db, _ = _clustered_dataset(seed=9, n_clusters=3, dim=4)
    dim_est = intrinsic_dimensionality(db, variance_threshold=0.99)
    assert 1 <= dim_est <= 4
    iso = isotropy_score(db)
    assert 0.0 <= iso <= 1.0


def test_diagnostics_detects_injected_near_duplicates():
    db, _ = _clustered_dataset(seed=10, n_clusters=3, dim=6)
    db_with_dup = np.vstack([db, db[0] + 1e-7])
    pairs = find_near_duplicates(db_with_dup, threshold=0.9999, metric="cosine")
    found = {(p[0], p[1]) for p in pairs}
    assert (0, len(db_with_dup) - 1) in found


def test_average_precision_used_directly_on_ann_output():
    db, labels = _clustered_dataset(seed=11, n_clusters=4)
    q = db[0:1]
    idx, _ = exact_search(q, db, k=len(db), metric="cosine")
    relevant = set(np.where(labels == labels[0])[0].tolist())
    ap = average_precision(list(idx[0]), relevant)
    assert ap > 0.9  # exact search should rank same-cluster points first
