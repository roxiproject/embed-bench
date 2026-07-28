import numpy as np

from embed_bench.kmeans import kmeans


def _make_well_separated_clusters(n_per_cluster=100, dim=4, n_clusters=4, spread=0.3, gap=20.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-gap, gap, size=(n_clusters, dim))
    # push centers apart deterministically so gap is guaranteed
    for i in range(n_clusters):
        centers[i] += i * gap * 3
    points = []
    true_labels = []
    for i, c in enumerate(centers):
        pts = c + rng.normal(scale=spread, size=(n_per_cluster, dim))
        points.append(pts)
        true_labels.append(np.full(n_per_cluster, i))
    return np.vstack(points), np.concatenate(true_labels), centers


def test_kmeans_recovers_well_separated_clusters():
    x, true_labels, true_centers = _make_well_separated_clusters()
    result = kmeans(x, k=4, seed=1, n_init=5)

    # Map each found cluster to the true cluster it overlaps with most, then
    # check that assignment agrees with ground truth almost everywhere.
    found_labels = result.labels
    mapping = {}
    for found_id in range(4):
        mask = found_labels == found_id
        if not np.any(mask):
            continue
        true_ids, counts = np.unique(true_labels[mask], return_counts=True)
        mapping[found_id] = true_ids[np.argmax(counts)]

    remapped = np.array([mapping[f] for f in found_labels])
    accuracy = np.mean(remapped == true_labels)
    assert accuracy > 0.98


def test_kmeans_centroids_close_to_true_centers():
    x, true_labels, true_centers = _make_well_separated_clusters(seed=2)
    result = kmeans(x, k=4, seed=3, n_init=5)

    # each found centroid should be near exactly one true center
    for centroid in result.centroids:
        dists = np.linalg.norm(true_centers - centroid, axis=1)
        assert dists.min() < 1.0


def test_kmeans_converges_within_max_iter():
    x, _, _ = _make_well_separated_clusters(seed=4)
    result = kmeans(x, k=4, seed=5, max_iter=100)
    assert result.n_iter < 100


def test_kmeans_inertia_decreases_with_more_clusters():
    x, _, _ = _make_well_separated_clusters(seed=6, n_clusters=4)
    r2 = kmeans(x, k=2, seed=7, n_init=3)
    r4 = kmeans(x, k=4, seed=7, n_init=3)
    assert r4.inertia < r2.inertia


def test_kmeans_labels_shape_and_range():
    x, _, _ = _make_well_separated_clusters(seed=8)
    result = kmeans(x, k=4, seed=9)
    assert result.labels.shape == (x.shape[0],)
    assert result.labels.min() >= 0
    assert result.labels.max() < 4


def test_kmeans_raises_on_k_too_large():
    x = np.random.default_rng(0).normal(size=(5, 3))
    try:
        kmeans(x, k=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kmeans_single_cluster_matches_mean():
    x = np.random.default_rng(0).normal(size=(50, 3))
    result = kmeans(x, k=1, seed=0)
    np.testing.assert_allclose(result.centroids[0], x.mean(axis=0), atol=1e-8)


def test_kmeans_deterministic_with_fixed_seed():
    x, _, _ = _make_well_separated_clusters(seed=11)
    r1 = kmeans(x, k=4, seed=42, n_init=1)
    r2 = kmeans(x, k=4, seed=42, n_init=1)
    np.testing.assert_array_equal(r1.labels, r2.labels)
