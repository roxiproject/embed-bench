import json

import numpy as np
import pytest

from embed_bench.cli import build_parser, main
from embed_bench.metrics import ndcg_at_k, precision_at_k, recall_at_k


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(42)
    db = rng.normal(size=(60, 4))
    q = db[:6]
    emb_path = tmp_path / "e.npy"
    q_path = tmp_path / "q.npy"
    np.save(emb_path, db)
    np.save(q_path, q)
    return str(emb_path), str(q_path)


def test_cli_eval_custom_metric_l2(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(["eval", "--embeddings", emb_path, "--queries", q_path, "--metric", "l2", "--k", "5"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metric"] == "l2"


def test_cli_eval_custom_metric_dot(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(["eval", "--embeddings", emb_path, "--queries", q_path, "--metric", "dot", "--k", "5"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metric"] == "dot"


def test_cli_eval_repeat_multiplies_latency_samples(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(["eval", "--embeddings", emb_path, "--queries", q_path, "--repeat", "2"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert "latency_ms" in report
    assert set(report["latency_ms"].keys()) == {"mean", "p50", "p95", "p99"}


def test_cli_eval_ivf_custom_cells_and_probe(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(
        [
            "eval",
            "--embeddings",
            emb_path,
            "--queries",
            q_path,
            "--method",
            "ivf",
            "--ivf-cells",
            "5",
            "--ivf-probe",
            "3",
        ]
    )
    assert rc == 0


def test_cli_eval_lsh_custom_bits(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(
        ["eval", "--embeddings", emb_path, "--queries", q_path, "--method", "lsh", "--lsh-bits", "8"]
    )
    assert rc == 0


def test_cli_eval_diagnostics_dup_threshold(dataset, capsys):
    emb_path, q_path = dataset
    rc = main(
        ["eval", "--embeddings", emb_path, "--queries", q_path, "--diagnostics", "--dup-threshold", "0.5"]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert isinstance(report["diagnostics"]["near_duplicate_pairs"], int)


def test_cli_eval_rejects_non_2d_queries(tmp_path):
    emb_path = tmp_path / "e.npy"
    q_path = tmp_path / "q.npy"
    np.save(emb_path, np.zeros((5, 3)))
    np.save(q_path, np.zeros(3))
    rc = main(["eval", "--embeddings", str(emb_path), "--queries", str(q_path)])
    assert rc == 1


def test_parser_method_choices_enforced():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--embeddings", "a", "--queries", "b", "--method", "bogus"])


def test_parser_metric_choices_enforced():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--embeddings", "a", "--queries", "b", "--metric", "bogus"])


# ---- additional metrics correctness checks ----

def test_recall_precision_agree_when_k_covers_all_relevant():
    retrieved = [[3, 1, 2, 4]]
    relevant = [{1, 2, 3}]
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(3 / 4)


def test_ndcg_monotonic_with_better_ranking():
    gains = [{0: 3, 1: 1, 2: 0}]
    good = ndcg_at_k([[0, 1, 2]], gains, k=3)
    bad = ndcg_at_k([[2, 1, 0]], gains, k=3)
    assert good >= bad


def test_precision_at_k_zero_when_nothing_relevant():
    assert precision_at_k([[1, 2, 3]], [{9}], k=3) == pytest.approx(0.0)


def test_recall_at_k_full_when_k_large_enough():
    retrieved = [[5, 4, 3, 2, 1]]
    relevant = [{1, 2}]
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)
