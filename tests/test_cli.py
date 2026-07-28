import json
import subprocess
import sys

import numpy as np
import pytest

from embed_bench.cli import build_parser, main


@pytest.fixture
def small_dataset(tmp_path):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 6)) * 10
    db = np.vstack([c + rng.normal(scale=0.2, size=(30, 6)) for c in centers])
    queries = db[rng.choice(len(db), size=8, replace=False)]
    emb_path = tmp_path / "vecs.npy"
    q_path = tmp_path / "queries.npy"
    np.save(emb_path, db)
    np.save(q_path, queries)
    return str(emb_path), str(q_path)


def test_parser_requires_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_eval_defaults():
    parser = build_parser()
    args = parser.parse_args(["eval", "--embeddings", "a.npy", "--queries", "b.npy"])
    assert args.k == 10
    assert args.method == "exact"
    assert args.metric == "cosine"


def test_cli_eval_exact_runs_and_prints_json(small_dataset, capsys):
    emb_path, q_path = small_dataset
    rc = main(["eval", "--embeddings", emb_path, "--queries", q_path, "--k", "3", "--method", "exact"])
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["method"] == "exact"
    assert report["recall_at_k"] == pytest.approx(1.0)


def test_cli_eval_lsh_runs(small_dataset, capsys):
    emb_path, q_path = small_dataset
    rc = main(
        ["eval", "--embeddings", emb_path, "--queries", q_path, "--k", "3", "--method", "lsh", "--lsh-tables", "8"]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert "recall_at_k" in report
    assert 0.0 <= report["recall_at_k"] <= 1.0


def test_cli_eval_ivf_runs(small_dataset, capsys):
    emb_path, q_path = small_dataset
    rc = main(
        ["eval", "--embeddings", emb_path, "--queries", q_path, "--k", "3", "--method", "ivf", "--ivf-cells", "4"]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert 0.0 <= report["recall_at_k"] <= 1.0


def test_cli_eval_with_diagnostics(small_dataset, capsys):
    emb_path, q_path = small_dataset
    rc = main(["eval", "--embeddings", emb_path, "--queries", q_path, "--diagnostics"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert "diagnostics" in report
    assert "intrinsic_dimensionality_95pct" in report["diagnostics"]
    assert "isotropy_score" in report["diagnostics"]


def test_cli_eval_rejects_non_2d_embeddings(tmp_path, capsys):
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, np.zeros(10))
    q_path = tmp_path / "q.npy"
    np.save(q_path, np.zeros((2, 3)))
    rc = main(["eval", "--embeddings", str(bad_path), "--queries", str(q_path)])
    assert rc == 1


def test_cli_entrypoint_via_subprocess(small_dataset):
    emb_path, q_path = small_dataset
    result = subprocess.run(
        [sys.executable, "-m", "embed_bench.cli", "eval", "--embeddings", emb_path, "--queries", q_path, "--k", "2"],
        capture_output=True,
        text=True,
        cwd=str((__import__("pathlib").Path(__file__).parent.parent)),
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["k"] == 2
