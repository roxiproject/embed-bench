"""Command-line interface for embed-bench.

Example:
    embed-bench eval --embeddings vecs.npy --queries q.npy --k 10 --method lsh
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from embed_bench.benchmark import measure_latency, recall_vs_exact
from embed_bench.diagnostics import (
    average_cosine_similarity,
    find_near_duplicates,
    intrinsic_dimensionality,
    isotropy_score,
)
from embed_bench.exact import search as exact_search
from embed_bench.ivf import IVFIndex
from embed_bench.lsh import LSHIndex

METHODS = ("exact", "lsh", "ivf")


def _build_index(method: str, database: np.ndarray, args: argparse.Namespace):
    if method == "exact":
        return None
    if method == "lsh":
        return LSHIndex(dim=database.shape[1], n_bits=args.lsh_bits, n_tables=args.lsh_tables, seed=args.seed).build(
            database
        )
    if method == "ivf":
        return IVFIndex(n_cells=args.ivf_cells, n_probe=args.ivf_probe, seed=args.seed).build(database)
    raise ValueError(f"unknown method {method!r}")


def _search(method: str, index, queries: np.ndarray, database: np.ndarray, k: int, metric: str):
    if method == "exact":
        return exact_search(queries, database, k=k, metric=metric)
    return index.search(queries, k=k, metric=metric)


def cmd_eval(args: argparse.Namespace) -> int:
    embeddings = np.load(args.embeddings)
    queries = np.load(args.queries)

    if embeddings.ndim != 2:
        print("embeddings file must contain a 2D array", file=sys.stderr)
        return 1
    if queries.ndim != 2:
        print("queries file must contain a 2D array", file=sys.stderr)
        return 1

    index = _build_index(args.method, embeddings, args)

    exact_idx, _ = exact_search(queries, embeddings, k=args.k, metric=args.metric)
    ann_idx, _ = _search(args.method, index, queries, embeddings, args.k, args.metric)
    recall = recall_vs_exact(ann_idx, exact_idx, k=args.k)

    def single_query(q):
        return _search(args.method, index, q, embeddings, args.k, args.metric)

    latency = measure_latency(single_query, queries, repeat=args.repeat)

    report = {
        "method": args.method,
        "metric": args.metric,
        "k": args.k,
        "n_embeddings": int(embeddings.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dim": int(embeddings.shape[1]),
        "recall_at_k": recall.recall_at_k,
        "latency_ms": {
            "mean": latency.mean_ms,
            "p50": latency.p50_ms,
            "p95": latency.p95_ms,
            "p99": latency.p99_ms,
        },
    }

    if args.diagnostics:
        report["diagnostics"] = {
            "intrinsic_dimensionality_95pct": intrinsic_dimensionality(embeddings, 0.95),
            "isotropy_score": isotropy_score(embeddings),
            "average_cosine_similarity": average_cosine_similarity(embeddings),
            "near_duplicate_pairs": len(find_near_duplicates(embeddings, threshold=args.dup_threshold)),
        }

    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed-bench", description="Benchmark vector-embedding retrieval quality.")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Evaluate an ANN method against exact search.")
    eval_p.add_argument("--embeddings", required=True, help="Path to a .npy file of shape (n, d).")
    eval_p.add_argument("--queries", required=True, help="Path to a .npy file of shape (m, d).")
    eval_p.add_argument("--k", type=int, default=10)
    eval_p.add_argument("--method", choices=METHODS, default="exact")
    eval_p.add_argument("--metric", choices=("cosine", "l2", "dot"), default="cosine")
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("--repeat", type=int, default=1)
    eval_p.add_argument("--lsh-bits", type=int, default=12)
    eval_p.add_argument("--lsh-tables", type=int, default=6)
    eval_p.add_argument("--ivf-cells", type=int, default=16)
    eval_p.add_argument("--ivf-probe", type=int, default=4)
    eval_p.add_argument("--diagnostics", action="store_true", help="Also report embedding-quality diagnostics.")
    eval_p.add_argument("--dup-threshold", type=float, default=0.999)
    eval_p.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
