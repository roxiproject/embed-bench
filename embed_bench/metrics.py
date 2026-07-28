"""Retrieval quality metrics: recall@k, precision@k, mAP, nDCG.

Every function takes a list of "retrieved" ranked ID lists and either a set
(binary relevance) or a dict of graded relevance scores per query, and
returns the metric averaged across queries unless a single query is passed.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Union

import numpy as np

Relevant = Union[Sequence[Iterable], Iterable]


def _as_list_of_sets(relevant):
    """Normalize a per-query relevant-items spec into a list of sets."""
    out = []
    for r in relevant:
        out.append(set(r))
    return out


def recall_at_k(retrieved: List[List], relevant: List[Iterable], k: int) -> float:
    """Mean fraction of each query's relevant items found in the top-k retrieved."""
    relevant_sets = _as_list_of_sets(relevant)
    if len(retrieved) != len(relevant_sets):
        raise ValueError("retrieved and relevant must have the same number of queries")
    scores = []
    for ret, rel in zip(retrieved, relevant_sets):
        if len(rel) == 0:
            continue
        topk = set(ret[:k])
        scores.append(len(topk & rel) / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(retrieved: List[List], relevant: List[Iterable], k: int) -> float:
    """Mean fraction of the top-k retrieved items that are relevant."""
    relevant_sets = _as_list_of_sets(relevant)
    if len(retrieved) != len(relevant_sets):
        raise ValueError("retrieved and relevant must have the same number of queries")
    scores = []
    for ret, rel in zip(retrieved, relevant_sets):
        topk = ret[:k]
        if len(topk) == 0:
            continue
        hits = sum(1 for item in topk if item in rel)
        scores.append(hits / len(topk))
    return float(np.mean(scores)) if scores else 0.0


def average_precision(ret: List, rel: Iterable, k: int = None) -> float:
    """Average precision for a single query (binary relevance)."""
    rel_set = set(rel)
    if len(rel_set) == 0:
        return 0.0
    items = ret if k is None else ret[:k]
    hits = 0
    precisions = []
    for i, item in enumerate(items, start=1):
        if item in rel_set:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(rel_set))


def mean_average_precision(retrieved: List[List], relevant: List[Iterable], k: int = None) -> float:
    relevant_sets = _as_list_of_sets(relevant)
    if len(retrieved) != len(relevant_sets):
        raise ValueError("retrieved and relevant must have the same number of queries")
    aps = [average_precision(ret, rel, k=k) for ret, rel in zip(retrieved, relevant_sets)]
    return float(np.mean(aps)) if aps else 0.0


def dcg_at_k(ret: List, gains: Dict, k: int) -> float:
    """Discounted cumulative gain for a single query.

    `gains` maps item id -> relevance grade (0 for not relevant / absent).
    """
    items = ret[:k]
    total = 0.0
    for i, item in enumerate(items, start=1):
        g = gains.get(item, 0)
        if g == 0:
            continue
        total += (2 ** g - 1) / np.log2(i + 1)
    return float(total)


def ndcg_at_k(retrieved: List[List], gains: List[Dict], k: int) -> float:
    """Mean normalized DCG@k across queries.

    `gains` is a list (one dict per query) of item id -> relevance grade.
    """
    if len(retrieved) != len(gains):
        raise ValueError("retrieved and gains must have the same number of queries")
    scores = []
    for ret, g in zip(retrieved, gains):
        dcg = dcg_at_k(ret, g, k)
        ideal_order = sorted(g.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_order, start=1))
        if idcg == 0:
            continue
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0
