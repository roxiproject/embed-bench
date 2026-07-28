# embed-bench

## embed-bench

A numpy-based benchmarking harness for evaluating vector-embedding quality and
retrieval performance. No `faiss`/`annoy` dependency — the core search
algorithms (exact brute-force search, LSH, IVF with from-scratch k-means) are
implemented directly on top of numpy.

### Features

- **Exact nearest-neighbor search** — vectorized brute-force cosine, L2, and
  dot-product search used as the ground-truth baseline.
- **Approximate nearest-neighbor search**
  - `LSHIndex` — random-hyperplane locality-sensitive hashing (multi-table
    SimHash-style) for approximate cosine search.
  - `IVFIndex` — inverted-file index with a from-scratch Lloyd's-algorithm
    k-means coarse quantizer; searches probe the nearest `n_probe` cells and
    exactly re-rank the candidates within them.
- **Retrieval metrics** — `recall@k`, `precision@k`, mean average precision
  (mAP), and nDCG, each covered by hand-computed test cases.
- **Benchmark harness** — measure an ANN method's `recall@k` against exact
  search, and measure real per-query latency (mean/p50/p95/p99), including at
  varying index sizes.
- **Embedding-quality diagnostics** — intrinsic dimensionality via PCA
  explained variance, isotropy/anisotropy (eigenvalue ratio and average
  pairwise cosine similarity), and near-duplicate detection.

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### CLI usage

```bash
embed-bench eval --embeddings vecs.npy --queries q.npy --k 10 --method lsh
```

Options include `--method {exact,lsh,ivf}`, `--metric {cosine,l2,dot}`,
`--lsh-bits`, `--lsh-tables`, `--ivf-cells`, `--ivf-probe`, `--repeat` (for
latency sampling), and `--diagnostics` to also report embedding-quality
diagnostics for the input embeddings.

### Library usage

```python
import numpy as np
from embed_bench.exact import search
from embed_bench.lsh import LSHIndex
from embed_bench.metrics import recall_at_k

db = np.load("vecs.npy")
queries = np.load("q.npy")

exact_idx, _ = search(queries, db, k=10, metric="cosine")

index = LSHIndex(dim=db.shape[1], n_bits=12, n_tables=8).build(db)
ann_idx, _ = index.search(queries, k=10, metric="cosine")
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

K-means and LSH correctness are verified against constructed, well-separated
clusters — the tests check that the algorithms actually recover the cluster
structure (label agreement, centroid proximity, and ANN recall against exact
search), not just that the code runs.

## Related projects

Part of the roxiproject ML/research thread:

- [attention](https://github.com/roxiproject/attention) — attention/KV-cache implementations, verified bit-for-bit against a full forward pass.
- [attention-probe-kit](https://github.com/roxiproject/attention-probe-kit) — instruments attention heads to extract/visualize what a probe attends to.
- [probe-experiments](https://github.com/roxiproject/probe-experiments) — linear/non-linear probing experiments over model activations.
- [lora-kit](https://github.com/roxiproject/lora-kit) — LoRA fine-tuning utilities, gradient-checked against dense-layer baselines.
- [corpus-kit / corpus-bench / corpus-tokenizer-kit](https://github.com/roxiproject) — corpus/tokenizer pipeline.
- [rlhf-experiments / rlhf-distill-experiments](https://github.com/roxiproject) — RLHF training/distillation.
- [roxiproject](https://github.com/roxiproject/roxiproject) — account root / full project index.
