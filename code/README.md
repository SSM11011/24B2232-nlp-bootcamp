# SciFact retrieval: reproduction guide

Run every command from the project root (the folder that holds `data/` and `check_format.py`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The GPU runs were done on an NVIDIA Jetson Thor (CUDA 13), using the `cu130` PyTorch wheel.

## Reproduce everything

```bash
bash code/run_all.sh
```

This runs every step below in order, builds `24B2232_nlp_bootcamp.zip` and checks it with `check_format.py`. Each step reads what the earlier ones produced. To run a single step, use `.venv/bin/python code/<step>.py`.

| Step | Script | Writes |
|---|---|---|
| Day 0 | `explore_data.py` | `data_stats` in results.json, `code/logs/claim_samples.md` |
| Task 1 | `task1_bm25.py` | `val_bm25.trec`, `t1_bm25.trec` |
| Task 2 | `task2_dense.py` | `t2_general.trec`, `val_general.trec`, `t2_scientific.trec` |
| Task 3 | `task3_hybrid_rerank.py` | `t3_hybrid.trec`, `t3_rerank.trec` |
| Task 4 | `task4_finetune.py` | `t4_random.trec`, `t4_hard.trec`, `code/logs/task4_curves.png` |
| Task 5 | `task5_verification.py` | `t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv` |
| Q1 | `report_q1.py` | `report_q1` in results.json |

**Reproducibility.** The global seed is `SEED = 42` in `code/common.py`, and every script calls `set_seed()`. Each script replaces its own section of `results.json` with its settings and validation numbers. GPU training is not bit-exact; repeated Task 4 runs differ by about ±0.004 validation nDCG@10.

**Rules followed.**
- Every model is an encoder with ≤ 400M parameters, checked in code, and the counts are recorded in `results.json`. The largest is about 335M.
- No generative LLMs are used.
- All tuning and model selection uses only training claims and the validation split. Eval claims are only encoded and written out.
- Hard negatives and NLI training data come from training claims only.

## Shared code

| File | Purpose |
|---|---|
| `common.py` | Paths, seed, student info, data loading, `doc_text` (`title + " " + " ".join(abstract)`, used by every retriever), and `results.json` logging |
| `evaluation.py` | One evaluation path for every retriever. `write_trec` writes exactly 100 unique docs per claim with non-increasing scores (ties broken by doc_id) and validates the file with `check_format.check_trec`. `evaluate` reports nDCG@10, Recall@100 and MRR (ranx) over claims with gold abstracts, using the same top-100 ranking that goes into the file. |
| `dense.py` | Bi-encoders (sentence-transformers models, plus MedCPT with separate query and article encoders), fp16 inference, and the embedding cache in `cache/embeddings/` |

## Day 0: data exploration

This prints corpus and claim statistics: NEI counts, one vs. several gold abstracts, and the SUPPORT/CONTRADICT ratio. It also writes 20 validation claims next to their gold abstracts, with evidence sentences in bold.

## Task 1: BM25

This uses bm25s with Lucene scoring and grid-searches 768 configurations on validation nDCG@10:
- k1 ∈ {0.6, 0.8, …, 2.0}
- b ∈ {0.3, 0.4, …, 1.0}
- lowercasing on or off
- English stopwords on or off
- stemmer: none, Porter, or Snowball

The full grid goes to `code/logs/task1_bm25_grid.json`. A tokenization ablation goes under `task1` in `results.json`.

## Task 2: dense retrieval

| Family | Candidates | Query / document prefixes |
|---|---|---|
| General | bge-base-en-v1.5, bge-large-en-v1.5, e5-large-v2, snowflake-arctic-embed-l | bge/arctic: retrieval instruction on queries; e5: `query: ` / `passage: ` |
| Scientific | MedCPT (query + article encoder), S-PubMedBert-MS-MARCO | MedCPT articles are fed as (title, abstract) pairs |

Retrieval details:
- Embeddings are L2-normalized and retrieval is an exact matrix product (no FAISS).
- Inference runs in fp16 on the GPU, about 8× faster than fp32 here, with embeddings at cosine ≥ 0.9997 to fp32.

For each model the script logs:
- index time (wall-clock to encode the corpus)
- query latency (batch size 1: encode, matmul and top-100)
- peak GPU memory (`torch.cuda.max_memory_allocated`)
- embedding-matrix size and process RSS (psutil)

The best general and the best scientific model on validation write the TREC files. All candidates are logged to `code/logs/task2_dense_candidates.json`.

## Task 3: hybrid and rerank

**Hybrid.** Fusion is tried for BM25 + the best dense model and for BM25 + both dense models, with two methods:
- RRF with k ∈ {5, 10, 20, 40, 60, 80, 100}
- min-max-normalized score interpolation

Both methods search per-system weights on a 0.1 grid, plus equal weights. The best on validation writes `t3_hybrid.trec`.

**Rerank.** The best first-stage system is reranked with three cross-encoders (ms-marco-MiniLM-L-6-v2, bge-reranker-base, MedCPT-Cross-Encoder) for k ∈ {20, 50, 100}.
- The reranked k docs take ranks 1..k.
- The remaining docs keep their first-stage order at ranks k+1..100, with filler scores below the lowest reranked score.
- This rule is asserted in code for every reranked run.
- The best (model, k) writes `t3_rerank.trec`.

Latencies (ms per claim) and the fusion-vs-rerank comparison for Q2 are under `task3.details` in `results.json`.

## Task 4: fine-tuning

The best Task 2 general model is fine-tuned on (claim, gold abstract) pairs from the training claims only. NEI claims are skipped, which leaves 440 pairs.

Training setup:
- `MultipleNegativesRankingLoss` with scale 50 (temperature 0.02), which adds in-batch negatives
- `NO_DUPLICATES` batch sampler, so a claim's other gold abstract is never an in-batch negative
- lr 1e-5, batch 32, 3 epochs, bf16

There are two kinds of run:
- **random**: one random non-gold abstract per pair.
- **hard**: two mined negatives per pair. One comes from the base dense model via `mine_hard_negatives`, the other from the tuned BM25 using the same rules. Each miner drops gold abstracts and candidates within `relative_margin=0.05` of the gold score, then skips the top 1 or 3 survivors. One hard run is trained per skip value, and the better one on validation becomes `t4_hard.trec`. Samples of mined rows for manual review are in `code/logs/task4_hard_negative_review_skip*.md`.

Validation nDCG@10 (full corpus) is computed before training and every 3 steps, and the best checkpoint is kept. The loss and validation curves (the Q3 plot) are in `code/logs/task4_curves.png`.

An earlier attempt (bge-large, scale 20, lr 2e-5, 4 epochs) overfit. It is kept for reference in `code/logs/task4_ablation_scale20_lr2e-5*` and is not regenerated by `run_all.sh`.

## Task 5: verification and error analysis

**Pipeline.**
1. Take the top 3 abstracts from the best retriever by validation nDCG@10.
2. Run NLI per sentence (premise = sentence, hypothesis = claim).
3. Keep each abstract's max entailment and max contradiction probability.
4. Predict SUPPORT or CONTRADICT only above a threshold.

The thresholds, how many of the top 3 to consider, and whether to force one label per claim are all tuned on validation F1.

**NLI models.** Three zero-shot MNLI models are compared. The best one is then fine-tuned on training claims:
- gold rationale sentences give entail/contradict examples
- sentences from retrieved non-gold abstracts and non-rationale sentences of gold abstracts give neutral ones

The model with the best validation F1 writes the two verification CSVs.

**Errors.** Every validation claim whose predicted (doc, label) set differs from gold gets exactly one category in `t5_val_errors.csv`. Rules are applied in priority order:
1. retrieval_miss: no gold doc in the top 3
2. needs_multiple_docs: several gold docs, only some retrieved or correct
3. numerical: standalone numbers, %, or comparatives
4. negation: a negation cue and a flipped label
5. entity_mismatch: a predicted non-gold abstract lacks a gene/drug-like name from the claim
6. other

The full error list is in `code/logs/task5_verification.json`.

## Report

`report.md` answers Q1–Q5. `report_q1.py` computes per-claim nDCG@10 from `val_bm25.trec` and `val_general.trec`. Its hand-picked examples are verified in code to be real wins.
