"""One evaluation path for every retriever.

A "run" is {claim_id: {doc_id: score}} and may contain any number of scored
documents per claim. `rank_run` turns it into exactly the top-100 ranking
that gets written to disk, and `evaluate` scores that same truncated ranking,
so the numbers we log always match the files we submit.
"""
import sys
import warnings

import numpy as np
from ranx import Qrels, Run, evaluate as ranx_evaluate

from common import PREDICTIONS_DIR, PROJECT_ROOT, gold_qrels, load_corpus

sys.path.insert(0, str(PROJECT_ROOT))
from check_format import check_trec  # noqa: E402  (the official format checker)

# ranx's numba kernels warn about a harmless uint64 -> int64 cast on every call.
warnings.filterwarnings("ignore", message="unsafe cast from uint64 to int64")

TOP_K = 100
METRICS = ["ndcg@10", "recall@100", "mrr"]


def run_from_scores(scores, claim_ids, doc_ids, depth=TOP_K):
    """Turn a (num_claims, num_docs) score matrix into a run of the `depth` best docs per claim.

    Ties are broken by doc_id, so the cut-off at `depth` is deterministic.
    """
    doc_ids = np.asarray(doc_ids)
    run = {}
    for claim_id, row in zip(claim_ids, scores):
        top = np.lexsort((doc_ids, -row))[:depth]
        run[int(claim_id)] = dict(zip(doc_ids[top].tolist(), row[top].tolist()))
    return run


def rank_run(run, top_k=TOP_K):
    """{claim_id: [(doc_id, score), ...]} with exactly `top_k` unique docs per claim.

    Sorted by score (descending); ties are broken by doc_id so the output is
    deterministic.
    """
    ranked = {}
    for claim_id, doc_scores in run.items():
        ordered = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) < top_k:
            raise ValueError(f"claim {claim_id}: only {len(ordered)} scored docs, need {top_k}")
        ranked[claim_id] = ordered[:top_k]
    return ranked


def write_trec(run, filename, run_name, claims):
    """Write the top-100 of `run` in TREC format and validate it with check_format.py."""
    claim_ids = {int(c["id"]) for c in claims}
    missing = claim_ids - set(run)
    if missing:
        raise ValueError(f"run is missing {len(missing)} claims, e.g. {sorted(missing)[:3]}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / filename
    ranked = rank_run({claim_id: run[claim_id] for claim_id in claim_ids})
    with open(path, "w", encoding="utf-8") as f:
        for claim_id in sorted(ranked):
            for rank, (doc_id, score) in enumerate(ranked[claim_id], start=1):
                f.write(f"{claim_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    doc_ids = {int(d["doc_id"]) for d in load_corpus()}
    check_trec(path, claim_ids, doc_ids)  # raises on any format problem
    return path


def read_trec(filename):
    """Read a run back from predictions/: {claim_id: {doc_id: score}}."""
    run = {}
    with open(PREDICTIONS_DIR / filename, encoding="utf-8") as f:
        for line in f:
            claim_id, _, doc_id, _, score, _ = line.split()
            run.setdefault(int(claim_id), {})[int(doc_id)] = float(score)
    return run


def _ranx_inputs(run, claims):
    qrels = gold_qrels(claims)
    ranked = rank_run({claim_id: run[claim_id] for claim_id in qrels})
    ranx_qrels = Qrels({str(q): {str(d): rel for d, rel in docs.items()} for q, docs in qrels.items()})
    ranx_run = Run({str(q): {str(d): float(s) for d, s in docs} for q, docs in ranked.items()})
    return ranx_qrels, ranx_run


def evaluate(run, claims):
    """Mean nDCG@10, Recall@100 and MRR over claims that have gold abstracts."""
    scores = ranx_evaluate(*_ranx_inputs(run, claims), METRICS)
    return {metric: round(float(value), 4) for metric, value in scores.items()}


def per_claim_ndcg10(run, claims):
    """{claim_id: nDCG@10} for every claim that has gold abstracts."""
    qrels, ranx_run = _ranx_inputs(run, claims)
    scores = ranx_evaluate(qrels, ranx_run, "ndcg@10", return_mean=False)
    return {int(q): round(float(s), 4) for q, s in zip(qrels.keys(), scores)}
