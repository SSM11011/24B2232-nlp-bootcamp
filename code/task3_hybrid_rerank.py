"""Task 3: hybrid BM25 + dense retrieval, then cross-encoder reranking.

Usage:  python code/task3_hybrid_rerank.py      (run task1 and task2 first)

Fusion works on full (num_claims, num_docs) score matrices:
  - RRF:           sum_s  w_s / (k + rank_s(doc))
  - interpolation: sum_s  w_s * minmax_s(score),  min-max over each claim's row
Both are tried for BM25 + best dense model and for BM25 + both dense models.
The best of these, by validation nDCG@10, is written to t3_hybrid.trec.

The best first-stage system (BM25, dense or hybrid) is then reranked with
each cross-encoder for k in {20, 50, 100}. The k reranked docs take ranks
1..k, and the remaining first-stage docs keep their order at ranks k+1..100,
with filler scores below the lowest reranked score. The best (model, k) goes
to t3_rerank.trec.
"""
import gc
import itertools
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from common import (doc_text, load_claims, load_corpus, load_results, relative_path, save_log, set_seed,
                    update_results)
from dense import dense_score_matrix, device, for_inference
from evaluation import TOP_K, evaluate, rank_run, run_from_scores, write_trec
from task1_bm25 import load_tuned_bm25

RRF_K_VALUES = [5, 10, 20, 40, 60, 80, 100]
WEIGHT_STEP = 0.1
CROSS_ENCODERS = {
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "MedCPT-Cross-Encoder": "ncbi/MedCPT-Cross-Encoder",
}
RERANK_DEPTHS = [20, 50, 100]
DIAGNOSTIC_DEPTH = 20        # the reference paper reranks a dense baseline's top 20
MAX_PAIR_TOKENS = 512
METRICS = ["ndcg@10", "recall@100", "mrr"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- fusion
def rank_matrix(scores):
    """1-based rank of every doc for every claim (ties broken by corpus order)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, scores.shape[1] + 1), axis=1)
    return ranks


def minmax(scores):
    low = scores.min(axis=1, keepdims=True)
    high = scores.max(axis=1, keepdims=True)
    return (scores - low) / np.maximum(high - low, 1e-9)


def fusion_features(score_matrices):
    """Per system: ranks (for RRF) and per-claim min-max normalized scores (for interpolation)."""
    return {"rank": {s: rank_matrix(m) for s, m in score_matrices.items()},
            "minmax": {s: minmax(m) for s, m in score_matrices.items()}}


def fuse(features, config):
    """Fused (num_claims, num_docs) score matrix for one fusion config."""
    fused = 0.0
    for system, weight in zip(config["systems"], config["weights"]):
        if config["method"] == "rrf":
            fused = fused + weight / (config["k"] + features["rank"][system])
        else:
            fused = fused + weight * features["minmax"][system]
    return fused


def weight_grid(num_systems):
    """Weight vectors with positive entries summing to 1: a 0.1 grid plus plain equal weights."""
    steps = round(1 / WEIGHT_STEP)
    grid = [
        [round(c * WEIGHT_STEP, 2) for c in combo]
        for combo in itertools.product(range(1, steps), repeat=num_systems)
        if sum(combo) == steps
    ]
    equal = [round(1 / num_systems, 4)] * num_systems
    return grid if equal in grid else grid + [equal]


def fusion_configs(system_groups):
    for systems in system_groups:
        for weights in weight_grid(len(systems)):
            yield {"method": "interpolation", "systems": systems, "weights": weights}
            for k in RRF_K_VALUES:
                yield {"method": "rrf", "systems": systems, "weights": weights, "k": k}


def describe(config):
    parts = " + ".join(f"{w}*{s}" for s, w in zip(config["systems"], config["weights"]))
    if config["method"] == "rrf":
        return f"RRF(k={config['k']}): {parts}"
    return f"min-max interpolation: {parts}"


def search_fusion(features, claims, doc_ids, system_groups):
    """Evaluate every fusion config on validation; returns (all rows, best config)."""
    claim_ids = [c["id"] for c in claims]
    rows = []
    for config in fusion_configs(system_groups):
        run = run_from_scores(fuse(features, config), claim_ids, doc_ids)
        rows.append({**config, **evaluate(run, claims)})
    best = max(rows, key=lambda r: tuple(r[m] for m in METRICS))
    best_config = {k: best[k] for k in ["method", "systems", "weights", "k"] if k in best}

    print(f"\nTried {len(rows)} fusion settings. Best per method and system group:")
    for method, systems in itertools.product(["rrf", "interpolation"], system_groups):
        top = max((r for r in rows if r["method"] == method and r["systems"] == systems),
                  key=lambda r: r["ndcg@10"])
        print(f"  {describe(top):70s} nDCG@10 {top['ndcg@10']:.4f}")
    return rows, best_config


# ---------------------------------------------------------------- reranking
def load_cross_encoder(model_id):
    # Raw logits: a sigmoid saturates at 1.0 in float32 and would create ties among the top docs.
    model = for_inference(
        CrossEncoder(model_id, max_length=MAX_PAIR_TOKENS, device=device(), activation_fn=torch.nn.Identity())
    )
    model.predict([("warm-up claim", "warm-up document")])
    return model


def rerank(cross_encoder, first_stage, claims, texts_by_doc, depth):
    """Rerank the top `depth` docs of each claim. Returns (run, mean cross-encoder ms per claim)."""
    run, elapsed = {}, 0.0
    for claim in claims:
        ranked_docs = [doc_id for doc_id, _ in first_stage[claim["id"]]]
        head, tail = ranked_docs[:depth], ranked_docs[depth:]
        pairs = [(claim["claim"], texts_by_doc[doc_id]) for doc_id in head]

        sync()
        start = time.perf_counter()
        scores = cross_encoder.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        sync()
        elapsed += time.perf_counter() - start

        reranked = dict(zip(head, scores.tolist()))
        floor = min(reranked.values()) - 1.0  # filler scores stay strictly below every reranked score
        run[claim["id"]] = {**reranked, **{doc_id: floor - i for i, doc_id in enumerate(tail)}}
    assert_rerank_rule(run, first_stage, depth)
    return run, 1000 * elapsed / len(claims)


def assert_rerank_rule(reranked_run, first_stage, depth):
    """Ranks 1..k hold exactly the first-stage top k; ranks k+1..100 keep first-stage order."""
    for claim_id, ranking in rank_run(reranked_run).items():
        reranked_docs = [doc_id for doc_id, _ in ranking]
        original_docs = [doc_id for doc_id, _ in first_stage[claim_id]]
        assert set(reranked_docs[:depth]) == set(original_docs[:depth]), f"claim {claim_id}: wrong top-{depth} set"
        assert reranked_docs[depth:] == original_docs[depth:], f"claim {claim_id}: tail order changed"


def rerank_grid(first_stage, claims, texts_by_doc):
    """Every cross-encoder at every depth on validation: nDCG and latency."""
    rows = []
    for name, model_id in CROSS_ENCODERS.items():
        cross_encoder = load_cross_encoder(model_id)
        parameters_m = round(sum(p.numel() for p in cross_encoder.parameters()) / 1e6, 1)
        for depth in RERANK_DEPTHS:
            run, latency = rerank(cross_encoder, first_stage, claims, texts_by_doc, depth)
            metrics = evaluate(run, claims)
            rows.append({"model": name, "model_id": model_id, "parameters_m": parameters_m, "k": depth,
                         "query_latency_ms": round(latency, 2), **metrics})
            print(f"  {name:24s} k={depth:<4d} {metrics}  {latency:7.1f} ms/claim")
        del cross_encoder
        free_gpu()
    return rows


# ---------------------------------------------------------------- latency of the first stage
def bm25_latency_ms(retriever, k1, b, claims):
    index = retriever.build_index(k1, b)
    start = time.perf_counter()
    for claim in claims:
        scores = retriever.score_query(index, claim["claim"])
        np.argpartition(-scores, TOP_K)[:TOP_K]
    return 1000 * (time.perf_counter() - start) / len(claims)


def fusion_latency_ms(features, config, num_claims):
    """Per-claim cost of the fusion step alone, applied one claim (row) at a time."""
    start = time.perf_counter()
    for row in range(num_claims):
        row_features = {kind: {s: m[s][row:row + 1] for s in config["systems"]} for kind, m in features.items()}
        np.argpartition(-fuse(row_features, config)[0], TOP_K)[:TOP_K]
    return 1000 * (time.perf_counter() - start) / num_claims


# ---------------------------------------------------------------- main
def main():
    set_seed()
    corpus = load_corpus()
    doc_ids = [doc["doc_id"] for doc in corpus]
    texts_by_doc = {doc["doc_id"]: doc_text(doc) for doc in corpus}
    claims = {split: load_claims(split) for split in ["val", "eval"]}
    val_claims, eval_claims = claims["val"], claims["eval"]
    task2 = load_results()["task2"]

    # First-stage score matrices: BM25 and the two Task 2 dense models, for both splits.
    bm25, k1, b = load_tuned_bm25(corpus)
    dense_names = task2["details"]["selected"]  # {"general": ..., "scientific": ...}
    scores = {split: {"bm25": bm25.score_matrix(claims[split], k1, b),
                      **{family: dense_score_matrix(name, split) for family, name in dense_names.items()}}
              for split in claims}
    features = {split: fusion_features(matrices) for split, matrices in scores.items()}

    def run_for(split, system=None, config=None):
        matrix = scores[split][system] if system else fuse(features[split], config)
        return run_from_scores(matrix, [c["id"] for c in claims[split]], doc_ids)

    single = {system: evaluate(run_for("val", system=system), val_claims) for system in scores["val"]}
    print("Single systems (val):")
    for system, metrics in single.items():
        print(f"  {system:12s} {metrics}")

    # ---- hybrid
    best_dense = max(dense_names, key=lambda family: single[family]["ndcg@10"])
    fusion_rows, fusion_config = search_fusion(features["val"], val_claims, doc_ids,
                                               [["bm25", best_dense], ["bm25", "general", "scientific"]])
    hybrid = evaluate(run_for("val", config=fusion_config), val_claims)
    print(f"Best hybrid: {describe(fusion_config)}  {hybrid}")
    write_trec(run_for("eval", config=fusion_config), "t3_hybrid.trec", "hybrid", eval_claims)

    # ---- rerank the best first-stage system
    first_stage_options = {**single, "hybrid": hybrid}
    first_stage_name = max(first_stage_options, key=lambda s: first_stage_options[s]["ndcg@10"])
    first_stage_ndcg = first_stage_options[first_stage_name]["ndcg@10"]
    print(f"\nReranking first stage: {first_stage_name} (val nDCG@10 {first_stage_ndcg})")

    def first_stage(split):
        if first_stage_name == "hybrid":
            return rank_run(run_for(split, config=fusion_config))
        return rank_run(run_for(split, system=first_stage_name))

    rerank_rows = rerank_grid(first_stage("val"), val_claims, texts_by_doc)
    best_rerank = max(rerank_rows, key=lambda r: (r["ndcg@10"], -r["query_latency_ms"]))
    print(f"Best rerank: {best_rerank['model']} k={best_rerank['k']}  nDCG@10 {best_rerank['ndcg@10']}")

    cross_encoder = load_cross_encoder(best_rerank["model_id"])
    eval_run, _ = rerank(cross_encoder, first_stage("eval"), eval_claims, texts_by_doc, best_rerank["k"])
    write_trec(eval_run, "t3_rerank.trec", "rerank", eval_claims)

    # Diagnostic: the same reranker on each single system's top 20.
    reranker_on_single_systems = {}
    for system in single:
        run, _ = rerank(cross_encoder, rank_run(run_for("val", system=system)), val_claims, texts_by_doc,
                        DIAGNOSTIC_DEPTH)
        reranker_on_single_systems[system] = {"before": single[system]["ndcg@10"],
                                              "after": evaluate(run, val_claims)["ndcg@10"]}
    print(f"{best_rerank['model']} on each single system's top {DIAGNOSTIC_DEPTH}: {reranker_on_single_systems}")
    del cross_encoder
    free_gpu()

    # ---- Q2: gain vs. added latency per claim
    latency = {"bm25": bm25_latency_ms(bm25, k1, b, val_claims), **task2["query_latency_ms"]}
    fusion_ms = fusion_latency_ms(features["val"], fusion_config, len(val_claims))
    hybrid_ms = sum(latency[s] for s in fusion_config["systems"]) + fusion_ms
    best_single = max(single, key=lambda s: single[s]["ndcg@10"])
    reranked_base_ms = hybrid_ms if first_stage_name == "hybrid" else latency[first_stage_name]
    q2 = {
        "best_single_system": best_single,
        "fusion_gain_ndcg10": round(hybrid["ndcg@10"] - single[best_single]["ndcg@10"], 4),
        "fusion_added_ms_per_claim": round(hybrid_ms - latency[best_single], 2),
        "rerank_gain_ndcg10": round(best_rerank["ndcg@10"] - first_stage_ndcg, 4),
        "rerank_added_ms_per_claim": best_rerank["query_latency_ms"],
        "latency_ms_per_claim": {
            **{system: round(ms, 2) for system, ms in latency.items()},
            "fusion_step": round(fusion_ms, 3),
            "hybrid_total": round(hybrid_ms, 2),
            "rerank_pipeline_total": round(reranked_base_ms + best_rerank["query_latency_ms"], 2),
        },
    }
    print(f"\nQ2: {q2}")

    log_path = save_log("task3_hybrid_rerank.json", {"single_systems_val": single, "fusion_grid": fusion_rows,
                                                      "rerank_grid": rerank_rows})
    update_results({
        "task3": {
            "hybrid_method": describe(fusion_config),
            "reranker_model": best_rerank["model_id"],
            "rerank_top_k": best_rerank["k"],
            "val_ndcg10_hybrid": hybrid["ndcg@10"],
            "val_ndcg10_rerank": best_rerank["ndcg@10"],
            "rerank_query_latency_ms": best_rerank["query_latency_ms"],
            "details": {
                "dense_models": dense_names,
                "single_systems_val": single,
                "best_hybrid_config": fusion_config,
                "hybrid_val_metrics": hybrid,
                "fusion_settings_tried": len(fusion_rows),
                "minmax_normalization": "per claim over all corpus documents",
                "reranked_first_stage": first_stage_name,
                "reranker_parameters_m": {r["model"]: r["parameters_m"] for r in rerank_rows},
                "rerank_grid_val": rerank_rows,
                f"best_reranker_top{DIAGNOSTIC_DEPTH}_on_single_systems_val_ndcg10": reranker_on_single_systems,
                "rerank_latency_protocol": "cross-encoder time only, mean ms per validation claim, "
                                           "all k pairs of a claim in one batch, fp16, max 512 tokens",
                "q2": q2,
                "full_log": relative_path(log_path),
            },
        }
    })


if __name__ == "__main__":
    main()
