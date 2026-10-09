"""Task 2: dense retrieval with a general-purpose and a scientific bi-encoder.

Usage:  python code/task2_dense.py

Every candidate (all <= 400M parameters) encodes the corpus once. Retrieval is
a matrix product of normalized embeddings, with no FAISS. For each candidate we record:
  - index time: wall-clock to encode the full corpus (after a short warm-up)
  - query latency: ms per claim with batch size 1 (encode + matmul + top-100),
    averaged over the validation claims
  - memory: peak GPU allocation, embedding-matrix size and process RSS
The best general and the best scientific model on validation nDCG@10 are
written to predictions/t2_general.trec, val_general.trec and t2_scientific.trec.
"""
import gc
import os
import time
from dataclasses import asdict

import psutil
import torch

from common import load_claims, load_corpus, relative_path, save_log, set_seed, update_results
from dense import DenseModelSpec, dense_score_matrix, device, load_encoder, save_embeddings
from evaluation import TOP_K, evaluate, run_from_scores, write_trec

RETRIEVAL_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # bge and arctic queries

CANDIDATES = [
    DenseModelSpec("bge-base-en-v1.5", "general", "BAAI/bge-base-en-v1.5", query_prefix=RETRIEVAL_INSTRUCTION),
    DenseModelSpec("bge-large-en-v1.5", "general", "BAAI/bge-large-en-v1.5", query_prefix=RETRIEVAL_INSTRUCTION),
    DenseModelSpec("e5-large-v2", "general", "intfloat/e5-large-v2", query_prefix="query: ", doc_prefix="passage: "),
    DenseModelSpec("snowflake-arctic-embed-l", "general", "Snowflake/snowflake-arctic-embed-l",
                   query_prefix=RETRIEVAL_INSTRUCTION),
    DenseModelSpec("MedCPT", "scientific", "ncbi/MedCPT-Query-Encoder", article_model_id="ncbi/MedCPT-Article-Encoder"),
    DenseModelSpec("S-PubMedBert-MS-MARCO", "scientific", "pritamdeka/S-PubMedBert-MS-MARCO"),
]
SPLITS = ["train", "val", "eval"]
WARMUP_QUERIES = 5
MAX_PARAMETERS = 400e6


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def query_latency_ms(encoder, doc_embeddings, texts):
    """Mean ms per claim, one claim at a time: encode, score all docs, take the top 100."""
    docs = torch.from_numpy(doc_embeddings).float().to(device())

    def retrieve(text):
        query = encoder.encode_queries([text], batch_size=1, as_tensor=True)
        torch.topk(query.float() @ docs.T, TOP_K).indices.cpu()

    for text in texts[:WARMUP_QUERIES]:
        retrieve(text)
    sync()
    start = time.perf_counter()
    for text in texts:
        retrieve(text)
    sync()
    return 1000 * (time.perf_counter() - start) / len(texts)


def encode_and_measure(spec, corpus, claims_by_split):
    """Encode corpus and claims (cached to disk) and return the efficiency measurements."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    encoder = load_encoder(spec)
    parameters = encoder.num_parameters()
    if parameters > MAX_PARAMETERS:
        raise ValueError(f"{spec.name} has {parameters / 1e6:.0f}M parameters (> 400M)")
    encoder.encode_docs(corpus[:64])  # warm-up: CUDA kernels, tokenizer caches

    sync()
    start = time.perf_counter()
    doc_embeddings = encoder.encode_docs(corpus)
    sync()
    index_time = time.perf_counter() - start

    save_embeddings(spec.name, "corpus", doc_embeddings)
    for split, claims in claims_by_split.items():
        save_embeddings(spec.name, split, encoder.encode_queries([c["claim"] for c in claims]))

    latency = query_latency_ms(encoder, doc_embeddings, [c["claim"] for c in claims_by_split["val"]])
    measurements = {
        "parameters_m": round(parameters / 1e6, 1),
        "embedding_dim": int(doc_embeddings.shape[1]),
        "index_time_sec": round(index_time, 2),
        "query_latency_ms": round(latency, 2),
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        "embedding_matrix_mb": round(doc_embeddings.nbytes / 2**20, 2),
        "process_rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1),
    }
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return measurements


def print_summary(results):
    print(f"\n  {'model':26s} {'family':10s} {'params':>7s} {'nDCG@10':>8s} {'R@100':>7s} {'MRR':>7s} "
          f"{'index s':>8s} {'ms/claim':>9s} {'GPU MB':>8s}")
    for r in results:
        m = r["val_metrics"]
        print(f"  {r['name']:26s} {r['family']:10s} {r['parameters_m']:6.0f}M {m['ndcg@10']:8.4f} "
              f"{m['recall@100']:7.4f} {m['mrr']:7.4f} {r['index_time_sec']:8.1f} {r['query_latency_ms']:9.2f} "
              f"{r['gpu_peak_mb']:8.0f}")


def main():
    set_seed()
    corpus = load_corpus()
    doc_ids = [doc["doc_id"] for doc in corpus]
    claims_by_split = {split: load_claims(split) for split in SPLITS}
    val_claims = claims_by_split["val"]

    results = []
    for spec in CANDIDATES:
        print(f"{spec.name} ({spec.family}) ...", flush=True)
        measurements = encode_and_measure(spec, corpus, claims_by_split)
        val_run = run_from_scores(dense_score_matrix(spec.name, "val"), [c["id"] for c in val_claims], doc_ids)
        results.append({**asdict(spec), "val_metrics": evaluate(val_run, val_claims), **measurements})
    print_summary(results)

    best = {
        family: max((r for r in results if r["family"] == family), key=lambda r: r["val_metrics"]["ndcg@10"])
        for family in ["general", "scientific"]
    }
    print(f"\nSelected: general={best['general']['name']}, scientific={best['scientific']['name']}")

    def write(model, split, filename, run_name):
        claims = claims_by_split[split]
        scores = dense_score_matrix(model["name"], split)
        write_trec(run_from_scores(scores, [c["id"] for c in claims], doc_ids), filename, run_name, claims)

    write(best["general"], "eval", "t2_general.trec", "dense_general")
    write(best["general"], "val", "val_general.trec", "dense_general")
    write(best["scientific"], "eval", "t2_scientific.trec", "dense_scientific")

    def model_ids(model):
        return model["model_id"] + (f" + {model['article_model_id']}" if model["article_model_id"] else "")

    log_path = save_log("task2_dense_candidates.json", results)
    general, scientific = best["general"], best["scientific"]
    update_results({
        "gpu_used": torch.cuda.get_device_name() if torch.cuda.is_available() else "none (CPU)",
        "task2": {
            "general_model": model_ids(general),
            "scientific_model": model_ids(scientific),
            "val_ndcg10_general": general["val_metrics"]["ndcg@10"],
            "val_ndcg10_scientific": scientific["val_metrics"]["ndcg@10"],
            "index_time_sec": {"general": general["index_time_sec"], "scientific": scientific["index_time_sec"]},
            "query_latency_ms": {"general": general["query_latency_ms"], "scientific": scientific["query_latency_ms"]},
            "memory_mb": {"general": general["gpu_peak_mb"], "scientific": scientific["gpu_peak_mb"]},
            "details": {
                "selected": {"general": general["name"], "scientific": scientific["name"]},
                "parameters_m": {"general": general["parameters_m"], "scientific": scientific["parameters_m"]},
                "embedding_matrix_mb": {"general": general["embedding_matrix_mb"],
                                        "scientific": scientific["embedding_matrix_mb"]},
                "process_rss_mb": {"general": general["process_rss_mb"], "scientific": scientific["process_rss_mb"]},
                "protocol": {
                    "similarity": "cosine (L2-normalized embeddings, exact matrix product, no FAISS)",
                    "index_time": "wall-clock to encode all 5,183 docs (fp16, batch 32), after a warm-up batch",
                    "query_latency": "mean ms per validation claim at batch size 1: encode + matmul + top-100",
                    "memory_mb": "torch.cuda.max_memory_allocated() during load + encode + search",
                    "medcpt_input": "articles as (title, abstract) pairs, queries max 64 tokens, [CLS] pooling",
                },
                "candidates": [{k: r[k] for k in ["name", "family", "parameters_m", "val_metrics"]} for r in results],
                "candidates_log": relative_path(log_path),
            },
        },
    })


if __name__ == "__main__":
    main()
