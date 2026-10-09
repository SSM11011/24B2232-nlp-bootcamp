"""Task 1: BM25 baseline tuned on the validation claims.

Usage:  python code/task1_bm25.py

Grid-searches tokenization (lowercasing, stopword removal, stemming) and the
BM25 parameters k1 / b on validation nDCG@10, then writes
predictions/val_bm25.trec and predictions/t1_bm25.trec with the best setting.
The full grid goes to code/logs/task1_bm25_grid.json.
"""
import itertools
import time
from dataclasses import asdict, dataclass

import bm25s
import numpy as np
import Stemmer

from common import (doc_text, load_claims, load_corpus, load_results, relative_path, save_log, set_seed,
                    update_results)
from evaluation import TOP_K, evaluate, run_from_scores, write_trec

K1_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
B_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STEMMERS = [None, "porter", "english"]  # PyStemmer: "english" is Snowball / Porter2
SCORE_METHOD = "lucene"
RUN_NAME = "bm25"


@dataclass(frozen=True)
class TokenizationConfig:
    lowercase: bool
    remove_stopwords: bool
    stemmer: str | None

    @property
    def name(self):
        return (
            f"lower={self.lowercase}, stopwords={'en' if self.remove_stopwords else 'none'}, "
            f"stemmer={self.stemmer or 'none'}"
        )


class BM25Retriever:
    """Tokenizes the corpus once; `search` can then be called for any k1 / b."""

    def __init__(self, corpus, tokenization):
        self.doc_ids = np.array([doc["doc_id"] for doc in corpus])
        self.tokenizer = bm25s.tokenization.Tokenizer(
            lower=tokenization.lowercase,
            stopwords="en" if tokenization.remove_stopwords else None,
            stemmer=Stemmer.Stemmer(tokenization.stemmer) if tokenization.stemmer else None,
        )
        self.corpus_tokens = self.tokenizer.tokenize(
            [doc_text(doc) for doc in corpus], update_vocab=True, return_as="ids", show_progress=False
        )
        self.vocab_size = len(self.tokenizer.get_vocab_dict())

    def _query_tokens(self, text):
        """Token ids of a query, keeping only terms that occur in the corpus vocabulary."""
        [ids] = self.tokenizer.tokenize([text], update_vocab=False, return_as="ids", show_progress=False)
        empty_id = self.tokenizer.word_to_id.get("")
        return [i for i in ids if i != empty_id and i < self.vocab_size]

    def build_index(self, k1, b):
        index = bm25s.BM25(k1=k1, b=b, method=SCORE_METHOD)
        index.index(self.corpus_tokens, show_progress=False)
        return index

    def score_query(self, index, text):
        """BM25 score of every corpus document for one query."""
        tokens = self._query_tokens(text)
        return index.get_scores_from_ids(tokens) if tokens else np.zeros(len(self.doc_ids), dtype=np.float32)

    def score_matrix(self, claims, k1, b):
        """(num_claims, num_docs) BM25 scores, columns in corpus order."""
        index = self.build_index(k1, b)
        return np.stack([self.score_query(index, claim["claim"]) for claim in claims])

    def search(self, claims, k1, b, depth=TOP_K):
        """The `depth` best documents per claim: {claim_id: {doc_id: score}}."""
        scores = self.score_matrix(claims, k1, b)
        return run_from_scores(scores, [c["id"] for c in claims], self.doc_ids, depth)


def load_tuned_bm25(corpus):
    """Rebuild the best BM25 found by this script from results.json: (retriever, k1, b)."""
    config = load_results()["task1"]["details"]["best_config"]
    tokenization = TokenizationConfig(config["lowercase"], config["remove_stopwords"], config["stemmer"])
    return BM25Retriever(corpus, tokenization), config["k1"], config["b"]


def grid_search(corpus, val_claims):
    """Evaluate every (tokenization, k1, b) combination on the validation claims."""
    rows = []
    tokenizations = [
        TokenizationConfig(lowercase, stopwords, stemmer)
        for lowercase, stopwords, stemmer in itertools.product([True, False], [True, False], STEMMERS)
    ]
    for tokenization in tokenizations:
        retriever = BM25Retriever(corpus, tokenization)
        tokenization_rows = []
        for k1, b in itertools.product(K1_VALUES, B_VALUES):
            metrics = evaluate(retriever.search(val_claims, k1, b), val_claims)
            tokenization_rows.append({**asdict(tokenization), "k1": k1, "b": b, **metrics})
        best = max(tokenization_rows, key=lambda r: r["ndcg@10"])
        print(f"  {tokenization.name:50s} best nDCG@10 {best['ndcg@10']:.4f} (k1={best['k1']}, b={best['b']})")
        rows.extend(tokenization_rows)
    return rows


def best_per_tokenization(rows):
    """Tokenization ablation: the best k1 / b row for each tokenization setting."""
    best = {}
    for row in rows:
        key = (row["lowercase"], row["remove_stopwords"], row["stemmer"])
        if key not in best or row["ndcg@10"] > best[key]["ndcg@10"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: -r["ndcg@10"])


def main():
    set_seed()
    corpus = load_corpus()
    val_claims = load_claims("val")
    eval_claims = load_claims("eval")
    train_claims = load_claims("train")

    print("Grid search on validation (nDCG@10):")
    start = time.perf_counter()
    rows = grid_search(corpus, val_claims)
    print(f"{len(rows)} configurations in {time.perf_counter() - start:.0f}s")

    # Ties on nDCG@10 are broken by Recall@100, then MRR.
    best = max(rows, key=lambda r: (r["ndcg@10"], r["recall@100"], r["mrr"]))
    tokenization = TokenizationConfig(best["lowercase"], best["remove_stopwords"], best["stemmer"])
    retriever = BM25Retriever(corpus, tokenization)
    settings = f"bm25s ({SCORE_METHOD}), k1={best['k1']}, b={best['b']}, {tokenization.name}"

    val_run = retriever.search(val_claims, best["k1"], best["b"])
    val_metrics = evaluate(val_run, val_claims)
    train_metrics = evaluate(retriever.search(train_claims, best["k1"], best["b"]), train_claims)
    write_trec(val_run, "val_bm25.trec", RUN_NAME, val_claims)
    write_trec(retriever.search(eval_claims, best["k1"], best["b"]), "t1_bm25.trec", RUN_NAME, eval_claims)

    default_metrics = evaluate(
        BM25Retriever(corpus, TokenizationConfig(True, True, None)).search(val_claims, 1.5, 0.75), val_claims
    )
    ablation = best_per_tokenization(rows)

    print(f"\nBest: {settings}")
    print(f"  val   {val_metrics}")
    print(f"  train {train_metrics}  (sanity check: not used for tuning)")
    print(f"  bm25s defaults (k1=1.5, b=0.75, lower, en stopwords, no stemming) val {default_metrics}")
    print("\nTokenization ablation (best k1/b each):")
    for row in ablation:
        name = TokenizationConfig(row["lowercase"], row["remove_stopwords"], row["stemmer"]).name
        print(f"  {name:50s} nDCG@10 {row['ndcg@10']:.4f}  R@100 {row['recall@100']:.4f}  MRR {row['mrr']:.4f}")

    grid_path = save_log("task1_bm25_grid.json", rows)
    update_results({
        "task1": {
            "bm25_settings": settings,
            "val_ndcg10": val_metrics["ndcg@10"],
            "details": {
                "best_config": {**asdict(tokenization), "k1": best["k1"], "b": best["b"]},
                "library": f"bm25s {bm25s.__version__} + PyStemmer",
                "document_text": 'title + " " + " ".join(abstract)',
                "search_space": {
                    "k1": K1_VALUES,
                    "b": B_VALUES,
                    "lowercase": [True, False],
                    "stopwords": ["en", "none"],
                    "stemmer": [s or "none" for s in STEMMERS],
                },
                "configurations_tried": len(rows),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "val_metrics_bm25s_defaults": default_metrics,
                "tokenization_ablation": ablation,
                "full_grid_log": relative_path(grid_path),
            },
        }
    })
    print(f"\nWrote predictions/val_bm25.trec, predictions/t1_bm25.trec and {grid_path}")


if __name__ == "__main__":
    main()
