"""Day 0: corpus / claim statistics and a side-by-side reading sample.

Usage:  python code/explore_data.py

Prints the statistics, stores them under "data_stats" in results.json and
writes 20 validation claims next to their gold abstracts to
code/logs/claim_samples.md for manual reading.
"""
import random
from collections import Counter

import numpy as np

from common import LOGS_DIR, SEED, doc_text, load_claims, load_corpus, set_seed, update_results

NUM_SAMPLES = 20


def claim_stats(claims, corpus_ids):
    gold_counts = [len({ev["doc_id"] for ev in c["evidence"]}) for c in claims]
    labels = Counter(ev["label"] for c in claims for ev in c["evidence"])
    claims_by_labels = Counter(
        "+".join(sorted({ev["label"] for ev in c["evidence"]})) for c in claims if c["evidence"]
    )
    missing_docs = {ev["doc_id"] for c in claims for ev in c["evidence"]} - corpus_ids
    lengths = [len(c["claim"].split()) for c in claims]
    return {
        "claims": len(claims),
        "nei": sum(n == 0 for n in gold_counts),
        "one_gold_abstract": sum(n == 1 for n in gold_counts),
        "several_gold_abstracts": sum(n > 1 for n in gold_counts),
        "max_gold_abstracts": max(gold_counts),
        "support_pairs": labels["SUPPORT"],
        "contradict_pairs": labels["CONTRADICT"],
        "support_ratio": round(labels["SUPPORT"] / max(1, sum(labels.values())), 3),
        "claims_by_label_set": dict(claims_by_labels),
        "gold_docs_missing_from_corpus": len(missing_docs),
        "mean_claim_words": round(float(np.mean(lengths)), 1),
    }


def corpus_stats(corpus):
    words = [len(doc_text(d).split()) for d in corpus]
    sentences = [len(d["abstract"]) for d in corpus]
    return {
        "documents": len(corpus),
        "mean_words": round(float(np.mean(words)), 1),
        "median_words": int(np.median(words)),
        "max_words": int(np.max(words)),
        "mean_sentences": round(float(np.mean(sentences)), 1),
        "empty_abstracts": sum(n == 0 for n in sentences),
    }


def write_reading_sample(claims, corpus_by_id, path):
    """Write `NUM_SAMPLES` non-NEI claims with their gold abstracts (evidence sentences in bold)."""
    with_evidence = [c for c in claims if c["evidence"]]
    sample = random.Random(SEED).sample(with_evidence, NUM_SAMPLES)

    lines = ["# Validation claims next to their gold abstracts\n"]
    for claim in sorted(sample, key=lambda c: c["id"]):
        lines.append(f"## Claim {claim['id']}: {claim['claim']}\n")
        for ev in claim["evidence"]:
            doc = corpus_by_id[ev["doc_id"]]
            evidence = set(ev["sentences"])
            lines.append(f"**[{ev['label']}] doc {ev['doc_id']}: {doc['title']}**\n")
            for i, sentence in enumerate(doc["abstract"]):
                lines.append(f"- **{sentence}**" if i in evidence else f"- {sentence}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(title, stats):
    print(f"\n{title}")
    print("-" * 60)
    for key, value in stats.items():
        print(f"  {key:32s} {value}")


def main():
    set_seed()
    corpus = load_corpus()
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    stats = {"corpus": corpus_stats(corpus)}
    print_table("corpus", stats["corpus"])
    for split in ["train", "val"]:
        stats[split] = claim_stats(load_claims(split), set(corpus_by_id))
        print_table(f"{split} claims", stats[split])
    stats["eval"] = {"claims": len(load_claims("eval"))}
    print_table("eval claims", stats["eval"])

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = LOGS_DIR / "claim_samples.md"
    write_reading_sample(load_claims("val"), corpus_by_id, sample_path)
    print(f"\nWrote {NUM_SAMPLES} claims with gold abstracts to {sample_path}")

    update_results({"data_stats": stats})


if __name__ == "__main__":
    main()
