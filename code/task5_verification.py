"""Task 5: claim verification with sentence-level NLI, plus error analysis.

Usage:  python code/task5_verification.py      (run tasks 1-4 first)

Pipeline per claim:
  1. Take the top 3 abstracts from the best retriever (highest validation nDCG@10).
  2. Run NLI on every sentence (premise = sentence, hypothesis = claim) and keep,
     per abstract, the max entailment and max contradiction probability.
  3. Predict SUPPORT / CONTRADICT only when that probability clears a threshold.
     Thresholds (and how many of the top 3 to consider, and whether to force one
     label per claim) are tuned on validation F1. Predicting nothing means NEI.

NLI models: several zero-shot MNLI models (all <= 400M parameters), plus the best
of them fine-tuned on (gold rationale sentence, claim, label) from training claims
and "neutral" sentences from retrieved non-gold abstracts. The best on validation
F1 writes t5_verification.csv and t5_val_verification.csv.

Every validation claim whose predicted (doc, label) set differs from gold gets one
error category (rules applied in priority order) in t5_val_errors.csv.
"""
import copy
import csv
import itertools
import random
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import (PREDICTIONS_DIR, SEED, load_claims, load_corpus, load_results, relative_path, save_log,
                    set_seed, update_results)
from dense import dense_score_matrix, device
from evaluation import evaluate, rank_run, run_from_scores

TOP_DOCS = 3
NLI_CANDIDATES = [
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "pritamdeka/PubMedBERT-MNLI-MedNLI",
]
MAX_PARAMETERS = 400e6
MAX_PAIR_TOKENS = 256

# Decision rule search space (tuned on validation F1)
THRESHOLDS = [0.01] + [round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05)] + [0.99]
MAX_RANKS = [1, 2, 3]

# NLI fine-tuning
NLI_EPOCHS = 3
NLI_LEARNING_RATE = 1e-5
NLI_BATCH_SIZE = 16
NEUTRAL_DOCS_PER_CLAIM = 3        # retrieved non-gold abstracts per training claim
NEUTRAL_SENTENCES_PER_DOC = 2
GOLD_NON_RATIONALE_PER_DOC = 1    # non-rationale sentences of gold abstracts, also labelled neutral
NEUTRAL_RATIO = 2.0               # at most this many neutral examples per entail/contradict example

LABELS = ["entailment", "contradiction", "neutral"]

# Standalone numbers (1/2000, 10%, 0.5) count; digits inside names (MeCP2, BCL-2, APOE4) do not.
NUMERICAL = re.compile(
    r"(?<![\w-])\d+(?:[.,/]\d+)*(?![\w-])|%|"
    r"\b(more|less|fewer|greater|higher|lower|larger|smaller|better|worse|than|most|least)\b",
    re.IGNORECASE)
NEGATION = re.compile(r"\b(not|no|fails?|failed|without|cannot|never|lack[s]?)\b|n't", re.IGNORECASE)
# Gene / drug / protein-like names: contain a digit, or two or more capitals (IL-10, p53, TNFAIP3, AMPK).
ENTITY = re.compile(r"\b(?=[\w-]*[A-Za-z])(?=[\w-]*(?:\d|[A-Z][\w-]*[A-Z]))[\w-]{2,}\b")


# ---------------------------------------------------------------- retrieval
def best_retriever(val_claims, doc_ids):
    """The dense model with the best validation nDCG@10: the Task 2 general model or a Task 4 fine-tune."""
    results = load_results()
    candidates = [results["task2"]["details"]["selected"]["general"], "finetuned-random",
                  f"finetuned-{results['task4']['details']['selected_hard_run']}"]
    scores = {}
    for name in candidates:
        run = run_from_scores(dense_score_matrix(name, "val"), [c["id"] for c in val_claims], doc_ids)
        scores[name] = evaluate(run, val_claims)["ndcg@10"]
    return max(scores, key=scores.get), scores


def top_documents(model_name, split, claims, doc_ids, depth=TOP_DOCS):
    """{claim_id: [doc_id, ...]} best `depth` docs per claim."""
    run = run_from_scores(dense_score_matrix(model_name, split), [c["id"] for c in claims], doc_ids, depth=depth)
    return {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in rank_run(run, depth).items()}


# ---------------------------------------------------------------- NLI model
class SentenceNLI:
    """A 3-way NLI classifier; `probabilities` returns columns in LABELS order."""

    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # fp32 weights (some checkpoints are saved in fp16, which diverges to NaN when fine-tuned);
        # bf16 autocast provides the speed.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32).to(device())
        id2label = {int(i): name.lower() for i, name in self.model.config.id2label.items()}
        self.label_ids = [next(i for i, name in id2label.items() if name.startswith(label[:6]))
                          for label in LABELS]
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        if self.num_parameters > MAX_PARAMETERS:
            raise ValueError(f"{model_id} has {self.num_parameters / 1e6:.0f}M parameters (> 400M)")

    def encode(self, premises, hypotheses):
        return self.tokenizer(premises, hypotheses, truncation=True, max_length=MAX_PAIR_TOKENS,
                              padding=True, return_tensors="pt").to(device())

    @torch.no_grad()
    def probabilities(self, premises, hypotheses, batch_size=64):
        self.model.eval()
        out = []
        for start in range(0, len(premises), batch_size):
            batch = self.encode(premises[start:start + batch_size], hypotheses[start:start + batch_size])
            with torch.autocast(device(), dtype=torch.bfloat16):
                logits = self.model(**batch).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, self.label_ids].cpu().numpy())
        return np.concatenate(out)


def document_scores(nli, claims, top_docs, corpus_by_id):
    """{claim_id: [(doc_id, rank, max_entailment, max_contradiction), ...]} over each abstract's sentences."""
    premises, hypotheses, owners = [], [], []
    for claim in claims:
        for rank, doc_id in enumerate(top_docs[claim["id"]], start=1):
            for sentence in corpus_by_id[doc_id]["abstract"]:
                premises.append(sentence)
                hypotheses.append(claim["claim"])
                owners.append((claim["id"], doc_id, rank))
    probs = nli.probabilities(premises, hypotheses)

    best = {}
    for (claim_id, doc_id, rank), (entail, contra, _) in zip(owners, probs):
        e, c = best.get((claim_id, doc_id, rank), (0.0, 0.0))
        best[(claim_id, doc_id, rank)] = (max(e, entail), max(c, contra))
    scores = {claim["id"]: [] for claim in claims}
    for (claim_id, doc_id, rank), (e, c) in best.items():
        scores[claim_id].append((doc_id, rank, float(e), float(c)))
    return scores


# ---------------------------------------------------------------- decision rule and F1
def predict(scores, rule):
    """{claim_id: {doc_id: label}} under a decision rule (thresholds, max_rank, one_label_per_claim)."""
    predictions = {}
    for claim_id, docs in scores.items():
        chosen = {}  # doc_id -> (label, probability)
        for doc_id, rank, entail, contra in docs:
            if rank > rule["max_rank"]:
                continue
            options = []
            if entail >= rule["support_threshold"]:
                options.append(("SUPPORT", entail))
            if contra >= rule["contradict_threshold"]:
                options.append(("CONTRADICT", contra))
            if options:
                chosen[doc_id] = max(options, key=lambda option: option[1])
        if rule["one_label_per_claim"] and chosen:
            # Gold labels never mix SUPPORT and CONTRADICT within a claim: keep the most confident label.
            label = max(chosen.values(), key=lambda option: option[1])[0]
            chosen = {d: option for d, option in chosen.items() if option[0] == label}
        predictions[claim_id] = {doc_id: label for doc_id, (label, _) in chosen.items()}
    return predictions


def gold_triples(claims):
    return {(c["id"], ev["doc_id"], ev["label"]) for c in claims for ev in c["evidence"]}


def triples(predictions):
    return {(claim_id, doc_id, label) for claim_id, docs in predictions.items() for doc_id, label in docs.items()}


def precision_recall_f1(predicted, gold):
    hits = len(predicted & gold)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def tune_rule(scores, claims):
    """Best decision rule on these claims by F1 (ties: higher precision, then stricter thresholds)."""
    gold = gold_triples(claims)
    candidates = []
    for support, contradict, max_rank, one_label in itertools.product(THRESHOLDS, THRESHOLDS, MAX_RANKS,
                                                                     [True, False]):
        rule = {"support_threshold": support, "contradict_threshold": contradict,
                "max_rank": max_rank, "one_label_per_claim": one_label}
        candidates.append((rule, precision_recall_f1(triples(predict(scores, rule)), gold)))
    return max(candidates, key=lambda c: (c[1]["f1"], c[1]["precision"],
                                          c[0]["support_threshold"] + c[0]["contradict_threshold"]))


# ---------------------------------------------------------------- NLI fine-tuning
def nli_training_examples(train_claims, corpus_by_id, retrieved):
    """(premise sentence, claim, label) from training claims only."""
    rng = random.Random(SEED)
    labelled, neutral = [], []
    for claim in train_claims:
        gold_docs = set()
        for ev in claim["evidence"]:
            gold_docs.add(ev["doc_id"])
            sentences = corpus_by_id[ev["doc_id"]]["abstract"]
            label = "entailment" if ev["label"] == "SUPPORT" else "contradiction"
            labelled += [(sentences[i], claim["claim"], label) for i in ev["sentences"]]
            others = [s for i, s in enumerate(sentences) if i not in set(ev["sentences"])]
            neutral += [(s, claim["claim"], "neutral")
                        for s in rng.sample(others, min(GOLD_NON_RATIONALE_PER_DOC, len(others)))]
        non_gold = [d for d in retrieved[claim["id"]] if d not in gold_docs][:NEUTRAL_DOCS_PER_CLAIM]
        for doc_id in non_gold:
            sentences = corpus_by_id[doc_id]["abstract"]
            neutral += [(s, claim["claim"], "neutral")
                        for s in rng.sample(sentences, min(NEUTRAL_SENTENCES_PER_DOC, len(sentences)))]
    neutral = rng.sample(neutral, min(len(neutral), int(NEUTRAL_RATIO * len(labelled))))
    examples = labelled + neutral
    rng.shuffle(examples)
    return examples


def fine_tune_nli(model_id, examples, evaluate_epoch):
    """Fine-tune; after each epoch call evaluate_epoch(nli) -> val F1 and keep the best weights."""
    set_seed()
    nli = SentenceNLI(model_id)
    label_to_id = dict(zip(LABELS, nli.label_ids))
    optimizer = torch.optim.AdamW(nli.model.parameters(), lr=NLI_LEARNING_RATE, weight_decay=0.01)
    steps_per_epoch = (len(examples) + NLI_BATCH_SIZE - 1) // NLI_BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * NLI_EPOCHS),
                                                steps_per_epoch * NLI_EPOCHS)
    rng = random.Random(SEED)

    history = [{"epoch": 0, "train_loss": None, **evaluate_epoch(nli)}]
    best_state, best_f1 = copy.deepcopy(nli.model.state_dict()), history[0]["f1"]
    for epoch in range(1, NLI_EPOCHS + 1):
        nli.model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), NLI_BATCH_SIZE):
            batch = [examples[i] for i in order[start:start + NLI_BATCH_SIZE]]
            inputs = nli.encode([b[0] for b in batch], [b[1] for b in batch])
            targets = torch.tensor([label_to_id[b[2]] for b in batch], device=device())
            with torch.autocast(device(), dtype=torch.bfloat16):
                logits = nli.model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite NLI training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nli.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), **evaluate_epoch(nli)})
        print(f"    epoch {epoch}: {history[-1]}")
        if history[-1]["f1"] > best_f1:
            best_state, best_f1 = copy.deepcopy(nli.model.state_dict()), history[-1]["f1"]
    nli.model.load_state_dict(best_state)
    return nli, history


# ---------------------------------------------------------------- error analysis
def error_category(claim, top_docs, predicted, corpus_by_id):
    """One category for a validation claim whose predictions differ from gold (rules in priority order)."""
    gold = {ev["doc_id"]: ev["label"] for ev in claim["evidence"]}
    retrieved_gold = set(gold) & set(top_docs)
    correct = {d for d, label in predicted.items() if gold.get(d) == label}

    if gold and not retrieved_gold:
        return "retrieval_miss"
    if len(gold) > 1 and (len(retrieved_gold) < len(gold) or 0 < len(correct) < len(gold)):
        return "needs_multiple_docs"
    if NUMERICAL.search(claim["claim"]):
        return "numerical"
    rationale = " ".join(corpus_by_id[ev["doc_id"]]["abstract"][i] for ev in claim["evidence"] for i in ev["sentences"])
    flipped = any(d in gold and label != gold[d] for d, label in predicted.items())
    if flipped and (NEGATION.search(claim["claim"]) or NEGATION.search(rationale)):
        return "negation"
    entities = {e.lower() for e in ENTITY.findall(claim["claim"])}
    wrong_docs = [d for d in predicted if d not in gold]
    for doc_id in wrong_docs:
        text = (corpus_by_id[doc_id]["title"] + " " + " ".join(corpus_by_id[doc_id]["abstract"])).lower()
        if any(entity not in text for entity in entities):
            return "entity_mismatch"
    return "other"


# ---------------------------------------------------------------- output
def write_verification(predictions, filename):
    path = PREDICTIONS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "doc_id", "label"])
        for claim_id in sorted(predictions):
            for doc_id, label in sorted(predictions[claim_id].items()):
                writer.writerow([claim_id, doc_id, label])
    return path


def write_errors(categories, filename="t5_val_errors.csv"):
    with open(PREDICTIONS_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "category"])
        for claim_id in sorted(categories):
            writer.writerow([claim_id, categories[claim_id]])


def main():
    set_seed()
    corpus = load_corpus()
    corpus_by_id = {doc["doc_id"]: doc for doc in corpus}
    doc_ids = [doc["doc_id"] for doc in corpus]
    claims = {split: load_claims(split) for split in ["train", "val", "eval"]}
    val_claims = claims["val"]

    retriever, retriever_scores = best_retriever(val_claims, doc_ids)
    top = {split: top_documents(retriever, split, claims[split], doc_ids) for split in ["val", "eval"]}
    gold = gold_triples(val_claims)
    gold_pairs = {(c, d) for c, d, _ in gold}
    retrieved_pairs = {(c, d) for c, docs in top["val"].items() for d in docs}
    print(f"Retriever: {retriever} {retriever_scores}")
    print(f"Gold (claim, doc) pairs in the top {TOP_DOCS}: {len(gold_pairs & retrieved_pairs)}/{len(gold_pairs)} "
          f"(upper bound on recall)")

    # ---- zero-shot NLI models
    results = {}
    for model_id in NLI_CANDIDATES:
        nli = SentenceNLI(model_id)
        scores = document_scores(nli, val_claims, top["val"], corpus_by_id)
        rule, metrics = tune_rule(scores, val_claims)
        results[model_id] = {"mode": "zero-shot", "model_id": model_id, "rule": rule, "val": metrics,
                             "parameters_m": round(nli.num_parameters / 1e6, 1)}
        print(f"  zero-shot {model_id:48s} {metrics}  {rule}")
        del nli
        torch.cuda.empty_cache()

    # ---- fine-tune the best zero-shot model on training claims
    base_id = max(results, key=lambda m: results[m]["val"]["f1"])
    # "Neutral" abstracts for training claims come from the Task 2 general model (it has train embeddings).
    train_retriever = load_results()["task2"]["details"]["selected"]["general"]
    train_retrieved = top_documents(train_retriever, "train", claims["train"], doc_ids, depth=TOP_DOCS + 5)
    examples = nli_training_examples(claims["train"], corpus_by_id, train_retrieved)
    label_counts = {label: sum(e[2] == label for e in examples) for label in LABELS}
    print(f"\nFine-tuning {base_id} on {len(examples)} sentence pairs {label_counts}")

    def evaluate_epoch(nli):
        rule, metrics = tune_rule(document_scores(nli, val_claims, top["val"], corpus_by_id), val_claims)
        return {**metrics, "rule": rule}

    tuned_nli, history = fine_tune_nli(base_id, examples, evaluate_epoch)
    best_epoch = max(history, key=lambda h: h["f1"])
    results["fine-tuned"] = {"mode": "fine-tuned", "model_id": base_id, "rule": best_epoch["rule"],
                             "val": {k: best_epoch[k] for k in ["precision", "recall", "f1"]},
                             "best_epoch": best_epoch["epoch"], "history": history,
                             "parameters_m": round(tuned_nli.num_parameters / 1e6, 1)}

    # ---- final system
    final_key = max(results, key=lambda k: results[k]["val"]["f1"])
    final = results[final_key]
    print(f"\nFinal NLI: {final_key} ({final['mode']}), val {final['val']}, rule {final['rule']}")
    nli = tuned_nli if final_key == "fine-tuned" else SentenceNLI(final["model_id"])
    val_predictions = predict(document_scores(nli, val_claims, top["val"], corpus_by_id), final["rule"])
    eval_predictions = predict(document_scores(nli, claims["eval"], top["eval"], corpus_by_id), final["rule"])
    write_verification(val_predictions, "t5_val_verification.csv")
    write_verification(eval_predictions, "t5_verification.csv")

    # ---- error analysis on validation
    categories = {}
    for claim in val_claims:
        predicted = val_predictions[claim["id"]]
        expected = {ev["doc_id"]: ev["label"] for ev in claim["evidence"]}
        if predicted != expected:
            categories[claim["id"]] = error_category(claim, top["val"][claim["id"]], predicted, corpus_by_id)
    write_errors(categories)
    counts = {c: list(categories.values()).count(c) for c in sorted(set(categories.values()))}
    retrieval_miss_fraction = round(counts.get("retrieval_miss", 0) / max(1, len(categories)), 4)
    print(f"Validation errors: {len(categories)}/{len(val_claims)} claims  {counts}")
    print(f"Q4 retrieval_miss fraction: {retrieval_miss_fraction}")

    error_rows = [{"claim_id": c["id"], "claim": c["claim"], "category": categories[c["id"]],
                   "gold": {ev["doc_id"]: ev["label"] for ev in c["evidence"]},
                   "predicted": val_predictions[c["id"]], "top_docs": top["val"][c["id"]]}
                  for c in val_claims if c["id"] in categories]
    log_path = save_log("task5_verification.json", {"nli_results": results, "errors": error_rows})
    update_results({
        "task5": {
            "retriever_used": f"{retriever} (best validation nDCG@10), top {TOP_DOCS} abstracts",
            "nli_model": final["model_id"],
            "zero_shot_or_finetuned": final["mode"],
            "val_f1": final["val"]["f1"],
            "details": {
                "val_metrics": final["val"],
                "decision_rule": final["rule"],
                "retriever_val_ndcg10": retriever_scores,
                "gold_pairs_in_top3": f"{len(gold_pairs & retrieved_pairs)}/{len(gold_pairs)}",
                "nli_scoring": "per sentence (premise=sentence, hypothesis=claim); per abstract max entailment "
                               "and max contradiction probability",
                "candidates": {k: {kk: v[kk] for kk in ["mode", "model_id", "val", "rule", "parameters_m"]}
                               for k, v in results.items()},
                "nli_fine_tuning": {"examples": len(examples), "label_counts": label_counts,
                                    "epochs": NLI_EPOCHS, "learning_rate": NLI_LEARNING_RATE,
                                    "batch_size": NLI_BATCH_SIZE, "train_retriever": train_retriever,
                                    "history": [{k: v for k, v in h.items() if k != "rule"} for h in history]},
                "val_error_counts": counts,
                "val_errors_total": len(categories),
                "q4_retrieval_miss_fraction": retrieval_miss_fraction,
                "full_log": relative_path(log_path),
            },
        }
    })


if __name__ == "__main__":
    main()
