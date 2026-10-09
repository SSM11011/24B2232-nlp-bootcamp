"""Task 4: fine-tune the best general bi-encoder on the training claims, twice.

Usage:  python code/task4_finetune.py      (run task1 and task2 first)

Training data: one (claim, gold abstract) pair per gold abstract of every
non-NEI training claim. Loss: MultipleNegativesRankingLoss, which also uses
the other abstracts in the batch as negatives. The NO_DUPLICATES sampler keeps
two pairs of the same claim out of one batch, so a claim's second gold
abstract never shows up as its in-batch "negative".

  random run: one random non-gold abstract per pair.
  hard run:   two mined negatives per pair, one from the base dense model
              (sentence-transformers' mine_hard_negatives) and one from the
              tuned BM25. For both, the relative_margin filter drops candidates
              scoring too close to the gold abstract and the top 1 or 3 survivors
              are skipped, since SciFact labels are incomplete and the closest
              "negatives" are often relevant. One hard run per skip value; the
              better on validation becomes t4_hard.trec.

Validation nDCG@10 is computed before training and every few steps, and the
best checkpoint is kept (early stopping by model selection). Loss and
validation curves are plotted to code/logs/task4_curves.png.
"""
import math
import random
import shutil

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import mine_hard_negatives

from common import (LOGS_DIR, PROJECT_ROOT, SEED, doc_text, load_claims, load_corpus, load_results,
                    relative_path, save_log, set_seed, update_results)
from dense import DenseModelSpec, dense_score_matrix, for_inference, load_encoder, save_embeddings
from evaluation import evaluate, run_from_scores, write_trec
from task1_bm25 import load_tuned_bm25
from task2_dense import CANDIDATES

# Training. A first attempt with the MNRL default scale (20) and lr 2e-5 drove the
# training loss to ~0 within an epoch and lowered validation nDCG@10 below the base
# model (code/logs/task4_ablation_scale20_lr2e-5.json). bge/arctic embedders are
# trained with temperature 0.02, so we match it (scale 50) and use a smaller learning rate.
EPOCHS = 3
LEARNING_RATE = 1e-5
LOSS_SCALE = 50.0          # = 1 / temperature 0.02
BATCH_SIZE = 32
WARMUP_RATIO = 0.1
EVAL_EVERY_STEPS = 3       # ~14 steps per epoch, so an early validation peak is not missed

# Hard-negative mining
SKIP_TOP_VALUES = [1, 3]   # then skip this many best surviving non-gold candidates (likely unlabeled
                           # positives); one hard run per value, the better on validation is kept
RANGE_MAX = 50            # only look this deep into each ranking
RELATIVE_MARGIN = 0.05     # drop negatives scoring above  pos - |pos| * 0.05  (sentence-transformers' definition)
REVIEW_SAMPLES = 20

CHECKPOINT_DIR = PROJECT_ROOT / "cache" / "checkpoints"
MODEL_DIR = PROJECT_ROOT / "cache" / "models"


# ---------------------------------------------------------------- training data
def training_pairs(claims, texts_by_doc, query_prefix):
    """One row per (claim, gold abstract); NEI claims have no evidence and are skipped."""
    pairs = []
    for claim in claims:
        for doc_id in sorted({ev["doc_id"] for ev in claim["evidence"]}):
            pairs.append({"claim_id": claim["id"], "anchor": query_prefix + claim["claim"],
                          "positive": texts_by_doc[doc_id], "positive_id": doc_id})
    return pairs


def gold_by_claim(claims):
    return {c["id"]: {ev["doc_id"] for ev in c["evidence"]} for c in claims}


def random_negative_dataset(pairs, doc_ids, gold, texts_by_doc):
    rng = random.Random(SEED)
    rows = {"anchor": [], "positive": [], "negative": []}
    for pair in pairs:
        negative = rng.choice(doc_ids)
        while negative in gold[pair["claim_id"]]:
            negative = rng.choice(doc_ids)
        rows["anchor"].append(pair["anchor"])
        rows["positive"].append(pair["positive"])
        rows["negative"].append(texts_by_doc[negative])
    return Dataset.from_dict(rows)


def mine_dense_negatives(pairs, base_spec, corpus_texts, skip_top):
    """{(anchor, positive): negative text} from the base dense model via mine_hard_negatives."""
    model = for_inference(SentenceTransformer(base_spec.model_id))
    mined = mine_hard_negatives(
        Dataset.from_dict({"anchor": [p["anchor"] for p in pairs], "positive": [p["positive"] for p in pairs]}),
        model,
        corpus=corpus_texts,
        range_min=skip_top,
        range_max=RANGE_MAX,
        relative_margin=RELATIVE_MARGIN,
        num_negatives=1,
        sampling_strategy="top",
        output_format="triplet",
        batch_size=64,
        verbose=True,
    )
    del model
    torch.cuda.empty_cache()
    return {(row["anchor"], row["positive"]): row["negative"] for row in mined}


def mine_bm25_negatives(pairs, train_claims, corpus, base_name, skip_top):
    """{claim_id: [doc_id, ...]} BM25 candidates, filtered exactly like mine_hard_negatives does:

    drop gold abstracts, drop candidates within the relative margin of the claim's
    lowest-scoring gold abstract (base dense model cosine), then skip the top `skip_top`
    survivors.
    """
    doc_ids = np.array([doc["doc_id"] for doc in corpus])
    column = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    bm25, k1, b = load_tuned_bm25(corpus)
    bm25_scores = bm25.score_matrix(train_claims, k1, b)
    dense_scores = dense_score_matrix(base_name, "train")
    gold = gold_by_claim(train_claims)

    candidates, skipped_by_margin = {}, 0
    claim_ids = {p["claim_id"] for p in pairs}
    for row, claim in enumerate(train_claims):
        if claim["id"] not in claim_ids:
            continue
        ranking = [d for d in doc_ids[np.lexsort((doc_ids, -bm25_scores[row]))][:RANGE_MAX]
                   if d not in gold[claim["id"]]]
        positive = min(dense_scores[row, column[d]] for d in gold[claim["id"]])
        threshold = positive - abs(positive) * RELATIVE_MARGIN
        kept = [int(d) for d in ranking if dense_scores[row, column[d]] <= threshold]
        skipped_by_margin += len(ranking) - len(kept)
        candidates[claim["id"]] = kept[skip_top:]
    return candidates, skipped_by_margin


def hard_negative_dataset(pairs, dense_negatives, bm25_candidates, texts_by_doc):
    """(anchor, positive, negative_1 = dense-mined, negative_2 = BM25-mined); pairs with no candidates are dropped."""
    doc_by_text = {text: doc_id for doc_id, text in texts_by_doc.items()}
    rows, review = {"anchor": [], "positive": [], "negative_1": [], "negative_2": []}, []
    for pair in pairs:
        bm25_options = bm25_candidates.get(pair["claim_id"], [])
        dense_text = dense_negatives.get((pair["anchor"], pair["positive"]))
        dense_id = doc_by_text.get(dense_text)
        bm25_options = [d for d in bm25_options if d != dense_id]
        if dense_id is None:  # the dense miner found nothing: fall back to a second BM25 candidate
            dense_id, bm25_options = (bm25_options[0], bm25_options[1:]) if bm25_options else (None, [])
        if dense_id is None or not bm25_options:
            continue
        rows["anchor"].append(pair["anchor"])
        rows["positive"].append(pair["positive"])
        rows["negative_1"].append(texts_by_doc[dense_id])
        rows["negative_2"].append(texts_by_doc[bm25_options[0]])
        review.append({**pair, "dense_negative_id": dense_id, "bm25_negative_id": bm25_options[0]})
    return Dataset.from_dict(rows), review


def write_review_sample(review, corpus_by_id, query_prefix, path):
    """A readable sample of mined rows to check for false negatives before training."""
    sample = random.Random(SEED).sample(review, min(REVIEW_SAMPLES, len(review)))

    def preview(doc_id):
        doc = corpus_by_id[doc_id]
        return f"[{doc_id}] **{doc['title']}** {' '.join(doc['abstract'][:2])[:400]}..."

    lines = ["# Hard-negative review sample\n"]
    for row in sample:
        lines += [f"## Claim {row['claim_id']}: {row['anchor'][len(query_prefix):]}\n",
                  f"- GOLD: {preview(row['positive_id'])}",
                  f"- dense negative: {preview(row['dense_negative_id'])}",
                  f"- BM25 negative: {preview(row['bm25_negative_id'])}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- validation during training
class ValidationNDCG(SentenceEvaluator):
    """Full-corpus validation nDCG@10 (plus Recall@100 and MRR) for the model being trained."""

    def __init__(self, corpus_texts, doc_ids, claims, query_prefix):
        super().__init__()
        self.name = "val"
        self.primary_metric = "ndcg@10"
        self.greater_is_better = True
        self.corpus_texts, self.doc_ids, self.claims = corpus_texts, doc_ids, claims
        self.queries = [query_prefix + c["claim"] for c in claims]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            docs = model.encode(self.corpus_texts, batch_size=64, normalize_embeddings=True,
                                convert_to_tensor=True, show_progress_bar=False)
            queries = model.encode(self.queries, batch_size=64, normalize_embeddings=True,
                                   convert_to_tensor=True, show_progress_bar=False)
        scores = (queries.float() @ docs.float().T).cpu().numpy()
        model.train(was_training)
        run = run_from_scores(scores, [c["id"] for c in self.claims], self.doc_ids)
        metrics = self.prefix_name_to_metrics(evaluate(run, self.claims), self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


def train(run_name, dataset, base_spec, evaluator):
    """Fine-tune and return (path of the best checkpoint, curve history)."""
    set_seed()
    model = SentenceTransformer(base_spec.model_id, device="cuda")
    steps_per_epoch = math.ceil(len(dataset) / BATCH_SIZE)
    output_dir = CHECKPOINT_DIR / run_name
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_RATIO,  # a float < 1 is a ratio of the total steps
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_STEPS,
        save_strategy="steps",
        save_steps=EVAL_EVERY_STEPS,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_ndcg@10",
        greater_is_better=True,
        logging_steps=1,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model, scale=LOSS_SCALE), evaluator=evaluator,
    )
    before = evaluator(model)["val_ndcg@10"]
    trainer.train()

    logs = trainer.state.log_history
    history = {
        "steps_per_epoch": steps_per_epoch,
        "loss": [(e["step"], e["loss"]) for e in logs if "loss" in e],
        "val_ndcg10": [(0, before)] + [(e["step"], e["eval_val_ndcg@10"]) for e in logs if "eval_val_ndcg@10" in e],
    }
    best_step, best_value = max(history["val_ndcg10"][1:], key=lambda point: point[1])
    history.update(best_step=best_step, best_val_ndcg10=best_value,
                   final_val_ndcg10=history["val_ndcg10"][-1][1])

    model_path = MODEL_DIR / run_name
    trainer.model.save_pretrained(str(model_path))  # the best checkpoint (load_best_model_at_end)
    del trainer, model
    torch.cuda.empty_cache()
    shutil.rmtree(output_dir, ignore_errors=True)
    return model_path, history


# ---------------------------------------------------------------- plotting
def plot_curves(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834"]  # categorical slots 1-2, validated for CVD
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, (loss_ax, val_ax) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)

    base_value = next(iter(histories.values()))["val_ndcg10"][0][1]
    val_ax.axhline(base_value, color=muted, linewidth=1, linestyle="--")
    last_step = max(step for h in histories.values() for step, _ in h["val_ndcg10"])
    val_ax.annotate(f"base model {base_value:.3f}", (last_step, base_value), textcoords="offset points",
                    xytext=(0, 5), ha="right", fontsize=9, color=muted)

    for color, (label, history) in zip(colors, histories.items()):
        steps, losses = zip(*history["loss"])
        loss_ax.plot(steps, losses, color=color, linewidth=2, label=label)
        steps, values = zip(*history["val_ndcg10"])
        val_ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=6, label=label)
        val_ax.plot(history["best_step"], history["best_val_ndcg10"], marker="*", markersize=16,
                    color=color, markeredgecolor=surface, markeredgewidth=1.5, zorder=5)
        near_right_edge = history["best_step"] > 0.6 * last_step
        val_ax.annotate(f"kept: {history['best_val_ndcg10']:.3f} @ step {history['best_step']}",
                        (history["best_step"], history["best_val_ndcg10"]), textcoords="offset points",
                        xytext=(-10, 10) if near_right_edge else (8, -16),
                        ha="right" if near_right_edge else "left", fontsize=9, color=ink,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": surface, "edgecolor": "none"})
    low, high = val_ax.get_ylim()
    val_ax.set_ylim(low, high + 0.15 * (high - low))  # headroom so labels stay inside the axes

    loss_ax.set_title("Training loss (MNRL)", color=ink, loc="left")
    val_ax.set_title("Validation nDCG@10 (★ = kept checkpoint)", color=ink, loc="left")
    for ax in (loss_ax, val_ax):
        ax.set_facecolor(surface)
        ax.set_xlabel("training step", color=muted)
        ax.grid(color=grid, linewidth=0.8)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    set_seed()
    corpus = load_corpus()
    claims = {split: load_claims(split) for split in ["train", "val", "eval"]}
    base_name = load_results()["task2"]["details"]["selected"]["general"]
    base_spec = next(spec for spec in CANDIDATES if spec.name == base_name)
    prefix = base_spec.query_prefix

    # Documents exactly as the base model sees them (shared doc_text plus its document prefix, if any).
    corpus_texts = [base_spec.doc_prefix + doc_text(doc) for doc in corpus]
    doc_ids = [doc["doc_id"] for doc in corpus]
    texts_by_doc = dict(zip(doc_ids, corpus_texts))
    corpus_by_id = {doc["doc_id"]: doc for doc in corpus}

    pairs = training_pairs(claims["train"], texts_by_doc, prefix)
    gold = gold_by_claim(claims["train"])
    print(f"Base model {base_spec.model_id}; {len(pairs)} training pairs from "
          f"{len({p['claim_id'] for p in pairs})} non-NEI training claims")

    datasets = {"random": random_negative_dataset(pairs, doc_ids, gold, texts_by_doc)}
    mining_stats = {}
    for skip_top in SKIP_TOP_VALUES:
        run_name = f"hard-skip{skip_top}"
        dense_negatives = mine_dense_negatives(pairs, base_spec, corpus_texts, skip_top)
        bm25_candidates, bm25_skipped = mine_bm25_negatives(pairs, claims["train"], corpus, base_name, skip_top)
        datasets[run_name], review = hard_negative_dataset(pairs, dense_negatives, bm25_candidates, texts_by_doc)
        review_path = LOGS_DIR / f"task4_hard_negative_review_skip{skip_top}.md"
        write_review_sample(review, corpus_by_id, prefix, review_path)
        mining_stats[run_name] = {"pairs_with_dense_negative": len(dense_negatives),
                                  "bm25_candidates_dropped_by_margin": bm25_skipped,
                                  "rows": len(datasets[run_name]),
                                  "review_sample": relative_path(review_path)}
        print(f"{run_name}: {mining_stats[run_name]}")

    evaluator = ValidationNDCG(corpus_texts, doc_ids, claims["val"], prefix)
    histories, val_metrics, eval_runs = {}, {}, {}
    for run_name, dataset in datasets.items():
        print(f"\n=== {run_name} negatives: {len(dataset)} rows ===")
        model_path, histories[run_name] = train(run_name, dataset, base_spec, evaluator)
        h = histories[run_name]
        print(f"  val nDCG@10: base {h['val_ndcg10'][0][1]:.4f} -> best {h['best_val_ndcg10']:.4f} "
              f"(step {h['best_step']}) -> final {h['final_val_ndcg10']:.4f}")

        # Re-encode with the kept checkpoint, exactly like the Task 2 models (fp16, cached embeddings).
        spec = DenseModelSpec(f"finetuned-{run_name}", "finetuned", str(model_path),
                              query_prefix=prefix, doc_prefix=base_spec.doc_prefix)
        encoder = load_encoder(spec)
        save_embeddings(spec.name, "corpus", encoder.encode_docs(corpus))
        for split in ["val", "eval"]:
            save_embeddings(spec.name, split, encoder.encode_queries([c["claim"] for c in claims[split]]))
        del encoder
        torch.cuda.empty_cache()

        def run_for(split):
            return run_from_scores(dense_score_matrix(spec.name, split), [c["id"] for c in claims[split]], doc_ids)

        val_metrics[run_name] = evaluate(run_for("val"), claims["val"])
        eval_runs[run_name] = run_for("eval")
        print(f"  kept checkpoint, val {val_metrics[run_name]}")

    hard_run = max((name for name in datasets if name.startswith("hard")),
                   key=lambda name: val_metrics[name]["ndcg@10"])
    print(f"\nSelected hard-negative run: {hard_run}")
    write_trec(eval_runs["random"], "t4_random.trec", "finetuned_random", claims["eval"])
    write_trec(eval_runs[hard_run], "t4_hard.trec", "finetuned_hard", claims["eval"])

    plot_path = LOGS_DIR / "task4_curves.png"
    skip = hard_run.removeprefix("hard-skip")
    plot_curves({"random negatives": histories["random"], f"hard negatives (skip top {skip})": histories[hard_run]},
                plot_path)
    log_path = save_log("task4_finetune.json", {"histories": histories, "val_metrics": val_metrics,
                                                "selected_hard_run": hard_run})
    base_val = histories["random"]["val_ndcg10"][0][1]
    update_results({
        "task4": {
            "base_model": base_spec.model_id,
            "val_ndcg10_random_negatives": val_metrics["random"]["ndcg@10"],
            "val_ndcg10_hard_negatives": val_metrics[hard_run]["ndcg@10"],
            "details": {
                "base_model_val_ndcg10": base_val,
                "base_model_parameters_m": load_results()["task2"]["details"]["parameters_m"]["general"],
                "selected_hard_run": hard_run,
                "training_pairs": len(pairs),
                "rows": {name: len(ds) for name, ds in datasets.items()},
                "training": {"loss": "MultipleNegativesRankingLoss (in-batch negatives)",
                             "batch_sampler": "NO_DUPLICATES", "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
                             "loss_scale": LOSS_SCALE, "batch_size": BATCH_SIZE, "warmup_ratio": WARMUP_RATIO,
                             "precision": "bf16", "eval_every_steps": EVAL_EVERY_STEPS,
                             "steps_per_epoch": histories["random"]["steps_per_epoch"],
                             "checkpoint_selection": "best validation nDCG@10 (load_best_model_at_end)"},
                "hard_negative_mining": {
                    "negatives_per_pair": "2: one from the base dense model, one from tuned BM25",
                    "dense": "sentence_transformers.util.mine_hard_negatives",
                    "order": "drop gold, drop within relative margin, then skip the top `skip_top` survivors",
                    "skip_top_values": SKIP_TOP_VALUES, "range_max": RANGE_MAX,
                    "relative_margin": RELATIVE_MARGIN,
                    "per_run": mining_stats,
                },
                "val_metrics": val_metrics,
                "ablation_first_attempt": "code/logs/task4_ablation_scale20_lr2e-5.json "
                                          "(bge-large, MNRL scale 20, lr 2e-5, 4 epochs)",
                "curves": {name: {k: h[k] for k in ["best_step", "best_val_ndcg10", "final_val_ndcg10"]}
                           for name, h in histories.items()},
                "curve_plot": relative_path(plot_path),
                "full_log": relative_path(log_path),
            },
        }
    })


if __name__ == "__main__":
    main()
