"""Shared helpers: paths, seeding, data loading and results.json logging.

Every task script imports from here so that all retrievers see the same
document text and every experiment is logged in the same place.
"""
import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
LOGS_DIR = PROJECT_ROOT / "code" / "logs"
RESULTS_PATH = PROJECT_ROOT / "results.json"
RESULTS_TEMPLATE_PATH = PROJECT_ROOT / "results_template.json"

SEED = 42
STUDENT = {"name": "Shibam Sundar Mahakud", "roll_no": "24B2232"}


# ---------------------------------------------------------------- reproducibility
def set_seed(seed=SEED):
    """Seed every RNG we might touch (torch only if it is installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    """Return the corpus as a list of {doc_id, title, abstract} dicts (file order)."""
    return read_jsonl(DATA_DIR / "corpus.jsonl")


def load_claims(split):
    """split is one of 'train', 'val', 'eval'."""
    return read_jsonl(DATA_DIR / f"{split}_claims.jsonl")


def doc_text(doc):
    """The single document representation used by every retriever."""
    return doc["title"] + " " + " ".join(doc["abstract"])


def gold_qrels(claims):
    """{claim_id: {doc_id: 1}} for claims with at least one gold abstract.

    NEI claims (empty evidence) are left out, matching how the graders
    compute retrieval metrics.
    """
    qrels = {}
    for claim in claims:
        doc_ids = {int(ev["doc_id"]) for ev in claim.get("evidence", [])}
        if doc_ids:
            qrels[int(claim["id"])] = {doc_id: 1 for doc_id in doc_ids}
    return qrels


# ---------------------------------------------------------------- results.json
def load_results():
    path = RESULTS_PATH if RESULTS_PATH.exists() else RESULTS_TEMPLATE_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_results(updates):
    """Write top-level sections into results.json (created from the template on first use).

    Each section (e.g. "task3") is replaced as a whole, so re-running a task never
    leaves stale keys from an earlier run behind.
    """
    results = load_results()
    results.update(STUDENT, seed=SEED)
    results.update(updates)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def relative_path(path):
    """A path as recorded in results.json: relative to the project root."""
    return str(path.relative_to(PROJECT_ROOT))


def save_log(name, payload):
    """Write a detailed experiment log (too big for results.json) to code/logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path
