"""Q1: per-claim nDCG@10 of val_bm25.trec vs. val_general.trec.

Usage:  python code/report_q1.py

Prints the claims with the largest gap in each direction (claim text, gold
abstracts and their rank in both runs). Two wins per side go into
results.json -> report_q1: the hand-picked CHOSEN claims if they are still
real wins, otherwise the largest-gap ones.
"""
from common import load_claims, load_corpus, save_log, update_results
from evaluation import per_claim_ndcg10, rank_run, read_trec

SHOW = 6
# Picked by hand from the printed lists as the clearest illustrations:
#   1390, 673: rare exact terms ("mcm5s2U", "NNR codons"; "LDL-cholesterol") that BM25 matches directly.
#   463, 1311: paraphrases ("genomic aberrations" ~ "mutational landscapes"; "lncRNAs" ~ "LincRNA-p21").
CHOSEN = {"bm25_wins": [1390, 673], "general_wins": [463, 1311]}


def gold_ranks(run, claim):
    ranking = [doc_id for doc_id, _ in rank_run({claim["id"]: run[claim["id"]]})[claim["id"]]]
    return {ev["doc_id"]: ranking.index(ev["doc_id"]) + 1 if ev["doc_id"] in ranking else None
            for ev in claim["evidence"]}


def main():
    val_claims = load_claims("val")
    claims_by_id = {c["id"]: c for c in val_claims}
    titles = {doc["doc_id"]: doc["title"] for doc in load_corpus()}
    runs = {"bm25": read_trec("val_bm25.trec"), "general": read_trec("val_general.trec")}
    ndcg = {name: per_claim_ndcg10(run, val_claims) for name, run in runs.items()}

    rows = []
    for claim_id in ndcg["bm25"]:
        claim = claims_by_id[claim_id]
        rows.append({
            "claim_id": claim_id,
            "claim": claim["claim"],
            "ndcg10_bm25": ndcg["bm25"][claim_id],
            "ndcg10_general": ndcg["general"][claim_id],
            "delta": round(ndcg["bm25"][claim_id] - ndcg["general"][claim_id], 4),
            "gold_rank_bm25": gold_ranks(runs["bm25"], claim),
            "gold_rank_general": gold_ranks(runs["general"], claim),
            "gold_titles": {ev["doc_id"]: titles[ev["doc_id"]] for ev in claim["evidence"]},
        })
    bm25_wins = sorted((r for r in rows if r["delta"] > 0), key=lambda r: -r["delta"])
    general_wins = sorted((r for r in rows if r["delta"] < 0), key=lambda r: r["delta"])

    print(f"BM25 better on {len(bm25_wins)} claims, general better on {len(general_wins)}, "
          f"tied on {len(rows) - len(bm25_wins) - len(general_wins)}")
    for title, wins in [("BM25 wins", bm25_wins), ("General (dense) wins", general_wins)]:
        print(f"\n{title}")
        for r in wins[:SHOW]:
            print(f"  [{r['claim_id']}] {r['claim']}")
            print(f"      nDCG@10 bm25 {r['ndcg10_bm25']:.3f} vs general {r['ndcg10_general']:.3f}; "
                  f"gold rank bm25 {r['gold_rank_bm25']} vs general {r['gold_rank_general']}")
            for doc_id, doc_title in r["gold_titles"].items():
                print(f"      gold {doc_id}: {doc_title}")

    picks = {"bm25_wins": pick_two(CHOSEN["bm25_wins"], bm25_wins),
             "general_wins": pick_two(CHOSEN["general_wins"], general_wins)}
    save_log("report_q1_per_claim.json", sorted(rows, key=lambda r: -r["delta"]))
    update_results({"report_q1": picks})
    print(f"\nreport_q1 -> {picks}")


def pick_two(preferred, wins):
    """The hand-picked claims that are still real wins, topped up with the largest-gap wins."""
    win_ids = [r["claim_id"] for r in wins]
    picks = [claim_id for claim_id in preferred if claim_id in win_ids]
    if len(picks) < len(preferred):
        print(f"  note: {sorted(set(preferred) - set(picks))} are no longer wins; using largest-gap claims instead")
    return (picks + [claim_id for claim_id in win_ids if claim_id not in picks])[:2]


if __name__ == "__main__":
    main()
