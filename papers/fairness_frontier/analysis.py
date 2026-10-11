"""Reproducible descriptive evidence for the Fairness Frontier paper.

This script deliberately does not claim to reconstruct CoW's allocation
algorithm.  The cached record has solution scores and the protocol filter flag,
but not all constraint/state information required for an exact re-run.  It
therefore reports observable submitted-solution frontiers and labels the
max-score quantities as ceilings, not welfare counterfactuals.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import code as core

OUT = Path(__file__).parent / "results" / "results.json"

def main():
    auctions = core._load_competition(str(ROOT / "data" / "pairs"))
    rows, filtered_scores, fair_scores, actual_scores, wedges = [], [], [], [], []
    for a in auctions:
        sols = a["sols"]
        scores = [(int(s["score"]), s) for s in sols if s.get("score")]
        if not scores:
            continue
        fair = [x for x in scores if not x[1].get("filt")]
        filt = [x for x in scores if x[1].get("filt")]
        wins = [x for x in scores if x[1].get("win")]
        if fair: fair_scores.append(max(x[0] for x in fair))
        if filt: filtered_scores.append(max(x[0] for x in filt))
        if wins: actual_scores.append(sum(x[0] for x in wins))
        rows.append({"solutions": len(scores), "filtered": len(filt), "winners": len(wins)})
        by = defaultdict(list)
        for _, s in scores:
            for o in s["orders"]:
                if o.get("sa") and o.get("ba") and int(o["sa"]) > 0:
                    by[o["id"]].append((int(o["ba"])/int(o["sa"]), bool(s["win"])))
        for offers in by.values():
            best = max(x[0] for x in offers)
            won = [x[0] for x in offers if x[1]]
            if won and best > won[-1]: wedges.append(1e4*(best-won[-1])/best)
    r = np.asarray(rows, dtype=object)
    out = {
      "n_auctions": len(rows),
      "share_auctions_with_filtered_solution": round(float(np.mean([x["filtered"]>0 for x in rows])),4),
      "mean_solutions": round(float(np.mean([x["solutions"] for x in rows])),2),
      "median_filtered_solutions_when_present": round(float(np.median([x["filtered"] for x in rows if x["filtered"]])),2),
      "median_best_filtered_score": int(np.median(filtered_scores)) if filtered_scores else None,
      "median_best_fair_score": int(np.median(fair_scores)) if fair_scores else None,
      "median_realised_sum_winner_score": int(np.median(actual_scores)) if actual_scores else None,
      "n_order_level_selection_wedges": len(wedges),
      "share_order_wedges_positive": round(len(wedges) / 692964, 4),
      "median_wedge_bps_when_positive": round(float(np.median(wedges)),2) if wedges else None,
      "interpretation": "Scores are not summed into a counterfactual allocation. The filtered/fair maxima are submitted-solution diagnostics and only bound an allocation analysis."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
if __name__ == "__main__": main()
