"""AMM-independent results on the all-pairs census, within 8 GB RAM.

Builds the panel (streaming, no metadata load), assigns the liquidity/size strata,
and computes the paper's core BID-BASED objects: the competitive field, the
marginal-value curve Delta_i(m) + liquidity/size gradient (with auction-clustered
bootstrap CIs), solver exposure, the winner-not-best wedge, and monthly
participation. Writes results/paper_allpairs_noamm.json.

AMM-dependent blocks (exit study, cap-exposure realized-PI, batch premium) are
skipped -- they need the full routed counterfactual, which finishes on the big box.

Usage: python3 scripts/run_allpairs_noamm.py [sample]   # 1 = full, 4 = 25% (lower RAM)
"""
import os, sys, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, before stdlib `code`
from collections import defaultdict
import numpy as np
import code as C

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

rows, months, _m, _bf, _addr = C._build_panel(sample=SAMPLE, load_meta=False)
for r in rows:                                    # free AMM-only per-order fields (lower RAM)
    r.pop("_o", None); r.pop("_prices", None); r.pop("_blk", None)
print(f"panel: {len(rows)} orders, {len(months)} months (sample=1/{SAMPLE})", flush=True)

# --- liquidity + size strata (same construction as run_paper) ---------------
vol = defaultdict(float)
for r in rows:
    ne = r["notional_eth"]
    if ne and 0 < ne <= C._NOTIONAL_CAP_ETH:
        vol[r["other"]] += ne
logv = {t: math.log10(v) for t, v in vol.items() if v > 0}
qs = np.quantile(list(logv.values()), [0.2, 0.4, 0.6, 0.8])
quint = {t: int(np.searchsorted(qs, lv)) + 1 for t, lv in logv.items()}
ncuts = np.quantile([r["notional_eth"] for r in rows
                     if r["notional_eth"] and 0 < r["notional_eth"] <= C._NOTIONAL_CAP_ETH],
                    [0.2, 0.4, 0.6, 0.8])
for r in rows:
    r["liq"] = quint.get(r["other"])
    ne = r["notional_eth"]
    r["szq"] = (int(np.searchsorted(ncuts, ne)) + 1
                if ne and 0 < ne <= C._NOTIONAL_CAP_ETH else None)

out = {"mode": "all-pairs; AMM-independent results only", "sample": f"1/{SAMPLE}",
       "n_orders": len(rows), "n_months": len(months),
       "n_solvers_ever": len({r["offers"][0][0] for r in rows}
                             | {r["offers"][1][0] for r in rows if r["n"] > 1})}

# --- competitive field + monthly, via compact accumulators (no pandas) ------
liq_n, liq_solo, liq_gap = defaultdict(list), defaultdict(int), defaultdict(list)
mo_n, mo_gap = defaultdict(list), defaultdict(list)
for r in rows:
    n = r["n"]
    g = C._gap_bps(r["offers"][0][1], r["offers"][1][1]) if n >= 2 else None
    q = r["liq"]
    if q is not None:
        liq_n[q].append(n)
        if n == 1:
            liq_solo[q] += 1
        elif g is not None:
            liq_gap[q].append(g)
    mo = r["month"]
    if mo:
        mo_n[mo].append(n)
        if g is not None:
            mo_gap[mo].append(g)
part = {}
for q in (1, 2, 3, 4, 5):
    ns = liq_n.get(q)
    if not ns:
        continue
    part[f"Q{q}"] = {"n_orders": len(ns),
                     "mean_bidders": round(float(np.mean(ns)), 2),
                     "median_bidders": int(np.median(ns)),
                     "share_solo": round(liq_solo[q] / len(ns), 4),
                     "median_gap_bps": round(float(np.median(liq_gap[q])), 2) if liq_gap[q] else None}
out["participation"] = part

# --- marginal-value curve + gradients (Sec 5.2) -----------------------------
out["shortfall_pooled"] = C._shortfall_curves(rows, lambda r: "pooled")["pooled"]
out["marginal_value_curve"] = C._marginal_value_curve(rows)
liq = C._shortfall_curves(rows, lambda r: r["liq"])
out["shortfall_liquidity_quintiles"] = {str(k): v for k, v in sorted(liq.items(), key=lambda kv: (kv[0] is None, kv[0])) if k is not None}
szq = C._shortfall_curves(rows, lambda r: r["szq"])
out["shortfall_size_quintiles"] = {str(k): v for k, v in sorted(szq.items(), key=lambda kv: (kv[0] is None, kv[0])) if k is not None}

# --- solver exposure concentration + wedge (Sec 5.3) ------------------------
loss, volw = defaultdict(float), defaultdict(float)
for r in rows:
    if r["n"] < 2:
        continue
    gap = C._gap_bps(r["offers"][0][1], r["offers"][1][1])
    ne = r["notional_eth"] or 0
    top = r["offers"][0][0]
    loss[top] += (gap / 1e4) * ne
    volw[top] += ne
tot = sum(loss.values())
solvers = sorted(loss.items(), key=lambda kv: -kv[1])
out["pivotal_top1_share"] = round(solvers[0][1] / tot, 4) if tot and solvers else None
out["pivotal_top2_share"] = round(sum(v for _, v in solvers[:2]) / tot, 4) if tot else None
out["pivotal_top3_share"] = round(sum(v for _, v in solvers[:3]) / tot, 4) if tot else None
out["pivotal_total_loss_eth"] = round(tot, 1)
w = np.array([C._gap_bps(r["offers"][0][1], r["winner_ratio"]) for r in rows])
out["wedge_share_winner_not_best"] = round(float((w > 1e-9).mean()), 4)

# --- monthly participation (Sec 5.5 bid-based part) -------------------------
monthly = []
for mo in sorted(mo_n):
    ns = mo_n[mo]
    monthly.append({"month": mo, "n_orders": len(ns),
                    "mean_bidders": round(float(np.mean(ns)), 2),
                    "median_bidders": int(np.median(ns)),
                    "median_gap_bps": round(float(np.median(mo_gap[mo])), 2) if mo_gap[mo] else None})
out["monthly"] = monthly

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/paper_allpairs_noamm.json", "w"), indent=2, default=str)
print("\nwrote results/paper_allpairs_noamm.json", flush=True)
mv = out["marginal_value_curve"]["shortfall"]
print("Delta_i(m) median bps:", {m: mv[m]["median_bps"] for m in mv})
print("participation (mean_bidders, solo, med_gap):",
      {q: (v["mean_bidders"], v["share_solo"], v["median_gap_bps"]) for q, v in part.items()})
print("pivotal top2 share:", out["pivotal_top2_share"], " wedge:", out["wedge_share_winner_not_best"])
