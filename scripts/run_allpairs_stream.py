"""Full-sample AMM-independent results by STREAMING the all-pairs census twice,
holding only compact numeric accumulators (never the 2.6M order rows) -- so it fits
in <0.5 GB and cannot thrash, regardless of other apps' memory use.

Pass 1: per-counter-asset ETH volume + notional distribution -> liquidity/size cuts.
Pass 2: per-order Delta_i(m), gap, exposure, participation -> the paper's bid-based
objects with the SAME formulas as code.py (subset_max_expectation, _gap_bps,
value_in_eth, the auction-clustered bootstrap). AMM-dependent blocks are not here.

Usage: python3 scripts/run_allpairs_stream.py [sample]   # 1 = full, 4 = 25% (for checks)
"""
import os, sys, math, json
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import code as C

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
RAW = os.path.join(C.DATA, "pairs", "competition_raw.jsonl.gz")
BTIME = C._block_time_model()
M, CAP = 5, C._NOTIONAL_CAP_ETH


def iter_orders():
    """Yield (auc, other, ne, ratios_desc, winner_ratio, month) per winning order,
    replicating _build_panel's per-order extraction but WITHOUT storing rows."""
    for d in C._iter_jsonl_any(RAW):
        a = C._as_extracted_auction(d)
        if a is None:
            continue
        aid = a.get("auctionId") or 0
        if SAMPLE > 1 and aid % SAMPLE:
            continue
        prices = a.get("prices", {})
        ts = BTIME(a["startBlock"]) if a.get("startBlock") else None
        month = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m") if ts else None
        by, win = defaultdict(dict), {}
        for s in a["sols"]:
            sv = s["solver"]
            for o in s["orders"]:
                sa, ba = o.get("sa"), o.get("ba")
                if not sa or not ba or not int(sa) or not int(ba):
                    continue
                r = int(ba) / int(sa)
                dd = by[o["id"]]
                if sv not in dd or r > dd[sv]:
                    dd[sv] = r
                if s["win"]:
                    win[o["id"]] = (r, o)
        for oid, dd in by.items():
            if oid not in win:
                continue
            wr, wo = win[oid]
            ratios = sorted(dd.values(), reverse=True)
            top = max(dd.items(), key=lambda kv: kv[1])[0]
            ne = C.value_in_eth(wo["st"], wo["sa"], prices)
            yield aid, wo["other"], ne, ratios, wr, month, top


# ---- Pass 1: liquidity + size cuts -----------------------------------------
print("pass 1: volumes + size distribution ...", flush=True)
vol = defaultdict(float)
nes = []
n_orders = 0
for aid, other, ne, ratios, wr, month, top in iter_orders():
    n_orders += 1
    if ne and 0 < ne <= CAP:
        vol[other] += ne
        nes.append(ne)
logv = {t: math.log10(v) for t, v in vol.items() if v > 0}
qs = np.quantile(list(logv.values()), [0.2, 0.4, 0.6, 0.8])
quint = {t: int(np.searchsorted(qs, lv)) + 1 for t, lv in logv.items()}
ncuts = np.quantile(np.asarray(nes), [0.2, 0.4, 0.6, 0.8])
del nes
print(f"  {n_orders} orders, {len(quint)} counter-assets", flush=True)

# ---- Pass 2: per-order objects into compact accumulators -------------------
print("pass 2: Delta_i(m), field, exposure ...", flush=True)
# curve: S rows [Delta_1..Delta_5] for balanced n>=M, tagged with liq & size group
S_pooled, auc_pooled, liq_tag, sz_tag = [], [], [], []
# field / monthly
liq_n, liq_solo, liq_gap = defaultdict(list), defaultdict(int), defaultdict(list)
mo_n, mo_gap = defaultdict(list), defaultdict(list)
# exposure / wedge
loss, volw = defaultdict(float), defaultdict(float)
wedge_hit = wedge_tot = 0
solvers_ever = set()
for aid, other, ne, ratios, wr, month, top in iter_orders():
    n = len(ratios)
    best = ratios[0]
    solvers_ever.add(top)
    g = C._gap_bps(ratios[0], ratios[1]) if n >= 2 else None
    q = quint.get(other)
    szq = int(np.searchsorted(ncuts, ne)) + 1 if (ne and 0 < ne <= CAP) else None
    # field + monthly
    if q is not None:
        liq_n[q].append(n)
        if n == 1:
            liq_solo[q] += 1
        elif g is not None:
            liq_gap[q].append(g)
    if month:
        mo_n[month].append(n)
        if g is not None:
            mo_gap[month].append(g)
    # exposure (notional-weighted runner-up margin by best-bidder) + wedge
    if n >= 2:
        loss[top] += (g / 1e4) * (ne or 0)
        volw[top] += (ne or 0)
    wedge_tot += 1
    if C._gap_bps(ratios[0], wr) > 1e-9:
        wedge_hit += 1
    # curve (balanced n>=M)
    if n >= M:
        S_pooled.append([C.BPS * (best - C.subset_max_expectation(ratios, m)) / best
                         for m in range(1, M + 1)])
        auc_pooled.append(aid)
        liq_tag.append(q)
        sz_tag.append(szq)

S = np.asarray(S_pooled, float); del S_pooled
aucs = np.asarray(auc_pooled, dtype=np.int64); del auc_pooled
liq_tag = np.asarray([x if x is not None else -1 for x in liq_tag]);
sz_tag = np.asarray([x if x is not None else -1 for x in sz_tag])
print(f"  balanced panel (n>={M}): {len(S)} orders", flush=True)


# ---- curve + bootstrap (replicates _marginal_value_curve on arrays) --------
def curve_boot(Smat, auc_ids, B=400, seed=C.SEED):
    keep = Smat[:, 0] <= np.quantile(Smat[:, 0], 0.99)
    Sm, ac = Smat[keep], auc_ids[keep]
    order = np.argsort(ac, kind="stable")
    Sm, ac = Sm[order], ac[order]
    _, starts, lens = np.unique(ac, return_index=True, return_counts=True)
    starts = starts.astype(np.int64); lens = lens.astype(np.int64); Cn = len(starts)

    def st(x):
        med = np.median(x, axis=0); mean = x.mean(axis=0)
        return med, mean, -np.diff(med), -np.diff(mean)
    med0, mean0, dmed0, dmean0 = st(Sm)
    rng = np.random.default_rng(seed)
    bmed = np.empty((B, M)); bmean = np.empty((B, M))
    bdmed = np.empty((B, M - 1)); bdmean = np.empty((B, M - 1))
    for b in range(B):
        idx = C._ragged_gather(starts, lens, rng.integers(0, Cn, Cn))
        bmed[b], bmean[b], bdmed[b], bdmean[b] = st(Sm[idx])

    def ci(arr, i):
        lo, hi = np.percentile(arr[:, i], [2.5, 97.5])
        return [round(float(lo), 2), round(float(hi), 2)]
    o = {"n_orders": int(len(Sm)), "n_auctions": int(Cn), "B": B, "shortfall": {}, "marginal_value": {}}
    for m in range(1, M + 1):
        o["shortfall"][str(m)] = {"median_bps": round(float(med0[m - 1]), 2), "median_ci": ci(bmed, m - 1),
                                  "mean_bps": round(float(mean0[m - 1]), 2), "mean_ci": ci(bmean, m - 1)}
    for m in range(2, M + 1):
        o["marginal_value"][str(m)] = {"median_bps": round(float(dmed0[m - 2]), 2), "median_ci": ci(bdmed, m - 2),
                                       "mean_bps": round(float(dmean0[m - 2]), 2), "mean_ci": ci(bdmean, m - 2)}
    return o


def group_curve(Smat, tags):
    """Median/mean S(m) per group with per-group top-1% depth-1 trim (= _shortfall_curves)."""
    res = {}
    for gval in sorted(set(int(t) for t in tags if t != -1)):
        sel = Smat[tags == gval]
        keep = sel[:, 0] <= np.quantile(sel[:, 0], 0.99)
        sel = sel[keep]
        res[str(gval)] = {str(m): {"median_bps": round(float(np.median(sel[:, m - 1])), 2),
                                   "mean_bps": round(float(np.mean(sel[:, m - 1])), 2)}
                          for m in range(1, M + 1)}
        res[str(gval)]["n"] = int(len(sel))
    return res


out = {"mode": "all-pairs; AMM-independent; full stream", "sample": f"1/{SAMPLE}",
       "n_orders": n_orders, "n_solvers_ever": len(solvers_ever), "n_months": len(mo_n)}
out["marginal_value_curve"] = curve_boot(S, aucs)
out["shortfall_pooled"] = {k: v for k, v in out["marginal_value_curve"]["shortfall"].items()}
out["shortfall_liquidity_quintiles"] = group_curve(S, liq_tag)
out["shortfall_size_quintiles"] = group_curve(S, sz_tag)

part = {}
for q in (1, 2, 3, 4, 5):
    ns = liq_n.get(q)
    if not ns:
        continue
    part[f"Q{q}"] = {"n_orders": len(ns), "mean_bidders": round(float(np.mean(ns)), 2),
                     "median_bidders": int(np.median(ns)), "share_solo": round(liq_solo[q] / len(ns), 4),
                     "median_gap_bps": round(float(np.median(liq_gap[q])), 2) if liq_gap[q] else None}
out["participation"] = part

tot = sum(loss.values())
sv = sorted(loss.values(), reverse=True)
out["pivotal_top1_share"] = round(sv[0] / tot, 4) if tot and sv else None
out["pivotal_top2_share"] = round(sum(sv[:2]) / tot, 4) if tot else None
out["pivotal_top3_share"] = round(sum(sv[:3]) / tot, 4) if tot else None
out["pivotal_total_loss_eth"] = round(tot, 1)
out["wedge_share_winner_not_best"] = round(wedge_hit / wedge_tot, 4) if wedge_tot else None
out["monthly"] = [{"month": mo, "n_orders": len(mo_n[mo]),
                   "mean_bidders": round(float(np.mean(mo_n[mo])), 2),
                   "median_bidders": int(np.median(mo_n[mo])),
                   "median_gap_bps": round(float(np.median(mo_gap[mo])), 2) if mo_gap[mo] else None}
                  for mo in sorted(mo_n)]

os.makedirs("results", exist_ok=True)
dest = "results/paper_allpairs_noamm.json" if SAMPLE == 1 else f"results/paper_allpairs_s{SAMPLE}.json"
json.dump(out, open(dest, "w"), indent=2, default=str)
print(f"\nwrote {dest}", flush=True)
mv = out["marginal_value_curve"]["shortfall"]
print("Delta_i(m) median bps:", {m: mv[m]["median_bps"] for m in mv})
print("participation:", {q: (v["mean_bidders"], v["share_solo"], v["median_gap_bps"]) for q, v in part.items()})
print("pivotal top1/top2:", out["pivotal_top1_share"], out["pivotal_top2_share"], " wedge:", out["wedge_share_winner_not_best"])
