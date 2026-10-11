"""Exploratory streaming pass over the full all-pairs census, hunting for patterns
that extend the paper's narrative (value of competition; concentration;
replaceability). Bid-based only; memory-safe (one pass, compact accumulators).

Reports:
 1. Concentration of competition-value: Pareto of per-order notional-weighted
    runner-up margin (how few orders carry most of the forgone value).
 2. Asset-level monopolization: per counter-asset best-bidder dominance, and the
    share of orders/exposure in effectively monopolized pairs.
 3. Margin vs depth: median runner-up margin by bidder count (the mechanism).
 4. Winner-not-best wedge: magnitude of the fair-auction filtering cost.
"""
import os, sys, math, json
from collections import defaultdict, Counter
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import code as C

RAW = os.path.join(C.DATA, "pairs", "competition_raw.jsonl.gz")
CAP = C._NOTIONAL_CAP_ETH

order_expo = []                      # per order (n>=2): notional-weighted runner-up margin (ETH*frac)
gap_by_n = defaultdict(list)         # n-bucket -> gaps (bps)
asset_bestcount = defaultdict(Counter)   # asset -> Counter(best solver)
asset_expo = defaultdict(float)      # asset -> total exposure (ETH)
asset_orders = defaultdict(int)
wedge_bps = []                       # winner-not-best loss (bps) when positive
n_orders = n_multi = 0

for d in C._iter_jsonl_any(RAW):
    a = C._as_extracted_auction(d)
    if a is None:
        continue
    prices = a.get("prices", {})
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
        n_orders += 1
        ratios = sorted(dd.values(), reverse=True)
        top = max(dd.items(), key=lambda kv: kv[1])[0]
        ne = C.value_in_eth(wo["st"], wo["sa"], prices) or 0.0
        other = wo["other"]
        asset_orders[other] += 1
        asset_bestcount[other][top] += 1
        nb = len(ratios)
        gap_by_n[min(nb, 12)].append(C._gap_bps(ratios[0], ratios[1]) if nb >= 2 else 0.0)
        if nb >= 2:
            n_multi += 1
            g = C._gap_bps(ratios[0], ratios[1])
            ex = (g / 1e4) * ne
            order_expo.append(ex)
            asset_expo[other] += ex
        wdg = C._gap_bps(ratios[0], wr)       # best offer vs winning execution
        if wdg > 1e-9:
            wedge_bps.append(wdg)

print(f"orders: {n_orders:,}  (multi-bid: {n_multi:,})\n")

# --- 1. Pareto of competition-value ----------------------------------------
ex = np.sort(np.asarray(order_expo))[::-1]
tot = ex.sum()
print("=== 1. Concentration of competition-value (notional-weighted runner-up margin) ===")
for p in (0.01, 0.05, 0.10, 0.25):
    k = max(1, int(len(ex) * p))
    print(f"  top {p*100:>4.0f}% of orders carry {100*ex[:k].sum()/tot:5.1f}% of total forgone value")
print(f"  total forgone value (multi-bid orders): {tot:,.0f} ETH-bps-weighted\n")

# --- 2. Asset-level monopolization -----------------------------------------
print("=== 2. Asset-level monopolization (best-bidder dominance per counter-asset) ===")
mono_assets = mono_orders = 0
mono_expo = 0.0
tot_orders = sum(asset_orders.values()); tot_expo = sum(asset_expo.values())
dom_shares = []
for asset, cnt in asset_bestcount.items():
    n = sum(cnt.values())
    if n < 20:                        # ignore tiny assets
        continue
    dom = max(cnt.values()) / n       # top solver's win share for this asset
    dom_shares.append(dom)
    if dom >= 0.90:
        mono_assets += 1
        mono_orders += asset_orders[asset]
        mono_expo += asset_expo[asset]
big = [a for a in asset_bestcount if sum(asset_bestcount[a].values()) >= 20]
print(f"  assets with >=20 orders: {len(big)}")
print(f"  median top-solver win-share across these assets: {np.median(dom_shares):.2f}")
print(f"  'monopolized' assets (one solver best on >=90% of its orders): {mono_assets} "
      f"({100*mono_assets/len(big):.1f}% of assets)")
print(f"    -> they hold {100*mono_orders/tot_orders:.1f}% of orders and "
      f"{100*mono_expo/tot_expo:.1f}% of total forgone value\n")

# --- 3. Margin vs depth (the mechanism) ------------------------------------
print("=== 3. Median runner-up margin by bidder count n ===")
for nb in sorted(gap_by_n):
    g = [x for x in gap_by_n[nb] if x > 0]
    lbl = f"{nb}+" if nb == 12 else str(nb)
    if g:
        print(f"  n={lbl:>3}: median gap {np.median(g):6.2f} bps   (orders {len(gap_by_n[nb]):,})")
print()

# --- 4. Winner-not-best wedge magnitude ------------------------------------
w = np.asarray(wedge_bps)
print("=== 4. Winner-not-best wedge (best offer exceeds winning execution) ===")
print(f"  share of orders winner != best: {len(w)/n_orders:.3f}")
if len(w):
    print(f"  when it happens: median {np.median(w):.2f} bps, mean {w.mean():.2f} bps, "
          f"p90 {np.percentile(w,90):.1f}, p99 {np.percentile(w,99):.1f}")

out = {
    "n_orders": n_orders, "n_multi": n_multi,
    "value_concentration": {f"top_{int(p*100)}pct": round(float(ex[:max(1,int(len(ex)*p))].sum()/tot), 4)
                            for p in (0.01, 0.05, 0.10, 0.25)},
    "monopolization": {"assets_ge20": len(big), "median_top_solver_share": round(float(np.median(dom_shares)), 3),
                       "monopolized_assets": mono_assets, "monopolized_order_share": round(mono_orders/tot_orders, 4),
                       "monopolized_value_share": round(mono_expo/tot_expo, 4)},
    "margin_by_n": {(f"{nb}+" if nb == 12 else str(nb)): round(float(np.median([x for x in gap_by_n[nb] if x > 0])), 2)
                    for nb in sorted(gap_by_n) if [x for x in gap_by_n[nb] if x > 0]},
    "wedge": {"share": round(len(w)/n_orders, 4),
              "median_bps": round(float(np.median(w)), 2) if len(w) else None,
              "p99_bps": round(float(np.percentile(w, 99)), 1) if len(w) else None},
}
json.dump(out, open("results/explore_allpairs.json", "w"), indent=2)
print("\nwrote results/explore_allpairs.json")
