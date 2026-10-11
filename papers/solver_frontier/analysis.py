"""Decompose AMM-relative execution into solver frontier and selection loss.

The AMM quote is reconstructed with the same conventions as the root analysis.
For each order, all proposals are evaluated at the observed price vector; this
is a fixed-bid decomposition, not a causal decomposition of solver entry.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import code as core
OUT=Path(__file__).parent/'results'/'results.json'
def main():
 auc=core._load_competition(str(ROOT/'data'/'pairs')); meta={m['uid']:m for m in core._load_meta(str(ROOT/'data'/'pairs'))}; cache=core._load_route_cache(); prices={}
 for a in auc: prices.update(a['prices'])
 rows=[]
 for a in auc:
  by={}; win={}
  for s in a['sols']:
   for o in s['orders']:
    if not o.get('sa') or not o.get('ba') or not int(o['sa']): continue
    ratio=int(o['ba'])/int(o['sa']); by.setdefault(o['id'],[]).append(ratio)
    if s['win']: win[o['id']]=o
  for oid,o in win.items():
   q=cache.get(oid)
   if not q or q.get('error'): continue
   buy=(meta.get(oid,{}) or {}).get('kind','sell')=='buy'; N=core.value_in_eth(o['st'],int(o['sa']),prices)
   if not N: continue
   if buy:
    base=core.value_in_eth(o['st'],int(q['quote']),prices)
    if not base: continue
    # Higher is better convention: cheaper realised input relative to AMM.
    realised=0.; frontier=0. # Exact buy-order welfare needs consistent input amounts; omit rather than mix conventions.
    continue
   base=core.value_in_eth(o['bt'],int(q['quote']),prices); actual=core.value_in_eth(o['bt'],int(o['ba']),prices)
   if not base or not actual: continue
   best=max(by[oid]); bestout=best*int(o['sa']); bestval=core.value_in_eth(o['bt'],int(bestout),prices)
   if not bestval: continue
   pi=1e4*(actual-base)/N; frontier=1e4*(bestval-base)/N; selection=1e4*(bestval-actual)/N
   if abs(pi)<=1000: rows.append({'tier':o['tier'],'pi':pi,'frontier':frontier,'selection':selection,'n':len(set(by[oid]))})
 d=pd.DataFrame(rows); out={'n_sell_orders_reliable':len(d),'note':'Sell orders only: exact-output buy orders are excluded rather than imposing an invalid ratio transformation. AMM prices use cached SOR quotes and native auction prices.', 'tiers':{}}
 for t,g in d.groupby('tier'):
  out['tiers'][t]={'n':len(g),'median_realised_pi_bps':round(float(g.pi.median()),2),'median_solver_frontier_bps':round(float(g.frontier.median()),2),'median_selection_loss_bps':round(float(g.selection.median()),2),'median_bidders':round(float(g.n.median()),1)}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
