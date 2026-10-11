"""Reproducible exposure and recovery descriptives for the resilience paper."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import code as core
OUT=Path(__file__).parent/'results'/'results.json'
def main():
 rows=core._competition_panel()
 monthly=defaultdict(lambda: [[],[]])
 pivotal=defaultdict(float)
 for r in rows:
  if r['month']:
   monthly[r['month']][0].append(r['n'])
   if r['n']>=2: monthly[r['month']][1].append(core._gap_bps(r['ratios'][0],r['ratios'][1]))
  if r['n']>=2 and r['notional_eth'] and r['notional_eth']<1e5:
   pivotal[r['top_solver']]+=r['notional_eth']*min(core._gap_bps(r['ratios'][0],r['ratios'][1]),2000)/1e4
 series=[{'month':m,'mean_bidders':round(float(np.mean(x[0])),2),'median_gap_bps':round(float(np.median(x[1])),2)} for m,x in sorted(monthly.items()) if x[1]]
 top=sorted(pivotal.items(),key=lambda x:-x[1]); total=sum(pivotal.values())
 out={'n_orders':len(rows),'pivotal_loss_eth':round(total,2),'top1_share':round(top[0][1]/total,4),'top2_share':round(sum(x[1] for x in top[:2])/total,4),'monthly':series,
      'december_to_february_change':{'mean_bidders':round(series[4]['mean_bidders']-series[6]['mean_bidders'],2),'median_gap_bps':round(series[6]['median_gap_bps']-series[4]['median_gap_bps'],2)},
      'interpretation':'Exposure is a fixed-bid, address-level vulnerability measure. The monthly series is descriptive and not an identified exit effect.'}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
