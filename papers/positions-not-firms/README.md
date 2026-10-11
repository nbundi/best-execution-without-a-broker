# Positions, Not Firms — a spin-off paper (seed)

Split out from *The Value of the Marginal Solver* on 2026-09-05. The parent paper was kept
at its measurement scope (curve + two experiments, no theory extension). This folder holds
the **theory-extension arc** that was developed and then removed from the parent, preserved
here as a starting point for a separate paper.

## Thesis

Repeated auctions for customer orders exhibit a pattern neither standard pole of entry theory
produces: **persistently asymmetric in who leads each niche, yet exchangeable at the
replacement margin.** We extend costly-entry auction theory so that heterogeneous *capacity
attaches to positions* (asset niches) that ex-ante symmetric firms *occupy* under entry costs.
Competition **for** the position disciplines outcomes **in** it; firms are replaceable
occupants. The market is therefore **contestable in positions but slow in aggregate depth** —
the first contestability test (Baumol 1982) of a market-making structure, with a portable
prediction for RFQ/dealer markets: where provider positions are contestable, share-based
concentration overstates fragility and the depth of the field is the statistic to watch.

## The five-step arc (paper structure)

1. **Starting point.** Symmetric costly-entry auctions (Levin–Smith 1994; McAfee–McMillan 1987):
   exchangeable bidders, no persistent leadership, exchangeable exits.
2. **The literature shows it fails, from both directions.** Persistent bidder asymmetry is
   documented (Athey–Levin–Seira 2011; Roberts–Sweeting 2013; De Silva–Dunne–Kosmopoulou 2003;
   Jofre-Bonet–Pesendorfer 2003). The standard repair — firm-attached asymmetry (Maskin–Riley
   2000) — predicts a strong provider's exit is costly, and in dealer markets that is confirmed:
   Wahal 1997 (dealer exits permanently widen spreads), Clark-Joseph–Ye–Zi 2017 and
   Venkataraman–Waisburd 2007 (removing DMMs hurts, removing voluntary providers does not),
   Menkveld 2013, Anand–Venkataraman 2016.
3. **The extension.** Capacity distribution `F_i` is a property of niche `i`; symmetric firms pay
   occupancy cost `κ_i` to hold a position (Salop 1979 entry into positions; Edelman–Ostrovsky–
   Schwarz 2007 slot-attached value); occupants compete in the second-score auction; free
   occupancy entry pins `n*_i` per niche. **Proposition (Occupant invariance):** niche outcomes
   depend only on `(F_i, n*_i)`, not on which firm occupies it; exit → reoccupation whenever
   rents cover `κ_i`; occupancy earns no rent beyond entry (Judd 1985 independently predicts this
   for spatial preemption when exit is cheap). **Remark (Two speeds):** reoccupation is fast
   (occupancy margin), aggregate capacity slow (firm formation) — endogenizes the resilience
   parameter Atkeson–Eisfeldt–Weill 2015 leave exogenous. Nearest ancestors where symmetric
   primitives yield persistent heterogeneous *roles*: Farboodi–Jarosch–Shimer 2023, Sambalaibat
   2025 (but there the role is firm-acquired via investment/sorting; here it is the position).
   Firm-attached OTC pole: Hugonnier–Lester–Weill 2020.
4. **Replication of the market's facts** (all computed; see `results.json` and `code.py`):
   - Persistent niche leadership: of 47 solvers, **21 lead ≥1 of 479 assets; the busiest leads
     only 155 (a third); median within-asset leader share 0.28**. → rejects symmetry.
   - Exit-neutrality with **reoccupation: of 32 niches the Dec-2025 exiters had led, 25 were
     reoccupied by a solver NEW to that niche, median lag 0 months.** → rejects firm-attachment;
     confirms occupant invariance.
   - Two speeds: median field 6→3 bidders for a quarter (slow aggregate depth), niches refilled
     within a month (fast reallocation).
   - Reward reform (CIP-74) moved participation, not bids → the policy corollary.
5. **Extrapolation (discussion).** Contestability verdict → general quote-driven markets; the
   Wahal/CJZ/Menkveld exit-cost results hold where positions are NON-contestable (obligations,
   thin/nascent margins, correlated stress). No prior contestability test in market-making
   (only Panzar–Rosse banking, e.g. Claessens–Laeven 2004).

## What's in this folder

- `paper.tex` — the full extension-integrated draft (was the parent at 19pp with the arc folded
  in). **To do:** carve down to a focused positions paper; the measure/curve/experiment
  machinery can be compressed to "established in companion paper [cite parent]".
- `code.py` — includes the `run_paper()` block **B6b (niche leadership + reoccupation)** that the
  parent's code.py does not. Run `python3 code.py` against the parent's `data/` census (symlink
  or copy `../../data`); writes leadership/reoccupation into results.json.
- `results.json` — snapshot with `leadership` and `reoccupation` keys.
- `references.bib` — includes the 20 verified extension refs (all Crossref/JSTOR-verified
  2026-09-05; see parent `reviews/CITATION_AUDIT.md` for the verification log).
- `fig_*.pdf` — figures as of the snapshot.

## Data

Shares the parent's census (`../../data/`, reassembled from `../../dataset/`). No new collection
needed for the leadership/reoccupation exhibits — they derive from the same `competition.jsonl.gz`.

## Open work before this is submittable

- Decide the venue/framing (this is a market-microstructure theory+evidence paper, not a CVA
  fit; targets a finance or market-design audience).
- Carve `paper.tex` down; the positions contribution should lead, with the measure as background.
- Consider a proper occupancy-entry equilibrium (currently the proposition is stated and its
  signature tested; a full model with the `κ_i` free-entry fixed point would strengthen it).
- Robustness on "position": operationalized at counter-asset level (coarse proxy for the true
  technology bundle — a solver's edge may span correlated assets/integrations). A finer niche
  definition (directed pair, or clustered assets) is a natural extension.
- The reoccupation exhibit could be sharpened: condition on niche depth, measure reoccupation
  latency in weeks not months, and test whether entrants vs. incumbents differ by niche liquidity.
