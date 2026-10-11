# The Value of the Marginal Solver

Working paper / conference submission: *The Value of the Marginal Solver: Competition and Best Execution in Intent-Based Markets*.

(Repositioned 2026-08-31 from the earlier draft *Best Execution Without a Broker*; the multi-factor broker comparison, fill cohort, and AMM counterfactual were cut in favour of the solver-competition analysis. The collectors and dataset are unchanged.)

This README is the single source of truth for all project and submission-venue details. Tooling and Claude instructions defer to it.

## One-line thesis

"What is one more competitor for a customer order worth?" — never answerable against a computed counterfactual because losing bids are unobserved everywhere; CoW publishes them (first production market to do so, to our knowledge). Fixed-bid thinning of the recorded field: single retained bidder costs the median order ~14 bps, second restores ~8, fifth <1; monotone along a sample-computed liquidity metric (96 bps in the thinnest asset quintile to 14 in the deepest — the metric is the paper's ONLY stratification; no hand-assigned tiers anywhere); exposure concentrated (top-2 solvers 62%). The Dec-2025 exit wave then *tested* the measure: aggregate field thinned and margins widened market-wide, but the most-exposed assets fared **no worse** than barely-exposed ones (differences tightly bounded around zero on bidders, realized PI vs routed-AMM benchmark, and gaps) — substitution absorbed the exits. Punchline: the fixed-bid measure is an upper bound on realized dependence; what binds is the depth of the field, not its membership. CIP-74: Prop 3 (participation channel) passes — membership churned, participation rose, predicted quality transmission <1 bp (inside Zhang's 7 bp bound), and the pre-specified cap-exposure test finds no bidding-margin response; conditional-on-n invariance is stated as an untestable remark. Post-review recalibration (2026-08-31): claims laddered to-my-knowledge/in-spirit/candidate-reconciliation; truthfulness is a *maintained idealization*, not a finding.

## Build

```sh
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex   # -> paper.pdf
python3 code.py               # full paper pipeline, paper-ordered blocks B1-B7
                              # -> results/paper.json + figures/fig_shortfall|liquidity|
                              #    pivotal|cip74|exitstudy.pdf (fig_anatomy is a one-off)
```

(Data collection stays behind `--collect-pairs/--collect-amm/--collect-basefees/--collect-arrivals/--collect-fill-census`. The former `--competition/--exitstudy/--refinements` flags and all legacy first-paper analyses were consolidated into the single `run_paper()` pipeline on 2026-09-01; results/paper.json is the sole canonical results file.)

Prop-3 reformulation + cap-exposure test (2026-09-01, later): Prop 3 restated as the **participation channel** (rewards move quality only via participation, priced as delta(n)*dn by the curve) with conditional invariance demoted to an untestable remark. New pre-specified cap-exposure test: 4.5% of pre-reform orders sat in cap-binding auctions (fixed 0.012 ETH cap); classifying assets by that exposure, bidding margins show NO response to CIP-74 (-1.0 bps, se 3.9; robust to threshold halving/doubling) while participation rose ~+1 bidder in previously cap-bound assets (imperfect pre-trend; suggestive). PI coefficient uninterpretable (classification-window selection). Prop 3-prime passes both its halves.

Iteration-2 findings (2026-09-01): medians are now the primary statistic; the continuous liquidity axis recovers the clean monotone ordering the coarse tiers blurred (single-bidder shortfall 96→14 bps across quintiles); solo orders are dust (44%→13% by size decile; 53% for small short-tail); detected CoW matches earn +4.5 bps on 1.2% of orders (aggregate < wedge, but a strict lower bound); the CIP-74 conditional-on-n tests fail in *every* cut (incl. survivor-only) because n is an equilibrium object — the paper now reports the participation response as the only robust reform fact, with December's exits framed (hedged) as plausibly part of the repricing. Propositions are anchored to Stigler (1961) / Burdett–Judd (1983): the shortfall curve is the value-of-search schedule drawn from observed quotations.

(`latexmk -pdf paper.tex` also works where latexmk is installed. figures/fig_anatomy.pdf is produced by a small one-off script over the census.)

TeX Live 2026 (`pdflatex` + `natbib` + `plainnat`). Python 3.9+, deps: `numpy`, `scipy`, `pandas`, `statsmodels` (`matplotlib` for figures). The script is deterministic (seeded).

### Data collection (§3) — done; commands for reproduction

```sh
python3 code.py --collect-pairs          # settled-auction census (competition records) -> data/pairs/*.jsonl.gz
python3 code.py --collect-amm            # route every order through the archival Uniswap SOR -> data/amm_routes.jsonl
python3 code.py --collect-basefees       # per-block base fees -> data/pairs/base_fees.jsonl.gz
python3 code.py --collect-arrivals       # risk-set scan -> arrivals.jsonl.gz (submission cohort, phase 1)
python3 code.py --collect-fill-census    # classify arrivals /USDC -> data/fill_cohort/cohort.jsonl.gz (phase 2)
```

Every collector is resumable (checkpoints), rate-limit hardened (backoff; transient failures retry rather than bank), and compact (gzipped JSON-lines). The AMM leg needs a Node smart-order-router bridge (`router/`) and an archival RPC in `.env` (`ETH_RPC_URL`).

### Dataset

The census the paper is computed from is committed under `dataset/` as a split `tar.gz` archive (parts `< 100 MB`, so it fits in normal git without LFS). Reassemble it into `data/` with `sh dataset/unpack.sh` (or `cat dataset/census-data.tar.gz.part* | tar xzf -`), then run the analysis line above. Contents:

| File | Scale | What it is |
|---|---|---|
| `data/pairs/competition.jsonl.gz` | 565,259 auctions | Solver-competition census — every settled auction's winning **and** losing solutions for the /USDC tier orders, with native prices. The observable-counterfactual record (realised competitive margin, size/tier gradients, time series, CIP-74 study). |
| `data/pairs/cohort_meta.jsonl.gz` | ~665,000 orders | Order lifecycle metadata (status, kind, class, partially-fillable flag, executed amounts) — order-type composition and fill decomposition. |
| `data/amm_routes.jsonl` | 509,309 routed | Uniswap smart-order-router counterfactual quotes, each pinned to its settlement block (Bachu et al. method) — price improvement vs a direct AMM route. |
| `data/pairs/base_fees.jsonl.gz` | 547,569 blocks | EIP-1559 base fee per settlement block — for gas-net price improvement. |
| `data/fill_cohort/cohort.jsonl.gz` | 1,853,859 orders | Full submission cohort — /USDC arrivals classified by marketability and fill status. Marketability rate and fill probability. |
| `data/pairs/manifest.json` | — | Census provenance (span, counts). |

## Target venue & submission requirements

**Primary venue.** Crypto Valley Association — Call for Papers 2026 (CFP26), run by the CVA Education Working Group. Outputs feed a journal publication and the **Academic Forum on 2026-11-19**.

**Theme.** *On-Chain Finance and the Reinvention of Financial Infrastructure.* Stated focus areas: tokenised assets and on-chain settlement; digital identity; compliant DeFi; convergence of banking and Web3; institutional adoption of blockchain infrastructure.

**Declared sub-themes (from the draft front matter — numbering to be verified against the submission form):**
- 3 — Institutional DeFi
- 7 — Payments and Settlement on Blockchain Rails
- 10 — Business Models for Banks in Web3

**Deadlines** (confirmed against the CFP26 announcement, https://members.cryptovalley.swiss/news/13219587, on 2026-08-14):
- Abstract: **2026-09-05** (extended from the previously listed 2026-08-15)
- Full research paper: **2026-10-15**
- Journal publication & Academic Forum: **2026-11-19**

**Submission channel.** Abstract via Typeform: https://cryptovalley.typeform.com/to/EN3JkIPG. Questions to tilmar.goos@cryptovalley.swiss.

**Format — CONFIRMED.** The official layout is the "CVA CFP 2026 — A Fill-in-the-Blanks Document" (Google Drive: https://drive.google.com/file/d/1NJ3Sb2CB-ln9oPNydU4OsVM7CrzczQoK/view; local copy in `literature/cva-cfp-2026-template.pdf`). It is a Word-style fill-in template in the *Ledger* journal house style (per Wilmer & Rizun, "How to Write and Format an Article for *Ledger*"). **No LaTeX template is provided.** Key requirements:
- Body: Times New Roman **11.5 pt**, 13.5 pt leading; first paragraph and headings flush left, subsequent paragraphs indented 0.6 cm.
- Margins: left/right **3.1 cm**, top 2.4 cm, bottom 1.0 cm; header 1.5 cm, footer 1.0 cm from edge. Running head "CVA Academic Research Journal 2026".
- Title Times **24 pt** centered; author names 13 pt centered with superscript affiliation symbols (\*, †, ‡, §, ¶); affiliations 8 pt.
- **Abstract:** 10 pt, 1.2 cm side indents, bold lead-in "**Abstract.**", ~150 words (**200 max**), **no citations, no equations, no code**. Followed by a "KEY WORDS" block (≤2 lines).
- Section headings: **13 pt Times Bold, numbered**; the first must be "Introduction" (~500–750 words, 1000 max, no subheadings). Subheadings: **inline, italic, followed by an em dash** (matches our `\paragraph` restyle).
- **References: numbered superscript ENDNOTES** in a single combined "**Notes and References**" section (10 pt) — *Ledger* style, **not author-year**. `paper.tex`'s `plainnat` `\citep{}` author-year scheme must be replaced.
- Figures: 300 DPI (photo) / 600 DPI (line art); in-figure text 8 pt Arial; captions 10 pt. Code: Courier New 10 pt. Display equations centered and numbered.
- Required trailing sections: Acknowledgement, **Author Contributions**, **Conflict of Interest** (may be omitted only if none).

**Review model — resolved: NOT double-blind.** The official template carries a full author block and mandates an Author Contributions section, so the submission is author-identified. The real author block stays; do not anonymize.

**Author.** Nils Bundi, Zurich University of Applied Sciences (ZHAW), bund@zhaw.ch.