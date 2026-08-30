# Best Execution Without a Broker

Working paper / conference submission: *Best Execution Without a Broker: Evaluating Intent-Based Settlement Against the Multi-Factor Best-Execution Standard*.

This README is the single source of truth for all project and submission-venue details. Tooling and Claude instructions defer to it.

## One-line thesis

Intent-based settlement is marketed as delivering best execution through solver competition; this paper tests that claim against the *multi-factor* best-execution standard that MiFID II Art. 27 and FinSA Art. 18 actually impose (price, costs, speed, likelihood of execution and settlement, size, order characteristics), corrects the censoring bias in the price-only literature, and argues that because losing solver bids are published, the execution counterfactual is directly *observable* — turning best execution from an evidentiary problem into a measurement problem.

## Build

```sh
latexmk -pdf paper.tex        # -> paper.pdf (runs bibtex automatically)
python3 code.py               # reproduces/【will reproduce】every §5 computed result
```

TeX Live 2026 (`pdflatex` + `natbib` + `plainnat`). Python 3.9+, deps: `numpy`, `scipy`, `pandas`, `statsmodels` (`matplotlib` for figures). The script is deterministic (seeded).

### Data collection (§3) — done; commands for reproduction

```sh
python3 code.py --collect-pairs          # settled-auction census (competition records) -> data/pairs/*.jsonl.gz
python3 code.py --collect-amm            # route every order through the archival Uniswap SOR -> data/amm_routes.jsonl
python3 code.py --collect-basefees       # per-block base fees -> data/pairs/base_fees.jsonl.gz
python3 code.py --collect-arrivals       # risk-set scan -> arrivals.jsonl.gz (submission cohort, phase 1)
python3 code.py --collect-fill-census    # classify arrivals /USDC -> data/fill_cohort/cohort.jsonl.gz (phase 2)
python3 code.py --pairs --amm-pairs --full --multifactor --timeseries --fill-cohort   # analysis from cache
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