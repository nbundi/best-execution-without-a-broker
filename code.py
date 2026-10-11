"""Canonical analysis script for "The Value of the Marginal Solver".

Single canonical script behind every computed result in the paper
(academic-coding.md). `python3 code.py` runs the full pipeline in paper
order from the cached census and writes results/paper.json plus every
figure; --collect-* flags re-pull the raw data (network, non-deterministic).

Paper elements backed by this script (run_paper blocks, paper-ordered)
----------------------------------------------------------------------
  B1  Sec. 5.1, Table 1, Fig. 5 series   participation by tier, monthly field
  B2  Sec. 5.2, Fig. 2 + 3, Table 2      marginal solver curve S(m), Eq. (2),
                                          by tier and by liquidity quintile
  B3  Sec. 5.3, Fig. 4                   fixed-bid exposure per solver
  B4  Sec. 5.4                           selection wedge; batch/CoW premium
  B5  Sec. 5.5, Fig. 5, Table 4          CIP-74: participation, gap-by-n in
                                          three windows (all bids / survivors),
                                          cap-exposure DiD (Prop. 3')
  B6  Sec. 5.6, Fig. 6                   exit event study (substitution)
  B7  Sec. 5.7                           size gradients, solo decomposition,
                                          descriptive regression
  Fig. 1 (auction anatomy) is a one-off over the same census (see README).
  Data (Sec. 3): CoWOrderbookClient + collectors -> data/ census.

Dependencies : numpy, scipy, pandas, statsmodels, matplotlib
Run          : python3 code.py                 (full paper pipeline)
               python3 code.py --collect-pairs (and other --collect-* flags)
Determinism  : the analysis is a pure function of the cached files in data/;
               block->time uses fixed anchors (_BLOCK_ANCHORS).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
import zlib
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Ledger / CVA house style for figures: 8 pt Arial, 600 DPI line art, fonts
# embedded as editable TrueType (Type 42). Falls back to DejaVu Sans if the
# system has no Arial.
for _p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
_ARIAL = "Arial" if any(f.name == "Arial" for f in _fm.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_ARIAL, "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SEED = 20260810
BPS = 1.0e4
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


# ===========================================================================
# DATA (Section 3): CoW Protocol orderbook API retrieval layer
# ===========================================================================

_COW = "https://api.cow.fi/mainnet"


@dataclass
class CoWOrderbookClient:
    """Client for the CoW Protocol orderbook API (Sec. 3.1).

    Endpoints: /api/v2/solver_competition/{latest,<id>} for the full solver
    competition record (winning and losing solutions, per-order proposed
    executions, scores, and the auction native-price vector), and
    /api/v1/orders/{uid} for order metadata and lifecycle status.
    """

    base_url: str = _COW
    timeout: float = 40.0

    def _get(self, path: str, retries: int = 6) -> dict:
        """GET with rate-limit-aware retry: exponential backoff + jitter on HTTP
        429/5xx (honouring Retry-After) and on transient network errors."""
        url = self.base_url + path
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                           "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return json.loads(body.decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30)
                    time.sleep(wait + random.uniform(0, 0.5))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 30) + random.uniform(0, 0.5))
                    continue
                raise

    def latest_competition(self) -> dict:
        return self._get("/api/v2/solver_competition/latest")

    def competition(self, auction_id: int) -> dict:
        return self._get(f"/api/v2/solver_competition/{auction_id}")

    def order(self, uid: str) -> dict:
        return self._get(f"/api/v1/orders/{uid}")


def value_in_eth(token: str, amount, prices: dict):
    """Value a token amount (atoms) in ETH via the auction native-price vector.

    CoW native prices satisfy value_wei = amount_atoms * price / 1e18, so
    value_eth = amount_atoms * price / 1e36. Confirmed against WETH (price 1e18).
    """
    p = prices.get(token)
    if p is None or amount is None:
        return None
    return int(amount) * int(p) / 1e36


# --- Block 4 (Sec. 4.4): competition intensity -----------------------------

def hhi(volume_by_solver):
    """Volume-weighted HHI over settling solvers, 0..1 scale (Sec. 4.4)."""
    v = np.asarray(volume_by_solver, float)
    s = v / v.sum()
    return float(np.sum(s ** 2))


# ===========================================================================
# PART B2 -- THREE-PAIR WINDOWED COLLECTION (Yuminaga-style short/mid/long-tail)
# ===========================================================================
# The study is scoped to three USDC pairs spanning the liquidity spectrum, sampled
# over monthly windows across the past ~12 months (serves both the pooled average
# and the historical time series). Native ETH (0xeee...) is mapped to WETH.

_USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
_WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
_ETH_PLACEHOLDER = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

# Liquidity tiers vs USDC (Yuminaga-style short/mid/long, but as BASKETS because
# single small-caps are too bursty on CoW). Membership by market-cap class.
TIER_SHORT = {                       # blue-chip crypto (ETH/BTC and their LSD/wrapped)
    _WETH,                                              # WETH
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",       # WBTC
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",       # cbBTC
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",       # stETH
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",       # wstETH
}
TIER_MID = {                         # mid-cap DeFi / protocol tokens
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",       # UNI
    "0x514910771af9ca656af840dff83e8264ecf986ca",       # LINK
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",       # AAVE
    "0xd533a949740bb3306d119cc777fa900ba034cd52",       # CRV
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",       # MKR
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32",       # LDO
    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f",       # SNX
    "0xc00e94cb662c3520282e6f5717214004a7f26888",       # COMP
    "0xc18360217d8f7ab5e7c516566761ea12ce7f9d72",       # ENS
    "0xc944e90c64b2c07662a292be6244bdf05cda44a7",       # GRT
    "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",       # SUSHI
    "0x111111111117dc0aa78b770fa6a738034120c302",       # 1INCH
    "0xba100000625a3754423978a60c9317c58a424e3d",       # BAL
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e",       # YFI
}
# Long-tail = every other non-stable token vs USDC (basket of small-caps).
STABLES = {                          # excluded: stable/stable is not a liquidity tier
    _USDC,
    "0xdac17f958d2ee523a2206206994597c13d831ec7",       # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f",       # DAI
    "0x4c9edd5852cd905f086c759e8383e09bff1e68b3",       # USDe
    "0xdc035d45d973e3ec169d2276ddab16f1e407384f",       # USDS
    "0xe343167631d89b6ffc58b88d6b7fb0228795491d",       # USDG
    "0xd166337499e176bbc38a1fbd113ab144e5bd2df7",       # sUSDat
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8",       # PYUSD
    "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",       # crvUSD
}
TIER_ORDER = ["short", "mid", "long"]


def _norm_tok(t):
    t = (t or "").lower()
    return _WETH if t == _ETH_PLACEHOLDER else t


def _usdc_order(st, bt):
    """For a USDC-paired, non-stable order return (other_token, liquidity_tier);
    None otherwise. Tier by market-cap class: short=blue-chip, mid=mid-cap DeFi,
    long=everything else (small-cap basket). Native ETH is mapped to WETH."""
    ns, nb = _norm_tok(st), _norm_tok(bt)
    if _USDC not in (ns, nb):
        return None
    other = nb if ns == _USDC else ns
    if other in STABLES:                    # skip USDC/USDC and stable/stable
        return None
    tier = "short" if other in TIER_SHORT else "mid" if other in TIER_MID else "long"
    return other, tier


# ---------------------------------------------------------------------------
# Compact, resumable storage for the census (Sec. 3).
#
# Records are stored as gzipped JSON-lines. Each flush appends one *self-
# contained* gzip member (gzip.compress of a JSON-lines blob); concatenated
# members are read transparently by gzip.open, and an interruption can only
# truncate the trailing member, which the reader tolerates. This keeps the raw
# footprint at a few hundred MB rather than the multi-GB of plain JSONL, while
# letting collection resume mid-progress after a crash.
# ---------------------------------------------------------------------------

def _append_gz_members(path, records):
    """Append `records` (dicts) as one complete gzip member of JSON lines.
    Atomic at member granularity: a crash mid-write leaves only a truncated
    final member, never a corrupt earlier one."""
    if not records:
        return
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records).encode("utf-8")
    member = gzip.compress(payload, compresslevel=9)
    with open(path, "ab") as f:
        f.write(member)
        f.flush()
        os.fsync(f.fileno())


def _iter_jsonl_any(path):
    """Yield dicts from a .jsonl or .jsonl.gz file. Tolerates a truncated final
    gzip member left by an interrupted append (stops cleanly at the tear)."""
    if not os.path.exists(path):
        return
    fh = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8")
    with fh:
        while True:
            try:
                line = fh.readline()
            except (EOFError, OSError, gzip.BadGzipFile, UnicodeDecodeError, zlib.error):
                break                                   # truncated trailing member
            if not line:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _pairs_path(out, stem):
    """Prefer the gzipped census file if present, else the legacy plain JSONL."""
    gz = os.path.join(out, stem + ".jsonl.gz")
    return gz if os.path.exists(gz) else os.path.join(out, stem + ".jsonl")


def _as_extracted_auction(r):
    """Normalize a competition record to the analysis schema, whether it is a RAW
    solver_competition API response or an already-extracted record (legacy files /
    the USDC backup). Returns the extracted view, or None if it has no usable orders."""
    if "sols" in r:                                     # already extracted
        return r
    rec, _ = _extract_pair_auction(r)                   # raw API response -> extracted view
    return rec


def _as_meta_row(m):
    """Normalize an order-metadata record to the compact analysis schema, whether it
    is a RAW /orders response or an already-compact row (legacy files)."""
    if "pf" in m or "execSell" in m:                    # already compact
        return m
    return {"uid": m.get("uid"), "status": m.get("status"), "kind": m.get("kind"),
            "class": m.get("class"), "pf": m.get("partiallyFillable"),
            "created": m.get("creationDate"), "validTo": m.get("validTo"),
            "execSell": m.get("executedSellAmount"), "execBuy": m.get("executedBuyAmount"),
            "st": m.get("sellToken"), "bt": m.get("buyToken")}


def _load_competition(out):
    """Auctions in the analysis schema, extracted on the fly from the RAW census
    (competition_raw.jsonl.gz, the single source of truth) if present, else a legacy
    pre-extracted competition.jsonl.gz (e.g. the USDC backup)."""
    raw = os.path.join(out, "competition_raw.jsonl.gz")
    src = raw if os.path.exists(raw) else _pairs_path(out, "competition")
    return [x for x in (_as_extracted_auction(r) for r in _iter_jsonl_any(src)) if x]


def _load_meta(out):
    """Order metadata in the compact schema, unioned (by uid, raw wins) over the RAW
    store (cohort_meta_raw.jsonl.gz, newly fetched) and a legacy compact file
    (cohort_meta.jsonl.gz, reused from a prior sample to skip re-fetching)."""
    raw = os.path.join(out, "cohort_meta_raw.jsonl.gz")
    legacy = _pairs_path(out, "cohort_meta")
    seen, rows = set(), []
    for src in (raw, legacy):                            # raw first so fresh records win
        if src and os.path.exists(src):
            for m in _iter_jsonl_any(src):
                r = _as_meta_row(m)
                u = r.get("uid")
                if u in seen:
                    continue
                seen.add(u)
                rows.append(r)
    return rows



_MAJORS = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
}


def _order_asset(st, bt):
    """The order's liquidity-defining leg for all-pairs stratification: the
    non-major token if exactly one side is a major, else the sell token."""
    st, bt = (st or "").lower(), (bt or "").lower()
    s_major, b_major = st in _MAJORS, bt in _MAJORS
    if s_major and not b_major:
        return bt
    if b_major and not s_major:
        return st
    return st

def _extract_pair_auction(d):
    """From a raw solver_competition record keep only /USDC tier orders and the
    minimal fields the analysis needs. Returns (record_or_None, set_of_uids)."""
    psols, pair_uids = [], set()
    for s in d.get("solutions", []):
        porders = []
        for o in s.get("orders", []):
            st = (o.get("sellToken") or "").lower()
            bt = (o.get("buyToken") or "").lower()
            sa, ba = o.get("sellAmount"), o.get("buyAmount")
            if not st or not bt or not sa or not ba:
                continue
            # all pairs (audit item 0): keep every order, ETH numeraire via the
            # auction native-price vector; "other" is the thinner leg (the token
            # not in the major/numeraire set), used for the liquidity metric.
            other = _order_asset(st, bt)
            porders.append({"id": o["id"], "st": st, "bt": bt,
                            "sa": sa, "ba": ba, "other": other, "tier": None})
            pair_uids.add(o["id"])
        if porders:
            psols.append({"solver": (s.get("solverAddress") or "").lower(),
                          "win": bool(s.get("isWinner")), "filt": bool(s.get("filteredOut")),
                          "score": s.get("score"), "orders": porders})
    if not psols:
        return None, pair_uids
    pv = d.get("auction", {}).get("prices", {})
    toks = {t for s in psols for o in s["orders"] for t in (o["st"], o["bt"])}
    rec = {"auctionId": d.get("auctionId"), "startBlock": d.get("auctionStartBlock"),
           "deadlineBlock": d.get("auctionDeadlineBlock"), "txHashes": d.get("transactionHashes"),
           "prices": {t: pv[t] for t in toks if t in pv}, "sols": psols}
    return rec, pair_uids


def collect_pairs(months=12, id_per_month=200_000, workers=12,
                  chunk=300, flush_every=6000, fresh=False, rate=None):  # pragma: no cover (network)
    """Collect the three study pairs over a full `months`-month census: enumerate
    *every* settlement auction over the trailing window and keep the /USDC tier
    orders. (The earlier twelve-window sampler is `collect_pairs_windowed`.)
    Resumable and compact:

      * data/pairs/competition.jsonl.gz  -- gzipped tier auctions (see helpers)
      * data/pairs/cohort_meta.jsonl.gz  -- gzipped order lifecycle metadata
      * data/pairs/progress.json         -- checkpoint {phase, next_id, ...}

    Collection walks auction ids downward from the latest, checkpointing the
    frontier after each flushed batch; a crash resumes from the checkpoint and an
    in-memory `seen` set (rebuilt from the file) makes re-appends idempotent. Run
    with `--fresh` to discard any prior progress and start over."""
    from concurrent.futures import ThreadPoolExecutor
    out = os.path.join(DATA, "pairs")
    os.makedirs(out, exist_ok=True)
    comp_path = os.path.join(out, "competition.jsonl.gz")
    meta_path = os.path.join(out, "cohort_meta.jsonl.gz")
    # RAW source-of-truth: every downloaded API response is banked verbatim so any
    # field can be re-extracted later WITHOUT re-fetching (the extracted *_path files
    # are just a compact view the analysis pipeline reads).
    raw_path = os.path.join(out, "competition_raw.jsonl.gz")
    meta_raw_path = os.path.join(out, "cohort_meta_raw.jsonl.gz")
    prog_path = os.path.join(out, "progress.json")
    cli = CoWOrderbookClient()

    # Global request pacer: the CoW API sits behind a CloudFront WAF rate rule
    # (~6-7 req/s per IP; exceeding it returns 429 with Retry-After:30). Pacing all
    # workers to a shared rate just under the ceiling maximises goodput and avoids
    # the penalty box. rate=None disables pacing.
    import threading as _thr
    _pace_lock = _thr.Lock()
    _next_slot = [0.0]
    _interval = (1.0 / rate) if rate else 0.0
    def _pace():
        if not _interval:
            return
        with _pace_lock:
            now = time.monotonic()
            wait = _next_slot[0] - now
            if wait > 0:
                time.sleep(wait)
            _next_slot[0] = max(now, _next_slot[0]) + _interval

    def save_prog(p):
        tmp = prog_path + ".tmp"
        json.dump(p, open(tmp, "w"))
        os.replace(tmp, prog_path)                       # atomic checkpoint

    if fresh:
        for p in (comp_path, meta_path, raw_path, meta_raw_path, prog_path):
            if os.path.exists(p):
                os.remove(p)

    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        print(f"[resume] phase={prog['phase']} next_id={prog.get('next_id')}", flush=True)
    else:
        latest = cli.latest_competition()["auctionId"]
        prog = {"phase": "competition", "latest": latest, "months": months,
                "stop_id": latest - months * id_per_month, "next_id": latest}
        save_prog(prog)
        print(f"[census] latest auction {latest}; span "
              f"{prog['stop_id']}..{latest} (~{months} months)", flush=True)

    # -- Phase 1: enumerate all auctions; bank the RAW response verbatim -----
    # The raw solver_competition record is the ONLY stored artifact: every field
    # (winning/losing solutions, per-order executions, native-price vector, gas,
    # ...) is preserved so any view can be re-extracted later WITHOUT re-fetching.
    if prog["phase"] == "competition":
        raw_seen = {r.get("auctionId") for r in _iter_jsonl_any(raw_path)}  # idempotent resume
        print(f"[census] {len(raw_seen)} raw auctions already stored", flush=True)
        next_id, stop_id = prog["next_id"], prog["stop_id"]

        def fetch(aid):
            _pace()
            try:
                return cli.competition(aid)
            except Exception:
                return None

        raw_buf, probed = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            while next_id > stop_id:
                lo = max(next_id - chunk, stop_id)
                ids = list(range(next_id, lo, -1))
                next_id = lo
                probed += len(ids)
                for d in ex.map(fetch, ids):
                    if not d:
                        continue
                    aid = d.get("auctionId")
                    if aid is None or aid in raw_seen:
                        continue
                    au = d.get("auction")
                    if isinstance(au, dict):
                        au.pop("orders", None)               # drop the huge, ~redundant ambient
                        # orderbook (~93% of bytes; recover via collect_arrivals if ever needed).
                        # Everything else -- all solver solutions, auction.prices numeraire vector,
                        # reference scores, tx hashes -- is banked verbatim.
                    raw_buf.append(d)                        # bank (near-)verbatim download
                    raw_seen.add(aid)
                if probed >= flush_every:
                    _append_gz_members(raw_path, raw_buf)
                    prog["next_id"] = next_id
                    save_prog(prog)
                    print(f"  down to id {next_id}; {len(raw_seen)} raw auctions "
                          f"(+{len(raw_buf)} this flush)", flush=True)
                    raw_buf, probed = [], 0
                time.sleep(0.05)                         # gentle pacing
        _append_gz_members(raw_path, raw_buf)
        prog["phase"], prog["next_id"] = "meta", stop_id
        save_prog(prog)
        print(f"[census] competition phase done: {len(raw_seen)} raw auctions", flush=True)

    # -- Phase 2: order lifecycle metadata; bank the RAW order verbatim ------
    if prog["phase"] == "meta":
        # Build the order-uid set by STREAMING the raw competition file (never
        # materialise the 1.5M-auction list -- that balloons RAM and thrashes swap).
        # Cache the uids so a restart is instant instead of re-reading tens of GB.
        uids_cache = os.path.join(out, "order_uids.txt.gz")
        if os.path.exists(uids_cache):
            with gzip.open(uids_cache, "rt") as fh:
                uids = {ln.strip() for ln in fh if ln.strip()}
            print(f"[census] meta: {len(uids)} order uids (from cache)", flush=True)
        else:
            uids = set()
            for d in _iter_jsonl_any(raw_path):
                for s in d.get("solutions", []):
                    for o in s.get("orders", []):
                        oid = o.get("id")
                        if oid:
                            uids.add(oid)
            with gzip.open(uids_cache, "wt") as fh:
                for u in uids:
                    fh.write(u + "\n")
            print(f"[census] meta: {len(uids)} order uids extracted (cached)", flush=True)
        done = {m.get("uid") for m in _iter_jsonl_any(meta_raw_path)}
        # Reuse metadata already on disk from a prior sample (cohort_meta.jsonl.gz):
        # order lifecycle is immutable once final, so skip re-fetching final-status
        # orders; non-final ("open" etc.) are left to refresh.
        _FINAL = {"fulfilled", "expired", "cancelled"}
        reused = 0
        for m in _iter_jsonl_any(_pairs_path(out, "cohort_meta")):
            if m.get("status") in _FINAL and m.get("uid") not in done:
                done.add(m.get("uid")); reused += 1
        todo = sorted(uids - done)
        print(f"[census] meta: {len(done)} done ({reused} reused from prior sample), "
              f"{len(todo)} to fetch", flush=True)

        def meta(uid):
            _pace()
            try:
                return cli.order(uid)                    # keep the raw order object verbatim
            except Exception:
                return None

        raw_buf = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, o in enumerate(ex.map(meta, todo)):
                if o:
                    raw_buf.append(o)                        # bank verbatim order
                if len(raw_buf) >= 3000:
                    _append_gz_members(meta_raw_path, raw_buf)
                    raw_buf = []
                    print(f"  meta {i + 1}/{len(todo)}", flush=True)
        _append_gz_members(meta_raw_path, raw_buf)
        prog["phase"] = "done"
        save_prog(prog)

    # -- Manifest (memory-light: stream the raw file, never materialise it) --
    n_auc = 0
    aid_min = aid_max = blk_min = blk_max = None
    for d in _iter_jsonl_any(raw_path):
        n_auc += 1
        aid = d.get("auctionId")
        if aid is not None:
            aid_min = aid if aid_min is None else min(aid_min, aid)
            aid_max = aid if aid_max is None else max(aid_max, aid)
        sb = d.get("auctionStartBlock")
        if sb:
            blk_min = sb if blk_min is None else min(blk_min, sb)
            blk_max = sb if blk_max is None else max(blk_max, sb)
    uids_cache = os.path.join(out, "order_uids.txt.gz")      # distinct orders (built in meta phase)
    n_orders = None
    if os.path.exists(uids_cache):
        with gzip.open(uids_cache, "rt") as fh:
            n_orders = sum(1 for _ in fh)
    meta_fetched = (sum(1 for _ in _iter_jsonl_any(meta_raw_path))
                    + sum(1 for _ in _iter_jsonl_any(_pairs_path(out, "cohort_meta"))))
    man = {"source": _COW, "chain": "ethereum-mainnet", "mode": "census",
           "latest": prog["latest"], "months": prog.get("months"),
           "n_auctions": n_auc, "pair_orders_total": n_orders,
           "auction_id_min": aid_min, "auction_id_max": aid_max,
           "block_min": blk_min, "block_max": blk_max,
           "meta_fetched": meta_fetched, "raw_auctions": n_auc}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"[census] DONE: {n_auc} raw auctions, {n_orders} distinct orders, "
          f"{meta_fetched} metadata records -> {raw_path}", flush=True)


def _census_span(out):
    """(lo, hi) auction-id range of the settled census, from its progress/manifest."""
    p = os.path.join(out, "progress.json")
    if os.path.exists(p):
        pr = json.load(open(p))
        return pr["stop_id"], pr["latest"]
    man = json.load(open(os.path.join(out, "manifest.json")))
    return man["auction_id_min"] - 1, man["latest"]


def _uid_key(uid):
    return int(uid[2:34], 16)                           # 128-bit hash for the seen-set


def collect_arrivals(workers=12, chunk=1000, flush_every=6000, stride=1):  # pragma: no cover (network)
    """Reorganized scan that captures the batch RISK SET (auction.orders) the settled
    census discarded, so the fill cohort can be censused WITHOUT re-scanning again.
    Walks the census id range in ascending order (first sight = arrival); the earliest
    auction's book is banked as pre-existing (arrived before the window) and excluded.
    Streams each new order's first appearance to arrivals.jsonl.gz. Resumable: the
    seen-set is rebuilt from the arrival/pre-existing files plus a last-id checkpoint;
    compact (uid+aid, gzipped).

    `stride` scans every `stride`-th auction. Because we already know every
    MARKETABLE /USDC order (settled census) and only need to discover UNMARKETABLE
    resting orders -- which persist across many consecutive auctions -- stride>1
    finds essentially all of them at 1/stride the cost (records are ~0.9 MB each)."""
    from concurrent.futures import ThreadPoolExecutor
    out = os.path.join(DATA, "pairs")
    arr_path = os.path.join(out, "arrivals.jsonl.gz")
    pre_path = os.path.join(out, "preexisting.jsonl.gz")
    prog_path = os.path.join(out, "arrivals_progress.json")
    cli = CoWOrderbookClient()
    lo_id, hi_id = _census_span(out)

    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
    else:
        prog = {"next_id": lo_id, "hi_id": hi_id, "seeded": False}
        json.dump(prog, open(prog_path, "w"))
    seen = {_uid_key(r["uid"]) for r in _iter_jsonl_any(pre_path)}
    seen |= {_uid_key(r["uid"]) for r in _iter_jsonl_any(arr_path)}
    print(f"[arrivals] resume: {len(seen)} uids known, next_id={prog['next_id']}/{hi_id}", flush=True)

    def fetch(aid):
        try:
            return cli.competition(aid)
        except Exception:
            return None

    def save_prog():
        tmp = prog_path + ".tmp"
        json.dump(prog, open(tmp, "w"))
        os.replace(tmp, prog_path)

    next_id, seeded = prog["next_id"], prog["seeded"]
    buf, probed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while next_id <= hi_id:
            hi = min(next_id + chunk, hi_id + 1)
            ids = list(range(next_id, hi, stride))
            next_id = hi
            probed += len(ids)
            recs = sorted((x for x in ex.map(fetch, ids) if x), key=lambda x: x.get("auctionId", 0))
            for d in recs:
                orders = d.get("auction", {}).get("orders", [])
                if not seeded:                          # earliest auction = pre-existing book
                    _append_gz_members(pre_path, [{"uid": u} for u in orders])
                    for u in orders:
                        seen.add(_uid_key(u))
                    seeded = True
                    continue
                aid = d.get("auctionId")
                for u in orders:
                    k = _uid_key(u)
                    if k not in seen:
                        seen.add(k)
                        buf.append({"uid": u, "aid": aid})
            if probed >= flush_every:
                _append_gz_members(arr_path, buf)
                prog.update(next_id=next_id, seeded=seeded)
                save_prog()
                print(f"[arrivals] id {next_id}/{hi_id}; {len(seen)} distinct uids "
                      f"(+{len(buf)} this flush)", flush=True)
                buf, probed = [], 0
    _append_gz_members(arr_path, buf)
    prog.update(next_id=next_id, seeded=seeded, done=True)
    save_prog()
    n_arr = sum(1 for _ in _iter_jsonl_any(arr_path))
    print(f"[arrivals] DONE: {len(seen)} distinct uids, {n_arr} in-window arrivals", flush=True)


def collect_fill_census(workers=12, flush_every=4000, batch=20000, max_passes=8):  # pragma: no cover
    """Classify every census arrival as /USDC or not and record its fill outcome, to
    build the full-census submission cohort in data/fill_cohort/cohort.jsonl.gz.
    Reuses what we already have -- marketable /USDC orders are known from the settled
    census (competition.jsonl.gz sols) and their status from cohort_meta.jsonl.gz, so
    only the UNMARKETABLE arrivals need an order-metadata fetch. Non-/USDC arrivals are
    remembered in nonusdc.jsonl.gz so they are never re-fetched. Resumable and
    rate-limit hardened; memory-bounded (streams arrivals, done-set keyed by hash)."""
    from concurrent.futures import ThreadPoolExecutor
    pout = os.path.join(DATA, "pairs")
    out = os.path.join(DATA, "fill_cohort")
    os.makedirs(out, exist_ok=True)
    coh_path = os.path.join(out, "cohort.jsonl.gz")
    non_path = os.path.join(out, "nonusdc.jsonl.gz")    # non-/USDC arrivals (uid hashes)
    arr_path = os.path.join(pout, "arrivals.jsonl.gz")
    cli = CoWOrderbookClient()

    mkt_tier = {}                                       # marketable /USDC order -> tier (no fetch)
    for a in _load_competition(pout):
        for s in a["sols"]:
            for o in s["orders"]:
                mkt_tier[o["id"]] = o["tier"]
    meta_status = {m["uid"]: m.get("status") for m in _load_meta(pout)}
    print(f"[fillcensus] {len(mkt_tier)} marketable /USDC orders, {len(meta_status)} cached statuses",
          flush=True)

    done = {r["k"] for r in _iter_jsonl_any(non_path)}      # non-/USDC, remembered by hash
    done |= {_uid_key(r["uid"]) for r in _iter_jsonl_any(coh_path)}
    print(f"[fillcensus] {len(done)} arrivals already classified", flush=True)

    def fetch_meta(uid):
        try:
            o = cli.order(uid)
            m = _usdc_order(o.get("sellToken"), o.get("buyToken"))
            if not m:
                return (uid, None)                      # not /USDC
            return (uid, {"uid": uid, "tier": m[1], "status": o.get("status"), "marketable": False})
        except Exception:
            return (uid, "retry")

    for p in range(max_passes):
        cbuf, nbuf, pending = [], [], []
        nfetch = 0
        progressed = 0

        def flush_fetch():
            nonlocal cbuf, nbuf, nfetch, progressed
            if not pending:
                return
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for uid, res in ex.map(fetch_meta, pending):
                    if res == "retry":
                        continue                        # transient -> next pass
                    k = _uid_key(uid)
                    if res is None:
                        nbuf.append({"k": k})
                    else:
                        cbuf.append(res)
                    done.add(k)
                    progressed += 1
            nfetch += len(pending)
            pending.clear()
            if cbuf:
                _append_gz_members(coh_path, cbuf); cbuf.clear()
            if nbuf:
                _append_gz_members(non_path, nbuf); nbuf.clear()
            print(f"[fillcensus] pass {p + 1}: fetched {nfetch}, {len(done)} classified", flush=True)

        for r in _iter_jsonl_any(arr_path):             # stream arrivals (bounded memory)
            uid = r["uid"]
            k = _uid_key(uid)
            if k in done:
                continue
            t = mkt_tier.get(uid)
            if t is not None:                           # marketable /USDC: status without fetch
                st = meta_status.get(uid)
                if st is not None:
                    cbuf.append({"uid": uid, "tier": t, "status": st, "marketable": True})
                    done.add(k); progressed += 1
                    if len(cbuf) >= flush_every:
                        _append_gz_members(coh_path, cbuf); cbuf.clear()
                    continue
            pending.append(uid)                         # needs a metadata fetch
            if len(pending) >= batch:
                flush_fetch()
        flush_fetch()
        if cbuf:
            _append_gz_members(coh_path, cbuf)
        if nbuf:
            _append_gz_members(non_path, nbuf)
        if progressed == 0:
            break
    usdc = sum(1 for _ in _iter_jsonl_any(coh_path))
    print(f"[fillcensus] DONE: {usdc} /USDC cohort orders, {len(done)} arrivals classified "
          f"-> {coh_path}", flush=True)


# ===========================================================================
# PART C -- AMM COUNTERFACTUAL (item 5; Sec. 3.2, Eq. pi, Table `tab:pi`)
# ===========================================================================
#
# The Bachu et al. (2024) counterfactual, replicated with the exact tool the
# literature uses: each executed order is routed through the Uniswap
# smart-order-router (router/amm_router.cjs), pinned to its settlement (auction
# deadline) block against the archival RPC. The router splits across v2/v3 pools
# and multi-hop routes -- which a single-path quote cannot -- so large-order
# slippage artifacts are avoided. Price improvement is then, in bps of notional,
#   sell (EXACT_INPUT):  PI = 1e4 * (realised_out - cf_out) / N
#   buy  (EXACT_OUTPUT): PI = 1e4 * (cf_in - realised_in) / N
# with a net variant crediting the direct-swap gas the CoW user avoids, and all
# quantities valued in ETH via the auction native-price vector.

_ROUTER = os.path.join(HERE, "router", "amm_router.cjs")    # Node smart-order-router bridge
_ROUTES_CACHE = os.path.join(DATA, "amm_routes.jsonl")      # cached routed counterfactuals


def _rpc_url():
    u = os.environ.get("ETH_RPC_URL")
    if u:
        return u
    envf = os.path.join(HERE, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            if line.strip().startswith("ETH_RPC_URL="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("ETH_RPC_URL not set (env var or .env file).")


def _rpc_urls():
    """All archival RPC endpoints (ETH_RPC_URL, ETH_RPC_URL_2, ...) from env and
    .env, de-duplicated. Routing fans out one Node subprocess per endpoint, so
    throughput scales with the number of providers. Falls back to _rpc_url()."""
    envmap = {}
    envf = os.path.join(HERE, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                envmap[k.strip()] = v.strip().strip('"').strip("'")
    urls, seen = [], set()
    for key in ["ETH_RPC_URL"] + [f"ETH_RPC_URL_{i}" for i in range(2, 9)]:
        v = (os.environ.get(key) or envmap.get(key) or "").strip().strip('"').strip("'")
        if v.startswith("http") and v not in seen:
            seen.add(v)
            urls.append(v)
    return urls or [_rpc_url()]


class EthRPC:
    """Thread-safe JSON-RPC client with a persistent on-disk response cache, so
    the AMM counterfactual is a deterministic pure function of the cache once
    collected (academic-coding reproducibility rule). Network failures are not
    cached (they retry); genuine empty/revert results are cached as None."""

    def __init__(self, url, cache_path):
        self.url = url
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self._new = 0

    def call(self, method, params, retries: int = 6):
        key = method + "|" + json.dumps(params, separators=(",", ":"))
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        req = urllib.request.Request(self.url, data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(retries):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=40).read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:      # rate limited: back off
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 20)
                    time.sleep(wait + random.uniform(0, 0.5))
                    continue
                return None
            except Exception:
                if attempt < retries - 1:                        # transient: retry, don't cache
                    time.sleep(min(2 ** attempt, 20) + random.uniform(0, 0.3))
                    continue
                return None
            if "error" in r:
                err = r["error"] if isinstance(r["error"], dict) else {}
                if (err.get("code") == -32005 or "rate" in str(r["error"]).lower()) and attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 20) + random.uniform(0, 0.5))
                    continue
                res = None                                       # genuine RPC error: cache as null
            else:
                res = r.get("result")
            with self.lock:
                self.cache[key] = res
                self._new += 1
                if self._new % 300 == 0:
                    self._flush_locked()
            return res
        return None

    def _flush_locked(self):
        tmp = self.cache_path + ".tmp"
        json.dump(self.cache, open(tmp, "w"))
        os.replace(tmp, self.cache_path)

    def flush(self):
        with self.lock:
            self._flush_locked()


def _blk(b):
    return b if isinstance(b, str) else hex(int(b))


def _load_route_cache():
    cache = {}
    if os.path.exists(_ROUTES_CACHE):
        for line in open(_ROUTES_CACHE):
            r = json.loads(line)
            cache[r["id"]] = r
    return cache


# Errors that mean "the RPC was busy", not "this order has no route": these are
# left uncached so a later pass retries them, rather than banked as permanent.
_TRANSIENT_ROUTE_RE = re.compile(
    r"429|rate.?limit|throttl|timeout|ETIMEDOUT|network|econn|socket|"
    r"esockettimedout|missing response|server error|50[234]\b", re.I)


def _is_transient_route_error(r):
    return isinstance(r, dict) and "error" in r and bool(_TRANSIENT_ROUTE_RE.search(str(r["error"])))


def _purge_transient_route_errors(cache):
    """Drop transient-error entries (so they re-route) and rewrite the cache file to
    match. Successes and genuine no-route failures are kept."""
    bad = [k for k, v in cache.items() if _is_transient_route_error(v)]
    if not bad:
        return 0
    for k in bad:
        del cache[k]
    tmp = _ROUTES_CACHE + ".tmp"
    with open(tmp, "w") as fh:
        for v in cache.values():
            fh.write(json.dumps(v) + "\n")
    os.replace(tmp, _ROUTES_CACHE)
    return len(bad)


def route_via_sor(jobs, cache):
    """Route the uncached jobs through the Node smart-order-router bridge, each
    pinned to its settlement block, and append results to _ROUTES_CACHE. Successes
    and genuine failures (e.g. no_route) are cached; rate-limit/transient failures
    are left uncached so a later pass retries them. The bridge's output is read
    incrementally, so even a killed subprocess banks the routes it finished. Re-runs
    skip cached ids; delete data/amm_routes.jsonl to force a full re-route."""
    import subprocess
    import tempfile
    import threading
    todo = [j for j in jobs if j["id"] not in cache]
    if not todo:
        return 0
    # Fan out across up to AMM_RPC_FANOUT endpoints. Default 1 (single RPC): fanning
    # out only helps across INDEPENDENT providers -- two apps on the same account
    # share a compute-unit budget, so they contend and the per-chunk join barrier
    # gates on the slower one. Set fanout>1 only with different-provider RPCs.
    fanout = max(1, int(os.environ.get("AMM_RPC_FANOUT", "1")))
    urls = _rpc_urls()[:fanout]
    n = max(1, len(urls))
    tmp = tempfile.mkdtemp(prefix="amm_")
    parts = [todo[i::n] for i in range(n)]              # round-robin split across endpoints

    def _run(idx, url, part):
        infile = os.path.join(tmp, f"in{idx}.json")
        outfile = os.path.join(tmp, f"out{idx}.jsonl")
        json.dump(part, open(infile, "w"))
        open(outfile, "w").close()
        env = dict(os.environ, ETH_RPC_URL=url)
        try:
            subprocess.run(["node", _ROUTER, infile, outfile], env=env, check=False)
        except Exception as e:                          # noqa: BLE001 -- still ingest partial output
            print(f"  [amm] router subprocess error: {e}", flush=True)

    threads, outfiles = [], []
    for idx, url in enumerate(urls):
        if not parts[idx]:
            continue
        outfiles.append(os.path.join(tmp, f"out{idx}.jsonl"))
        t = threading.Thread(target=_run, args=(idx, url, parts[idx]))
        t.start()
        threads.append(t)
    for t in threads:                                   # wait for all endpoints
        t.join()

    added = 0
    with open(_ROUTES_CACHE, "a") as fh:                # single writer: no cache race
        for outfile in outfiles:
            if not os.path.exists(outfile):
                continue
            for line in open(outfile):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in r or r["id"] in cache or _is_transient_route_error(r):
                    continue
                cache[r["id"]] = r
                fh.write(json.dumps(r) + "\n")
                added += 1
    return added


def collect_amm_routes(chunk=2000, max_passes=8, concurrency=4, fanout=1):  # pragma: no cover (network)
    """Full-census AMM counterfactual: route EVERY winning census order through the
    Uniswap smart-order-router (Bachu et al.), block-pinned. Resumable and
    monitorable -- routes are cached per order in data/amm_routes.jsonl, so
    re-running skips finished ids and continues where it left off, batching in
    chunks so an interruption loses at most one chunk. Rate-limited orders are
    retried across passes (and across re-runs) rather than banked, since Alchemy
    throttles the SOR's many archival multicalls."""
    out = os.path.join(DATA, "pairs")
    raw_path = os.path.join(out, "competition_raw.jsonl.gz")
    meta_raw_path = os.path.join(out, "cohort_meta_raw.jsonl.gz")

    # Memory-light build (fits 8GB RAM): STREAM the raw competition file for winning
    # orders, then stream metadata for just the buy-side uids -- instead of
    # materialising the full 1.5M-auction list and the full metadata dict in RAM
    # (which thrashes swap). Only winning orders are routed; kind defaults to sell,
    # so we need to know only which winners are buy-side.
    # Job list (one per winning order) is cached to disk so a restart skips the
    # ~15-min re-stream of the 47GB raw file; delete amm_jobs.jsonl.gz to rebuild.
    jobs_cache = os.path.join(out, "amm_jobs.jsonl.gz")
    if os.path.exists(jobs_cache):
        with gzip.open(jobs_cache, "rt") as fh:
            jobs = [json.loads(ln) for ln in fh]
        print(f"[amm] {len(jobs)} jobs (from cache)", flush=True)
    else:
        winners = {}                                     # uid -> (block, st, bt, sa, ba)
        for d in _iter_jsonl_any(raw_path):
            blk = d.get("auctionDeadlineBlock")
            if not blk:
                continue
            for s in d.get("solutions", []):
                if not s.get("isWinner"):
                    continue
                for o in s.get("orders", []):
                    sa, ba = o.get("sellAmount"), o.get("buyAmount")
                    st, bt = o.get("sellToken"), o.get("buyToken")
                    if sa and ba and st and bt and int(sa) and int(ba):
                        winners[o["id"]] = (int(blk), st, bt, sa, ba)
        buy_uids = set()                                 # winning orders that are buy-side
        for src in (meta_raw_path, _pairs_path(out, "cohort_meta")):
            if not os.path.exists(src):
                continue
            for m in _iter_jsonl_any(src):
                r = _as_meta_row(m)
                if r.get("kind") == "buy" and r.get("uid") in winners:
                    buy_uids.add(r.get("uid"))
        jobs = []
        for oid, (blk, st, bt, sa, ba) in winners.items():
            buy = oid in buy_uids
            jobs.append({"id": oid, "tokenIn": st, "tokenOut": bt,
                         "amount": str(ba if buy else sa),
                         "tradeType": "EXACT_OUTPUT" if buy else "EXACT_INPUT", "block": blk})
        winners.clear(); buy_uids.clear()
        with gzip.open(jobs_cache, "wt") as fh:
            for j in jobs:
                fh.write(json.dumps(j) + "\n")
        print(f"[amm] {len(jobs)} jobs built and cached", flush=True)
    total = len(jobs)
    cache = _load_route_cache()
    purged = _purge_transient_route_errors(cache)
    print(f"[amm] {total} census orders to route; {len(cache)} already cached "
          f"({purged} transient errors purged for retry)", flush=True)

    # Route across all RPC endpoints INDEPENDENTLY: one worker thread per endpoint,
    # each with its own Node SOR subprocess, pulling batches from a shared dispenser
    # and ingesting under a lock. No cross-endpoint barrier, so a fast (paid) account
    # never waits on a slow (free) one. Primary endpoint gets `concurrency`; extras
    # get half (free tiers are smaller). Node memory caps concurrency on small RAM.
    import threading
    import subprocess
    import tempfile
    urls = _rpc_urls()
    concs = [concurrency] + [max(2, concurrency // 2)] * (len(urls) - 1)
    lock = threading.Lock()
    pos = [0]

    def next_batch():
        with lock:
            batch = []
            while pos[0] < len(jobs) and len(batch) < chunk:
                j = jobs[pos[0]]
                pos[0] += 1
                if j["id"] not in cache:
                    batch.append(j)
            return batch

    def run_node(batch, url, conc, tmpdir, idx):
        infile = os.path.join(tmpdir, f"in{idx}.json")
        outfile = os.path.join(tmpdir, f"out{idx}.jsonl")
        json.dump(batch, open(infile, "w"))
        open(outfile, "w").close()
        env = dict(os.environ, ETH_RPC_URL=url, ROUTER_CONCURRENCY=str(conc))
        try:
            subprocess.run(["node", _ROUTER, infile, outfile], env=env, check=False)
        except Exception as e:                          # noqa: BLE001 -- ingest partial
            print(f"  [amm] router error (ep{idx}): {e}", flush=True)
        res = []
        for line in open(outfile):
            line = line.strip()
            if line:
                try:
                    res.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return res

    def worker(idx, url, conc):
        tmpdir = tempfile.mkdtemp(prefix=f"amm{idx}_")
        while True:
            batch = next_batch()
            if not batch:
                return
            res = run_node(batch, url, conc, tmpdir, idx)
            with lock:
                added = 0
                with open(_ROUTES_CACHE, "a") as fh:    # single lock-held writer
                    for r in res:
                        if "id" in r and r["id"] not in cache and not _is_transient_route_error(r):
                            cache[r["id"]] = r
                            fh.write(json.dumps(r) + "\n")
                            added += 1
                n = len(cache)
            print(f"[amm]   ep{idx}(c={conc}) +{added}; {n}/{total} ({100*n/total:.1f}%)", flush=True)

    for p in range(max_passes):
        before = len(cache)
        if sum(1 for j in jobs if j["id"] not in cache) == 0:
            break
        print(f"[amm] pass {p+1}/{max_passes}: {total-len(cache)} remaining; "
              f"{len(urls)} endpoint(s), conc={concs}", flush=True)
        pos[0] = 0
        threads = [threading.Thread(target=worker, args=(i, u, c))
                   for i, (u, c) in enumerate(zip(urls, concs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if len(cache) == before:
            print("[amm] pass made no progress; stopping -- re-run to continue.", flush=True)
            break

    routed = sum(1 for j in jobs if (cache.get(j["id"]) or {}).get("quote"))
    remaining = sum(1 for j in jobs if j["id"] not in cache)
    print(f"[amm] DONE: {len(cache)}/{total} resolved ({routed} routed, "
          f"{len(cache) - routed} no-route/error); {remaining} still to retry.", flush=True)


def _base_fee(rpc, block):
    b = rpc.call("eth_getBlockByNumber", [_blk(block), False])
    if not b:
        return None
    try:
        return int(b.get("baseFeePerGas", "0x0"), 16)
    except (ValueError, AttributeError):
        return None


def _block_ts(rpc, block):
    b = rpc.call("eth_getBlockByNumber", [_blk(block), False])
    if not b:
        return None
    try:
        return int(b.get("timestamp", "0x0"), 16)
    except (ValueError, AttributeError):
        return None


def collect_base_fees(workers=12, flush_every=5000, max_passes=6):  # pragma: no cover (network)
    """Fetch the EIP-1559 base fee for every distinct settlement (deadline) block in
    the census and store it compactly in data/pairs/base_fees.jsonl.gz -- separate
    from the heavyweight rpc_cache.json (we keep only {blk, bf}, ~10 B/block gzipped,
    not the full block header). Resumable (skips blocks already stored) and rate-limit
    hardened (exponential backoff on 429/5xx; transient failures are retried across
    passes rather than banked). The full-census PI (--amm-pairs --full) reads it."""
    from concurrent.futures import ThreadPoolExecutor
    out = os.path.join(DATA, "pairs")
    bf_path = os.path.join(out, "base_fees.jsonl.gz")
    auc = _load_competition(out)
    blocks = sorted({a["deadlineBlock"] for a in auc if a.get("deadlineBlock")})
    total = len(blocks)
    url = _rpc_url()

    def fetch(blk):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber",
                           "params": [hex(int(blk)), False]}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=40).read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 20)
                    time.sleep(wait + random.uniform(0, 0.5))
                    continue
                return None
            except Exception:                          # noqa: BLE001 -- transient: retry then drop
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 20) + random.uniform(0, 0.3))
                    continue
                return None
            bfx = (r.get("result") or {}).get("baseFeePerGas")
            return {"blk": int(blk), "bf": int(bfx, 16)} if bfx else None
        return None

    done = {r["blk"] for r in _iter_jsonl_any(bf_path)}
    print(f"[basefee] {total} distinct settlement blocks; {len(done)} already stored", flush=True)
    for p in range(max_passes):
        todo = [b for b in blocks if b not in done]
        if not todo:
            break
        print(f"[basefee] pass {p + 1}/{max_passes}: {len(todo)} to fetch", flush=True)
        before = len(done)
        buf = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(fetch, todo):
                if rec:                                # only persist successes; failures retry
                    buf.append(rec)
                    done.add(rec["blk"])
                if len(buf) >= flush_every:
                    _append_gz_members(bf_path, buf)
                    buf = []
                    print(f"[basefee]   {len(done)}/{total} ({100 * len(done) / total:.1f}%)", flush=True)
        _append_gz_members(bf_path, buf)
        if len(done) == before:
            print("[basefee] pass made no progress (rate-limited); stopping -- re-run to continue.", flush=True)
            break
    print(f"[basefee] DONE: {len(done)}/{total} blocks stored -> {bf_path}", flush=True)


# ===========================================================================
# PART E -- TIME SERIES & GOVERNANCE EVENT STUDY (item 4; Sec. 4.5, Eq. did)
# ===========================================================================

_CIP74_TS = 1765152000.0        # 2025-12-08 00:00 UTC: CoW CIP-74 effective date


# ===========================================================================
# PART F -- SOLVER COMPETITION: THE VALUE OF THE MARGINAL SOLVER
# ===========================================================================
#
# The repositioned paper's core analysis. One streaming pass over the census
# builds an order x solver bid panel (per-solver best offer ratio per order),
# from which everything follows without estimation:
#   B. within-order marginal-solver (shortfall) curve -- exact expectation of
#      the best offer over random m-subsets of the observed bidder set,
#   C. leave-one-out pivotal solver values (loss if the best offerer exits),
#   D. winner-vs-best-offer wedge (cost of batch-level selection),
#   E. CIP-74 decomposition into participation vs bidding margins,
#   F. solver entry/exit events with a token-level DiD,
#   G. size and liquidity gradients of the pivotal gap.
# Block->UTC time uses two anchor timestamps (publicnode, fetched 2026-08-31)
# so the analysis stays deterministic and needs no RPC.

_BLOCK_ANCHORS = ((23128367, 1755042743), (25761651, 1786811423))


def _block_time_model():
    (b0, t0), (b1, t1) = _BLOCK_ANCHORS
    slope = (t1 - t0) / (b1 - b0)
    return lambda b: t0 + (b - b0) * slope


def subset_max_expectation(sorted_desc, m):
    """E[max of a uniform random m-subset] of the given offers (Sec. 4, Eq. Vm).

    For offers r_(1) >= ... >= r_(n), P(max = r_(i)) = C(n-i, m-1)/C(n, m):
    exact, distribution-free, within one order's observed bidder set."""
    n = len(sorted_desc)
    if m >= n:
        return sorted_desc[0]
    denom = math.comb(n, m)
    return sum(r * math.comb(n - 1 - i, m - 1) for i, r in enumerate(sorted_desc)) / denom


def _gap_bps(r1, r2):
    return BPS * (r1 - r2) / r2 if r2 else np.nan


def _figure_pivotal(solvers):
    """Concentration of pivotality: per-solver share of aggregate leave-one-out
    user loss (bars) with the cumulative share (line)."""
    top = solvers[:10]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(top))
    tot = sum(s["pivotal_loss_eth"] for s in solvers) or 1.0
    shares = [100 * s["pivotal_loss_eth"] / tot for s in top]
    ax.bar(xs, shares, color="#1f4e79")
    ax.set_xticks(xs, [f"S{i+1}" for i in xs])
    ax.set_xlabel("Solver (by exposure rank)")
    ax.set_ylabel("Share of aggregate fixed-bid exposure (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIGURES, "fig_pivotal.pdf"))
    plt.close(fig)


def _figure_cip74(monthly):
    """Monthly participation (mean bidders, active solvers) and price margin
    (median pivotal gap) with CIP-74 marked -- the decomposition figure."""
    pts = sorted((m for m in monthly if m["month"]), key=lambda x: x["month"])
    xs = [datetime.strptime(m["month"], "%Y-%m").replace(tzinfo=timezone.utc) for m in pts]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True)
    ax1.plot(xs, [m["median_bidders"] for m in pts], "o-", color="#1f4e79",
             label="median bidders per order")
    ax1b = ax1.twinx()
    ax1b.plot(xs, [m["active_solvers"] for m in pts], "s--", color="#a6611a",
              label="active solvers")
    ax1.set_ylabel("Median bidders", color="#1f4e79")
    ax1b.set_ylabel("Active solvers", color="#a6611a")
    ax1b.tick_params(axis="y", labelcolor="#a6611a")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    mg = [m["median_gap_bps"] for m in pts]
    lo = [m.get("median_gap_ci", [None, None])[0] for m in pts]
    hi = [m.get("median_gap_ci", [None, None])[1] for m in pts]
    ax2.plot(xs, mg, "o-", color="#1f4e79", label="median runner-up margin (bps)")
    if all(x is not None for x in lo):
        ax2.fill_between(xs, lo, hi, color="#1f4e79", alpha=0.18, lw=0)
    ax2.set_ylabel("Median runner-up margin (bps)")
    cip = datetime.fromtimestamp(_CIP74_TS, tz=timezone.utc)
    for ax in (ax1, ax2):
        ax.axvline(cip, color="#888888", lw=0.8, ls=":")
        ax.spines[["top"]].set_visible(False)
    ax1.annotate("CIP-74", xy=(cip, ax1.get_ylim()[1]), fontsize=8,
                 ha="left", va="top", color="#555555")
    fig.savefig(os.path.join(FIGURES, "fig_cip74.pdf"))
    plt.close(fig)


_GAP_CAP_BPS = 2000.0     # gaps beyond 20% mean the runner-up bid is not an
                          # economic counterfactual (junk/fallback route); winsorise
_NOTIONAL_CAP_ETH = 1e5   # native-price glitches on exotic tokens produce absurd
                          # ETH valuations; treat as missing above this


# ===========================================================================
# PART G -- EXIT EVENT STUDY (Sec. 5.5, Fig. 5): realized execution quality
# around the December 2025 solver-exit wave
# ===========================================================================
#
# The paper's identified result. Exposure is defined ex ante: each counter-
# asset's share of Sep-Nov 2025 orders whose best offer came from a solver
# that exited in Dec 2025. Outcomes are (a) realized price improvement vs the
# routed AMM counterfactual (settlement-block-pinned SOR quotes cached in
# data/amm_routes.jsonl; net of direct-swap gas via base_fees) and (b) the
# bid-based pivotal gap. Two-way (token x month) demeaned DiD plus an event-
# time path with token-clustered SEs; treated = top exposure tercile.

_EXIT_MONTH = "2025-12"
_PRE_MONTHS = ("2025-09", "2025-10", "2025-11")


def _pi_net_bps(rows, meta, bf):
    """Realized net price improvement vs the cached routed-AMM counterfactual,
    per order id (Sec. 3 outcome; |gross| > 1000 bps dropped as unreliable)."""
    cache = _load_route_cache()
    prices_all = {}
    out = {}
    for r in rows:
        o = r.get("_o")
        if o is None:
            continue
        rt = cache.get(r["oid"])
        if not rt or "error" in rt:
            continue
        prices = r["_prices"]
        N = r["notional_eth"]
        if not N or N <= 0:
            continue
        base = bf.get(int(r["_blk"])) if r.get("_blk") else None
        gas_eth = int(rt["gasUsed"]) * base / 1e18 if base else 0.0
        buy = (meta.get(r["oid"], {}) or {}).get("kind", "sell") == "buy"
        try:
            if buy:
                cf_in = value_in_eth(o["st"], int(rt["quote"]), prices)
                if not cf_in:
                    continue
                pg = BPS * (cf_in - N) / N
                pn = BPS * ((cf_in + gas_eth) - N) / N
            else:
                cf_out = value_in_eth(o["bt"], int(rt["quote"]), prices)
                ro = value_in_eth(o["bt"], int(o["ba"]), prices)
                if not cf_out or not ro:
                    continue
                pg = BPS * (ro - cf_out) / N
                pn = BPS * (ro - (cf_out - gas_eth)) / N
        except (TypeError, ValueError):
            continue
        if abs(pg) <= 1000:
            out[r["oid"]] = pn
    return out


def _figure_exitstudy(paths):
    """Event-time DiD paths for realized PI (vs routed AMM) and pivotal gap
    around the December 2025 exit wave; base month November 2025."""
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.8), sharex=True)
    for ax, key, lab, col in ((axes[0], "pi", "realized PI vs AMM route (bps)", "#1f4e79"),
                              (axes[1], "gap", "bidders per order", "#a6611a")):
        pts = paths[key]
        xs = [datetime.strptime(p["month"], "%Y-%m").replace(tzinfo=timezone.utc) for p in pts]
        bs = [p["beta"] for p in pts]
        se = [p["se"] for p in pts]
        ax.axhline(0, color="#999999", lw=0.7)
        ax.errorbar(xs, bs, yerr=[1.96 * s for s in se], fmt="o-", color=col,
                    lw=1.1, ms=4, capsize=2)
        ax.axvline(datetime(2025, 12, 1, tzinfo=timezone.utc), color="#888888",
                   lw=0.8, ls=":")
        ax.set_ylabel(f"$\\beta$: {lab}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Treated (high-exposure) minus control, relative to Nov 2025", fontsize=8)
    fig.savefig(os.path.join(FIGURES, "fig_exitstudy.pdf"))
    plt.close(fig)


# ===========================================================================
# PART H -- REFINEMENTS (second iteration): continuous liquidity axis,
# size-within-tier curves, solo-order decomposition, batch/CoW premium, and
# the CIP-74 recovery and survivor tests
# ===========================================================================
#
# Paper elements:
#   liquidity   Sec. 5.2, Fig. 3 -- shortfall curve by liquidity quintile
#               (sample-computed metric: log10 census ETH volume per asset)
#   size        Sec. 5.2 -- shortfall by size tercile within tier
#   solo        Sec. 5.1 -- single-bid share by size decile (dust decomposition)
#   batch       Sec. 5.4 -- realized batch/CoW premium vs the selection wedge
#   cip74tests  Sec. 5.5, Table 4 -- conditional gap by n in pre/transition/late
#               windows, all bids vs survivor-only bids (membership-churn test)

_TRANSITION = ("2025-12", "2026-01", "2026-02")
_LATE = ("2026-05", "2026-06", "2026-07", "2026-08")


# ===========================================================================
# ANALYSIS -- one census pass, paper-ordered blocks (run: python3 code.py)
# ===========================================================================

_CAP_PRE_WEI = int(0.012e18)   # fixed pre-CIP-74 per-auction reward cap (mainnet);
                               # replaced by a revenue-tied cap on 8 Dec 2025.
                               # Robustness at half and double this threshold.


def _build_panel(sample=1, load_meta=True):
    """One streaming pass over the census -> order-level rows carrying every
    field the paper's blocks need. Solver addresses are interned to int ids.
    `sample`>1 keeps 1-in-`sample` auctions (by auctionId) to fit small RAM;
    `load_meta=False` skips the metadata dict (only the AMM/fill blocks need it)."""
    out_dir = os.path.join(DATA, "pairs")
    meta = {m["uid"]: m for m in _load_meta(out_dir)} if load_meta else {}
    bf = {r["blk"]: r["bf"] for r in _iter_jsonl_any(_pairs_path(out_dir, "base_fees"))}
    btime = _block_time_model()
    sid, rows = {}, []
    # Stream the competition record source, extracting on the fly from the RAW census
    # (competition_raw.jsonl.gz) if present -- never materialise the auction list.
    _raw = os.path.join(out_dir, "competition_raw.jsonl.gz")
    if os.path.exists(_raw):
        _comp = (_as_extracted_auction(d) for d in _iter_jsonl_any(_raw))
    else:
        _comp = _iter_jsonl_any(_pairs_path(out_dir, "competition"))
    for a in _comp:
        if a is None:
            continue
        if sample > 1 and (a.get("auctionId") or 0) % sample != 0:
            continue
        prices, ts = a.get("prices", {}), None
        if a.get("startBlock"):
            ts = btime(a["startBlock"])
        by, win, winsol = defaultdict(dict), {}, {}
        wscores, lscores = [], []
        for k, s in enumerate(a["sols"]):
            try:
                sc = int(s.get("score") or 0)
            except (TypeError, ValueError):
                sc = 0
            (wscores if s["win"] else lscores).append(sc)
            for o in s["orders"]:
                sa, ba = o.get("sa"), o.get("ba")
                if not sa or not ba or not int(sa) or not int(ba):
                    continue
                r = int(ba) / int(sa)
                d = by[o["id"]]
                j = sid.setdefault(s["solver"], len(sid))
                if j not in d or r > d[j]:
                    d[j] = r
                if s["win"]:
                    win[o["id"]] = (r, o)
                    winsol[o["id"]] = k
        # auction-level winner-vs-runner-up solution score gap (single winner only)
        sgap = (wscores[0] - max(lscores)) if (len(wscores) == 1 and lscores
                                              and wscores[0] > 0) else None
        # batch structure of winning solutions, on the kept (/USDC) orders
        wcount, wcow = Counter(), defaultdict(bool)
        for k, s in enumerate(a["sols"]):
            if not s["win"]:
                continue
            os_ = [o for o in s["orders"] if o.get("sa") and o.get("ba")]
            wcount[k] = len(os_)
            pairs = {(o["st"], o["bt"]) for o in os_}
            for o in os_:
                if (o["bt"], o["st"]) in pairs:
                    wcow[k] = True
        for oid, d in by.items():
            if oid not in win:
                continue
            wr, wo = win[oid]
            offers = sorted(d.items(), key=lambda kv: -kv[1])
            k = winsol[oid]
            rows.append({
                "oid": oid, "auc": a.get("auctionId"),
                "tier": wo["tier"], "other": wo["other"], "ts": ts,
                "month": (datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
                          if ts else None),
                "notional_eth": value_in_eth(wo["st"], wo["sa"], prices),
                "offers": offers, "n": len(offers), "winner_ratio": wr,
                "batched": wcount[k] >= 2, "cow": wcow[k], "sgap": sgap,
                "_o": wo, "_prices": prices, "_blk": a.get("deadlineBlock"),
            })
    months = sorted({r["month"] for r in rows if r["month"]})
    addr = [None] * len(sid)
    for a_, j in sid.items():
        addr[j] = a_
    return rows, months, meta, bf, addr


def _ragged_gather(starts, lens, sampled):
    """Row indices for a cluster-resample: concatenate the contiguous blocks
    [starts[c] : starts[c]+lens[c]] for each sampled cluster c, vectorized."""
    take = lens[sampled]
    total = int(take.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    off = np.zeros(len(sampled) + 1, dtype=np.int64)
    np.cumsum(take, out=off[1:])
    idx = np.arange(total, dtype=np.int64) - np.repeat(off[:-1], take)
    return np.repeat(starts[sampled], take) + idx


def _marginal_value_curve(rows, M=5, B=400, seed=SEED):
    """The paper's central estimand (audit A/E): the order-level competition-
    depth counterfactual Delta_i(m) = 10^4 (r_i^(1) - Vhat_i(m)) / r_i^(1), the
    value forgone by admitting only m of the order's observed competitive bids.
    Reports mean, median, and the marginal value delta(m)=Delta(m-1)-Delta(m),
    each with an auction-clustered percentile bootstrap 95% CI (B replicates,
    resampling auctions). Pooled over the balanced n>=M panel, top-1% single-
    bidder-shortfall trimmed."""
    S, aucs = [], []
    for r in rows:
        if r["n"] < M:
            continue
        ratios = [x[1] for x in r["offers"]]
        best = ratios[0]
        S.append([BPS * (best - subset_max_expectation(ratios, m)) / best
                  for m in range(1, M + 1)])
        aucs.append(r["auc"])
    S = np.asarray(S, float)
    keep = S[:, 0] <= np.quantile(S[:, 0], 0.99)
    S, aucs = S[keep], np.asarray(aucs, object)[keep]
    order = np.argsort(aucs, kind="stable")
    S = S[order]
    aucs = aucs[order]
    _, starts, lens = np.unique(aucs, return_index=True, return_counts=True)
    starts = starts.astype(np.int64)
    lens = lens.astype(np.int64)
    C = len(starts)

    def stats(Smat):
        med = np.median(Smat, axis=0)
        mean = Smat.mean(axis=0)
        dmed = -np.diff(med)          # delta(m) = Delta(m-1) - Delta(m)
        dmean = -np.diff(mean)
        return med, mean, dmed, dmean

    med0, mean0, dmed0, dmean0 = stats(S)
    rng = np.random.default_rng(seed)
    bmed = np.empty((B, M)); bmean = np.empty((B, M))
    bdmed = np.empty((B, M - 1)); bdmean = np.empty((B, M - 1))
    for b in range(B):
        idx = _ragged_gather(starts, lens, rng.integers(0, C, C))
        bmed[b], bmean[b], bdmed[b], bdmean[b] = stats(S[idx])

    def ci(arr, i):
        lo, hi = np.percentile(arr[:, i], [2.5, 97.5])
        return [round(float(lo), 2), round(float(hi), 2)]
    out = {"n_orders": int(len(S)), "n_auctions": int(C), "B": B,
           "shortfall": {}, "marginal_value": {}}
    for m in range(1, M + 1):
        out["shortfall"][str(m)] = {
            "median_bps": round(float(med0[m - 1]), 2), "median_ci": ci(bmed, m - 1),
            "mean_bps": round(float(mean0[m - 1]), 2), "mean_ci": ci(bmean, m - 1)}
    for m in range(2, M + 1):
        out["marginal_value"][str(m)] = {
            "median_bps": round(float(dmed0[m - 2]), 2), "median_ci": ci(bdmed, m - 2),
            "mean_bps": round(float(dmean0[m - 2]), 2), "mean_ci": ci(bdmean, m - 2)}
    return out


def _median_ci(values, clusters, B=300, seed=SEED):
    """Auction-clustered percentile-bootstrap 95% CI for a median (audit H)."""
    v = np.asarray(values, float)
    c = np.asarray(clusters, object)
    order = np.argsort(c, kind="stable")
    v = v[order]; c = c[order]
    _, starts, lens = np.unique(c, return_index=True, return_counts=True)
    starts = starts.astype(np.int64); lens = lens.astype(np.int64); C = len(starts)
    rng = np.random.default_rng(seed)
    meds = np.empty(B)
    for b in range(B):
        idx = _ragged_gather(starts, lens, rng.integers(0, C, C))
        meds[b] = np.median(v[idx])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return round(float(np.median(v)), 2), [round(float(lo), 2), round(float(hi), 2)]


def _shortfall_curves(rows, key_fn, M=5):
    """Median (and mean) S(m) curves grouped by key_fn(row), balanced n>=M
    panel, top-1% single-bidder-shortfall trim per group (Eq. subset)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = key_fn(r)
        if k is None or r["n"] < M:
            continue
        ratios = [x[1] for x in r["offers"]]
        best = ratios[0]
        for m in range(1, M + 1):
            acc[k][m].append(BPS * (best - subset_max_expectation(ratios, m)) / best)
    out = {}
    for k, cell in acc.items():
        keep = np.asarray(cell[1]) <= np.quantile(cell[1], 0.99)
        out[k] = {str(m): {"median_bps": round(float(np.median(np.asarray(cell[m])[keep])), 2),
                           "mean_bps": round(float(np.mean(np.asarray(cell[m])[keep])), 2)}
                  for m in range(1, M + 1)}
        out[k]["n"] = int(keep.sum())
    return out


def _figure_liquidity(liq, szq):
    """Fig. 2: median shortfall curve along the two axes -- liquidity quintiles
    (left) and order-size quintiles (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    for ax, data, lo, hi, ttl in (
            (axes[0], liq, "L1 (thinnest)", "L5 (deepest)", "By liquidity quintile"),
            (axes[1], szq, "S1 (smallest)", "S5 (largest)", "By order-size quintile")):
        pref = lo[0]
        for q in range(1, 6):
            c = data.get(q) or data.get(str(q))
            if not c:
                continue
            ms = range(1, 6)
            ax.plot(ms, [c[str(m)]["median_bps"] for m in ms], "o-", color=cols[q - 1],
                    label=(lo if q == 1 else hi if q == 5 else f"{pref}{q}"))
        ax.set_xlabel("Number of competing solvers $m$")
        ax.set_title(ttl, fontsize=8)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median shortfall from full competition (bps)")
    fig.savefig(os.path.join(FIGURES, "fig_liquidity.pdf"))
    plt.close(fig)


def _asset_did(df, expo, months, hi, lo, base_month="2025-11"):
    """Shared asset-level DiD: counter-asset x month panel with equal asset
    weights, treated = exposure >= hi, control <= lo, within-asset demeaning
    plus liquidity-quintile x month dummies, asset-clustered SEs. Quintiles
    without both a treated and a control asset carry no contrast and are
    dropped. Returns DiD coefficients (impact = Dec-Feb, recovery = Mar+)
    and event-time paths."""
    d = df[df["other"].isin(expo)].dropna(subset=["logv"]).copy()
    d["expo"] = d["other"].map(expo)
    d["treated"] = d["expo"] >= hi
    d = d[d["treated"] | (d["expo"] <= lo)].copy()
    p = d.groupby(["other", "logv", "treated", "month"]).agg(
        n_orders=("oid", "size"), bidders=("n", "mean"),
        piw=("piw", "median"), gapw=("gapw", "median")).reset_index()
    p = p[p["n_orders"] >= 10]
    # liquidity strata recut over the analysis assets themselves, so each
    # stratum mixes treated and control (a stratum without both is dropped)
    am = p.drop_duplicates("other")[["other", "logv"]]
    am["stratum"] = pd.qcut(am["logv"], 5, labels=False, duplicates="drop")
    p = p.merge(am[["other", "stratum"]], on="other")
    ctr = p.groupby("stratum")["treated"].agg(["any", "all"])
    good = ctr.index[ctr["any"] & ~ctr["all"]]
    p = p[p["stratum"].isin(good)]
    ok = p[p["month"].isin(_PRE_MONTHS)].groupby("other")["month"].nunique()
    p = p[p["other"].isin(ok[ok >= 2].index)].copy()
    p["t_imp"] = (p["treated"] & p["month"].isin(_TRANSITION)).astype(float)
    p["t_rec"] = (p["treated"] & (p["month"] >= "2026-03")).astype(float)
    tmd = pd.get_dummies(p["stratum"].astype(int).astype(str) + "|" + p["month"],
                         drop_first=True).astype(float)

    def _fe_ols(y, xcols, dummies):
        dd = pd.concat([p[[y, "other"] + xcols], dummies], axis=1).dropna()
        X = dd.drop(columns=[y, "other"])
        Xd = X.sub(X.groupby(dd["other"]).transform("mean"))
        keep_x = [c for c in Xd.columns if c in xcols or Xd[c].std() > 1e-9]
        Xd = Xd[keep_x]
        yd = dd[y] - dd[y].groupby(dd["other"]).transform("mean")
        f = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": dd["other"]})
        return f, dd

    res = {"n_treated_assets": int(p.loc[p["treated"], "other"].nunique()),
           "strata_with_contrast": sorted(int(q) for q in good),
           "n_control_assets": int(p.loc[~p["treated"], "other"].nunique()),
           "median_exposure_treated": round(float(np.median(
               [expo[t] for t in p.loc[p["treated"], "other"].unique()])), 3)
           if p["treated"].any() else None}
    for y in ("bidders", "piw", "gapw"):
        f, dd = _fe_ols(y, ["t_imp", "t_rec"], tmd)
        res[f"did_{y}"] = {"impact": round(float(f.params["t_imp"]), 2),
                           "se_impact": round(float(f.bse["t_imp"]), 2),
                           "recovery": round(float(f.params["t_rec"]), 2),
                           "se_recovery": round(float(f.bse["t_rec"]), 2),
                           "n_obs": int(f.nobs), "n_assets": int(dd["other"].nunique())}
    paths = {}
    for y, lbl in (("bidders", "bidders"), ("piw", "pi")):
        cols, evd = [], pd.DataFrame(index=p.index)
        for m in months:
            if m == base_month:
                continue
            c = f"t_{m}"
            evd[c] = ((p["month"] == m) & p["treated"]).astype(float)
            cols.append(c)
        f, _ = _fe_ols(y, [], pd.concat([evd, tmd], axis=1))
        paths[lbl] = [{"month": c[2:], "beta": round(float(f.params[c]), 2),
                       "se": round(float(f.bse[c]), 2)} for c in cols]
    res["event_time"] = paths
    return res


def run_paper():
    """Reproduce every computed result in the paper, in paper order, from the
    cached census. Writes results/paper.json and all figures."""
    rows, months, meta, bf, addr = _build_panel()
    pi = _pi_net_bps(rows, meta, bf)

    # the paper's sole stratification: a sample-computed liquidity metric per
    # counter-asset (log10 census ETH volume), cut into quintiles Q1..Q5
    vol = defaultdict(float)
    for r in rows:
        ne = r["notional_eth"]
        if ne and 0 < ne <= _NOTIONAL_CAP_ETH:
            vol[r["other"]] += ne
    logv = {t: math.log10(v) for t, v in vol.items() if v > 0}
    qs = np.quantile(list(logv.values()), [0.2, 0.4, 0.6, 0.8])
    quint = {t: int(np.searchsorted(qs, lv)) + 1 for t, lv in logv.items()}
    ncuts = np.quantile([r["notional_eth"] for r in rows
                         if r["notional_eth"] and 0 < r["notional_eth"] <= _NOTIONAL_CAP_ETH],
                        [0.2, 0.4, 0.6, 0.8])
    for r in rows:
        r["liq"] = quint.get(r["other"])
        r["logv"] = logv.get(r["other"])
        ne = r["notional_eth"]
        r["szq"] = (int(np.searchsorted(ncuts, ne)) + 1
                    if ne and 0 < ne <= _NOTIONAL_CAP_ETH else None)
    out = {"n_orders": len(rows), "months": months, "n_solvers_ever": len(
        {r["offers"][0][0] for r in rows} | {r["offers"][1][0] for r in rows if r["n"] > 1})}

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("offers", "_o", "_prices", "_blk")} for r in rows])
    out["n_orders_unclassified_liquidity"] = int(df["liq"].isna().sum())
    df["top"] = [r["offers"][0][0] for r in rows]
    df["top2"] = [r["offers"][1][0] if r["n"] > 1 else -1 for r in rows]
    df["gap_bps"] = [(_gap_bps(r["offers"][0][1], r["offers"][1][1])
                      if r["n"] >= 2 else np.nan) for r in rows]
    df["gapw"] = df["gap_bps"].clip(upper=_GAP_CAP_BPS)
    df.loc[df["notional_eth"] > _NOTIONAL_CAP_ETH, "notional_eth"] = np.nan
    df["wedge_bps"] = [_gap_bps(r["offers"][0][1], r["winner_ratio"]) for r in rows]
    df["piw"] = df["oid"].map(pi).clip(-100, 100)
    df["post"] = df["ts"] >= _CIP74_TS
    out["share_gap_winsorised"] = round(float((df["gap_bps"] > _GAP_CAP_BPS).mean()), 5)

    # --- B1: the competitive field (Sec. 5.1, Table 1; Fig. 5 series) -----
    part = {}
    for q in (1, 2, 3, 4, 5):
        s = df[df["liq"] == q]
        part[f"Q{q}"] = {"n_orders": int(len(s)),
                      "mean_bidders": round(float(s["n"].mean()), 2),
                      "median_bidders": int(s["n"].median()),
                      "p10_bidders": int(s["n"].quantile(0.1)),
                      "p90_bidders": int(s["n"].quantile(0.9)),
                      "share_solo": round(float((s["n"] == 1).mean()), 4),
                      "median_gap_bps": round(float(s["gap_bps"].median()), 2),
                      "p10_gap_bps": round(float(s["gap_bps"].quantile(0.1)), 2),
                      "p90_gap_bps": round(float(s["gap_bps"].quantile(0.9)), 2),
                      "mean_gap_w_bps": round(float(s["gapw"].mean()), 2)}
    out["participation"] = part
    monthly = []
    for mo, g in df[df["month"].notna()].groupby("month"):
        gg = g.dropna(subset=["gap_bps"])
        med, ci = _median_ci(gg["gap_bps"].values, gg["auc"].values) if len(gg) else (None, [None, None])
        monthly.append({"month": mo, "n_orders": int(len(g)),
                        "mean_bidders": round(float(g["n"].mean()), 2),
                        "median_bidders": int(g["n"].median()),
                        "active_solvers": int(len(set(g["top"]) | set(g.loc[g["top2"] >= 0, "top2"]))),
                        "median_gap_bps": med, "median_gap_ci": ci})
    out["monthly"] = monthly
    out["monthly_corr_medbidders_medgap"] = round(float(np.corrcoef(
        [m["median_bidders"] for m in monthly],
        [m["median_gap_bps"] for m in monthly])[0, 1]), 3)
    out["monthly_corr_meanbidders_medgap"] = round(float(np.corrcoef(
        [m["mean_bidders"] for m in monthly],
        [m["median_gap_bps"] for m in monthly])[0, 1]), 3)

    # --- B2: marginal solver curve by liquidity quintile (Sec. 5.2, Fig. 2,
    #         Table 2) ----------------------------------------------------
    pooled = _shortfall_curves(rows, lambda r: "pooled")["pooled"]
    out["shortfall_pooled"] = pooled
    out["marginal_value_curve"] = _marginal_value_curve(rows)   # audit A/E: Delta_i + bootstrap CIs
    out["marginal_value_median_bps"] = {
        str(m): round(pooled[str(m - 1)]["median_bps"] - pooled[str(m)]["median_bps"], 2)
        for m in range(2, 6)}
    out["marginal_value_mean_bps"] = {
        str(m): round(pooled[str(m - 1)]["mean_bps"] - pooled[str(m)]["mean_bps"], 2)
        for m in range(2, 6)}
    liq = _shortfall_curves(rows, lambda r: r["liq"])
    out["shortfall_liquidity_quintiles"] = {str(k): v for k, v in sorted(liq.items())}
    szq = _shortfall_curves(rows, lambda r: r["szq"])
    out["shortfall_size_quintiles"] = {str(k): v for k, v in sorted(szq.items())}
    _figure_liquidity(liq, szq)

    # --- B3: pivotal solvers / exposure concentration (Sec. 5.3, Fig. 4) --
    piv = defaultdict(lambda: {"orders": 0, "loss_eth": 0.0, "gaps": []})
    volw = defaultdict(float)
    for r in rows:
        ne = r["notional_eth"]
        if ne and ne > _NOTIONAL_CAP_ETH:
            ne = None
        if ne:
            volw[r["offers"][0][0]] += ne
        if r["n"] < 2:
            continue
        g = _gap_bps(r["offers"][0][1], r["offers"][1][1])
        if not np.isfinite(g) or g <= 0:
            continue
        g = min(g, _GAP_CAP_BPS)
        p_ = piv[r["offers"][0][0]]
        p_["orders"] += 1
        p_["gaps"].append(g)
        if ne:
            p_["loss_eth"] += ne * g / BPS
    solvers = sorted(({"solver": addr[j], "pivotal_orders": v["orders"],
                       "pivotal_loss_eth": round(v["loss_eth"], 2),
                       "median_gap_when_pivotal_bps": round(float(np.median(v["gaps"])), 2)}
                      for j, v in piv.items() if v["orders"]),
                     key=lambda x: -x["pivotal_loss_eth"])
    tot = sum(s["pivotal_loss_eth"] for s in solvers) or 1.0
    for i, s in enumerate(solvers):
        s["rank"], s["loss_share"] = i + 1, round(s["pivotal_loss_eth"] / tot, 4)
    out["pivotal_solvers"] = solvers[:15]
    out["pivotal_total_loss_eth"] = round(tot, 1)
    out["pivotal_top1_share"] = solvers[0]["loss_share"] if solvers else None
    out["pivotal_top2_share"] = round(sum(s["loss_share"] for s in solvers[:2]), 4)
    out["pivotal_top3_share"] = round(sum(s["loss_share"] for s in solvers[:3]), 4)
    out["hhi_pivotal"] = round(hhi([s["pivotal_loss_eth"] for s in solvers]), 4)
    out["hhi_best_offer_volume"] = round(hhi(list(volw.values())), 4)
    _figure_pivotal(solvers)

    # --- B4: selection wedge and batch/CoW premium (Sec. 5.4) -------------
    w = df["wedge_bps"].dropna()
    out["wedge"] = {"share_winner_not_best": round(float((w > 1e-9).mean()), 4),
                    "median_wedge_when_positive_bps":
                        round(float(w[w > 1e-9].median()), 2) if (w > 1e-9).any() else None}
    okd = df.dropna(subset=["notional_eth", "liq"])
    tc = {q: np.quantile(okd.loc[okd["liq"] == q, "notional_eth"], [1 / 3, 2 / 3])
          for q in (1, 2, 3, 4, 5)}
    okd = okd.dropna(subset=["piw", "month"]).copy()
    okd["szb"] = [int(np.searchsorted(tc[q], v))
                  for q, v in zip(okd["liq"], okd["notional_eth"])]
    for key in ("batched", "cow"):
        diffs, share_n, tot_n = [], 0, 0
        for _, g in okd.groupby(["liq", "szb", "month"]):
            a_, b_ = g.loc[g[key], "piw"], g.loc[~g[key], "piw"]
            if len(a_) >= 5 and len(b_) >= 5:
                diffs.append(float(a_.median() - b_.median()))
            share_n += len(a_)
            tot_n += len(g)
        out[f"{key}_premium_bps_median_matched"] = round(float(np.median(diffs)), 2)
        out[f"{key}_share_of_orders"] = round(share_n / tot_n, 3) if tot_n else None

    # --- B5: competition over time and CIP-74 (Sec. 5.5, Fig. 5, Table 4) -
    _figure_cip74(monthly)
    cip = {}
    for lbl, s in (("pre", df[~df["post"]]), ("post", df[df["post"]])):
        cip[lbl] = {"months": int(s["month"].nunique()), "n_orders": int(len(s)),
                    "mean_bidders": round(float(s["n"].mean()), 2),
                    "share_solo": round(float((s["n"] == 1).mean()), 4),
                    "active_solvers_per_month": round(float(
                        s.groupby("month").apply(lambda g: len(
                            set(g["top"]) | set(g.loc[g["top2"] >= 0, "top2"]))).mean()), 1),
                    "median_gap_bps": round(float(s["gap_bps"].median()), 2)}
    out["cip74_participation"] = cip

    def period(mo):
        if mo is None:
            return None
        if mo <= "2025-11":
            return "pre"
        if mo in _TRANSITION:
            return "transition"
        if mo in _LATE:
            return "late"
        return None
    act = defaultdict(lambda: defaultdict(int))
    for r in rows:
        p_ = period(r["month"])
        if p_ is None:
            continue
        for j, _ in r["offers"][:2]:
            act[j][p_] += 1
    survivors = {j for j, pp in act.items()
                 if pp.get("pre", 0) >= 100 and pp.get("late", 0) >= 100}
    out["n_survivors"] = len(survivors)

    def gaptab(offsel):
        tab = defaultdict(lambda: defaultdict(list))
        for r in rows:
            p_ = period(r["month"])
            if p_ is None:
                continue
            off = offsel(r)
            if len(off) < 2:
                continue
            g = min(_GAP_CAP_BPS, _gap_bps(off[0], off[1]))
            if np.isfinite(g):
                tab[len(off)][p_].append(g)
        return {str(nb): {p_: round(float(np.median(v)), 2)
                          for p_, v in per.items() if len(v) >= 200}
                for nb, per in sorted(tab.items()) if 2 <= nb <= 12}
    out["gap_by_n_allbids"] = gaptab(lambda r: [x[1] for x in r["offers"]])
    out["gap_by_n_survivors"] = gaptab(
        lambda r: sorted((x[1] for x in r["offers"] if x[0] in survivors), reverse=True))

    # cap-exposure test (Prop. 3'): ex ante share of pre-period orders in
    # auctions whose score margin exceeded the fixed pre-reform cap
    pre_rows = [r for r in rows if r["month"] in _PRE_MONTHS]
    captest = {}
    for mult, lbl in ((1.0, "cap"), (0.5, "cap_half"), (2.0, "cap_double")):
        thr = _CAP_PRE_WEI * mult
        tot_c, hit_c = Counter(), Counter()
        for r in pre_rows:
            if r["sgap"] is None:
                continue
            tot_c[r["other"]] += 1
            if r["sgap"] > thr:
                hit_c[r["other"]] += 1
        expo = {t: hit_c[t] / c for t, c in tot_c.items() if c >= 30}
        vals = sorted(expo.values())
        if not vals:
            continue
        hi = float(np.quantile(vals, 2 / 3))
        lo = float(np.quantile(vals, 1 / 3))
        r_ = _asset_did(df, expo, months, hi=max(hi, 1e-9), lo=lo)
        r_["threshold_eth"] = round(thr / 1e18, 4)
        r_["share_binding_pre_orders"] = round(
            sum(hit_c.values()) / max(1, sum(tot_c.values())), 4)
        r_["exposure_cuts"] = [round(lo, 4), round(hi, 4)]
        if mult != 1.0:
            r_ = {k: r_[k] for k in ("did_gapw", "threshold_eth")}
        captest[lbl] = r_
    out["cap_exposure_test"] = captest

    # --- B6: exit event study (Sec. 5.6, Fig. 6) --------------------------
    act_m = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r["month"]:
            continue
        for j, _ in r["offers"][:2]:
            act_m[j][r["month"]] += 1
    exits = set()
    for j, mm in act_m.items():
        active = [m for m in months if mm.get(m, 0) >= 100]
        if active and active[-1] == _EXIT_MONTH and max(mm.values()) >= 1500:
            exits.add(j)
    ex1 = defaultdict(int)
    tot_e = Counter()
    for r in pre_rows:
        tot_e[r["other"]] += 1
        if r["offers"][0][0] in exits or (r["n"] > 1 and r["offers"][1][0] in exits):
            ex1[r["other"]] += 1
    expo_exit = {t: ex1[t] / c for t, c in tot_e.items() if c >= 30}
    exit_res = _asset_did(df, expo_exit, months, hi=0.25, lo=0.05)
    exit_res["exit_solvers"] = sorted(addr[j] for j in exits)
    exit_res["exiter_best_offer_share_pre"] = round(float(np.mean(
        [1.0 if r["offers"][0][0] in exits else 0.0 for r in pre_rows])), 3)
    out["exit_study"] = exit_res
    _figure_exitstudy({"pi": exit_res["event_time"]["pi"],
                       "gap": exit_res["event_time"]["bidders"]})

    # --- B7: size, solo, and the cross-section (Sec. 5.7) -----------------
    dm = df.dropna(subset=["gapw", "notional_eth"]).copy()
    dm = dm[dm["notional_eth"] > 0]
    dm["dec"] = pd.qcut(dm["notional_eth"], 10, labels=False, duplicates="drop")
    out["gap_by_size_decile"] = [
        {"decile": int(d) + 1, "n": int(len(g)),
         "median_notional_eth": round(float(g["notional_eth"].median()), 3),
         "median_gap_bps": round(float(g["gapw"].median()), 2),
         "mean_bidders": round(float(g["n"].mean()), 2)}
        for d, g in dm.groupby("dec")]
    n2r = {r["oid"]: r for r in rows}
    sz = {}
    for q in (1, 2, 3, 4, 5):
        sub = [n2r[o] for o in dm.loc[dm["liq"] == q, "oid"] if n2r[o]["n"] >= 5]
        if len(sub) < 100:
            continue
        cuts = np.quantile([r["notional_eth"] for r in sub], [1 / 3, 2 / 3])
        cells = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list)}
        for r in sub:
            b = int(np.searchsorted(cuts, r["notional_eth"]))
            ratios = [x[1] for x in r["offers"]]
            best = ratios[0]
            for m in range(1, 6):
                cells[b][m].append(BPS * (best - subset_max_expectation(ratios, m)) / best)
        sz[f"Q{q}"] = {lbl: {str(m): round(float(np.median(v[m])), 2) for m in range(1, 6)}
                       for lbl, v in zip(("small", "mid", "large"), cells.values())}
    out["shortfall_by_quintile_and_size_median"] = sz
    # solo shares must include single-bid orders, which have no margin, so
    # they are computed on the full frame rather than the margin panel
    ds = df.dropna(subset=["notional_eth"]).copy()
    ds = ds[ds["notional_eth"] > 0]
    ds["dec"] = pd.qcut(ds["notional_eth"], 10, labels=False, duplicates="drop")
    solo = ds.groupby("dec").apply(lambda g: round(float((g["n"] == 1).mean()), 3))
    out["solo_share_by_size_decile"] = {str(int(d) + 1): v for d, v in solo.items()}
    dq = ds.dropna(subset=["liq"]).copy()
    dq["szb"] = np.where(dq["dec"] <= 2, "small", np.where(dq["dec"] >= 7, "large", "mid"))
    out["solo_share_by_quintile_size"] = {
        f"Q{int(q)}|{b}": round(float((g["n"] == 1).mean()), 3)
        for (q, b), g in dq.groupby(["liq", "szb"]) if len(g) >= 200}
    os.makedirs(RESULTS, exist_ok=True)
    # --- sample flow (audit I): the sequence from all census auctions to the
    #     balanced >=5-bidder estimand panel, in exact final counts
    out["sample_flow"] = {
        "census_auctions": int(df["auc"].nunique()),
        "settled_order_instances": int(len(rows)),
        "distinct_executed_orders": int(df["oid"].nunique()),
        "orders_with_two_or_more_bids": int((df["n"] >= 2).sum()),
        "orders_in_balanced_panel_n5": int((df["n"] >= 5).sum()),
        "balanced_panel_after_trim": out["marginal_value_curve"]["n_orders"],
        "orders_with_routed_counterfactual": int(df["piw"].notna().sum()),
    }

    json.dump(out, open(os.path.join(RESULTS, "paper.json"), "w"), indent=2)
    return out


def main():
    collectors = {"--collect-pairs": lambda: collect_pairs(months=16, workers=8, flush_every=2000, rate=6.0),
                  "--collect-amm": lambda: collect_amm_routes(concurrency=8),
                  "--collect-basefees": lambda: collect_base_fees(),
                  "--collect-arrivals": lambda: collect_arrivals(),
                  "--collect-fill-census": lambda: collect_fill_census()}
    for flag, fn in collectors.items():
        if flag in sys.argv:
            fn()
            return
    print("=" * 74)
    print("THE VALUE OF THE MARGINAL SOLVER -- full paper pipeline (results/paper.json)")
    print("=" * 74)
    s = run_paper()
    print(f"\norders {s['n_orders']}, solvers {s['n_solvers_ever']}, months {len(s['months'])}")
    print("participation:", {q: (v["mean_bidders"], v["share_solo"], v["median_gap_bps"]) for q, v in s["participation"].items()})
    print("pooled shortfall medians:", {m: v["median_bps"] for m, v in s["shortfall_pooled"].items() if m != "n"})
    print("liquidity quintile S(1) medians:", {q: c["1"]["median_bps"] for q, c in s["shortfall_liquidity_quintiles"].items()})
    print(f"pivotal top2 {100*s['pivotal_top2_share']:.1f}%  wedge {100*s['wedge']['share_winner_not_best']:.1f}%")
    print(f"batch premium {s['batched_premium_bps_median_matched']} bps, cow {s['cow_premium_bps_median_matched']} bps")
    c = s["cap_exposure_test"].get("cap", {})
    print(f"cap test: binding share {c.get('share_binding_pre_orders')}, gap DiD impact "
          f"{c.get('did_gapw', {}).get('impact')} (se {c.get('did_gapw', {}).get('se_impact')})")
    e = s["exit_study"]
    print(f"exit study: bidders {e['did_bidders']['impact']} (se {e['did_bidders']['se_impact']}), "
          f"pi {e['did_piw']['impact']} (se {e['did_piw']['se_impact']}), "
          f"gap {e['did_gapw']['impact']} (se {e['did_gapw']['se_impact']})")
    print("  -> results/paper.json + figures/fig_shortfall|liquidity|pivotal|cip74|exitstudy.pdf")


if __name__ == "__main__":
    main()
