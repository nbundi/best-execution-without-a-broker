"""Generate the compact reproducibility dataset from the raw census: the extracted
bid panel (competition.jsonl.gz) and compact order metadata (cohort_meta.jsonl.gz),
which together with base_fees.jsonl.gz and the completed amm_routes.jsonl reproduce
every result the pipeline computes. Streaming; memory-light. Writes into
data/repro/ so the 47 GB raw is untouched.
"""
import os, sys, gzip, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import code as C

SRC = os.path.join(C.DATA, "pairs")
OUT = os.path.join(C.DATA, "repro")
os.makedirs(OUT, exist_ok=True)

# --- 1. extracted bid panel (competition.jsonl.gz) --------------------------
raw = os.path.join(SRC, "competition_raw.jsonl.gz")
dst = os.path.join(OUT, "competition.jsonl.gz")
n = kept = 0
with gzip.open(dst, "wt") as g:
    for d in C._iter_jsonl_any(raw):
        n += 1
        rec, _u = C._extract_pair_auction(d)
        if rec:
            g.write(json.dumps(rec, separators=(",", ":")) + "\n")
            kept += 1
        if n % 200000 == 0:
            print(f"  competition: {n} auctions scanned, {kept} written", flush=True)
print(f"[done] competition.jsonl.gz: {kept}/{n} auctions -> {os.path.getsize(dst)/1e9:.2f} GB", flush=True)

# --- 2. compact order metadata (cohort_meta.jsonl.gz) -----------------------
dstm = os.path.join(OUT, "cohort_meta.jsonl.gz")
seen = set(); mkept = 0
with gzip.open(dstm, "wt") as g:
    for src in (os.path.join(SRC, "cohort_meta_raw.jsonl.gz"),
                C._pairs_path(SRC, "cohort_meta")):
        if not os.path.exists(src):
            continue
        for m in C._iter_jsonl_any(src):
            r = C._as_meta_row(m)
            u = r.get("uid")
            if u in seen:
                continue
            seen.add(u)
            g.write(json.dumps(r, separators=(",", ":")) + "\n")
            mkept += 1
print(f"[done] cohort_meta.jsonl.gz: {mkept} orders -> {os.path.getsize(dstm)/1e6:.0f} MB", flush=True)

# --- 3. base_fees passthrough -----------------------------------------------
import shutil
bf = C._pairs_path(SRC, "base_fees")
if os.path.exists(bf):
    shutil.copy(bf, os.path.join(OUT, "base_fees.jsonl.gz"))
    print("[done] base_fees.jsonl.gz copied", flush=True)
print("REPRO-DATASET-DONE", flush=True)
