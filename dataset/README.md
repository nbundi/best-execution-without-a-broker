# Census dataset

The full-census dataset used to produce every number in the paper, split into
`< 100 MB` parts so it fits in normal git (GitHub rejects files over 100 MB).

## Reassemble and extract

From the repository root:

```sh
cat dataset/census-data.tar.gz.part* | tar xzf -
# or:
sh dataset/unpack.sh
```

This restores the following into `data/` (the paths the analysis expects):

| File | Contents |
|---|---|
| `data/pairs/competition.jsonl.gz` | Settled-auction solver-competition census (winning + losing solutions, /USDC tier orders), 565,259 auctions |
| `data/pairs/cohort_meta.jsonl.gz` | Order lifecycle metadata (status, kind, class, executed amounts) |
| `data/pairs/base_fees.jsonl.gz` | Per-settlement-block EIP-1559 base fees (for net price improvement) |
| `data/amm_routes.jsonl` | Uniswap smart-order-router counterfactual quotes, block-pinned (464,880 reliable) |
| `data/fill_cohort/cohort.jsonl.gz` | Submission-cohort /USDC arrivals with marketability + fill status (1,853,859) |
| `data/pairs/manifest.json` | Census provenance |

Then reproduce the results:

```sh
python3 code.py --pairs --amm-pairs --full --multifactor --timeseries --fill-cohort
```

## Not included (regenerable)

Collection *intermediates and caches* are excluded to keep the release small; they
are not read by the analysis and can be regenerated with the `code.py` collectors
(see the top-level README): the arrival scan (`arrivals.jsonl.gz`), the non-/USDC
classification cache (`nonusdc.jsonl.gz`), and the RPC cache (`rpc_cache.json`).
`--timeseries` and `--multifactor` will re-fetch a small number of block timestamps
from an archival RPC if `rpc_cache.json` is absent.
