"""Recover the standardized capacity shape G from the bid data and fit parametric
families to it. Pools within-order standardized offers (n_i>=8) and compares
candidate distributions by AIC and Kolmogorov-Smirnov, then reports the implied
standardized marginal-value curve d_G(m)=E[max_m Z]-E[max_{m-1} Z] for the best fits.
Standalone; reads the USDC backup. Writes figures/fig_shape.pdf and prints a table."""
import gzip, json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")

# ---- 1. recover pooled standardized offers -------------------------------
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data/pairs/competition_raw.jsonl.gz")
if not os.path.exists(SRC):
    SRC = os.path.join(HERE, "data/pairs_usdc/competition.jsonl.gz")
print("shape source:", SRC)
NMIN, N_AUC = 8, 400000

def _sols(rec):                                          # raw OR extracted schema
    for s in rec.get("solutions", rec.get("sols", [])):
        sv = s.get("solverAddress") or s.get("solver")
        for o in s.get("orders", []):
            sa = o.get("sellAmount", o.get("sa"))
            ba = o.get("buyAmount", o.get("ba"))
            oid = o.get("id")
            if sa and ba and oid:
                yield sv, oid, sa, ba

z = []
with gzip.open(SRC, "rt") as f:
    for li, line in enumerate(f):
        if li >= N_AUC:
            break
        rec = json.loads(line)
        per = {}
        for sv, oid, sa, ba in _sols(rec):
            try:
                sa, ba = float(sa), float(ba)
                if sa <= 0 or ba <= 0:
                    continue
                r = ba / sa
            except Exception:
                continue
            d = per.setdefault(oid, {})
            if r > d.get(sv, 0):
                d[sv] = r
        for d in per.values():
            rs = np.array(sorted(d.values(), reverse=True))
            if len(rs) < NMIN:
                continue
            m, sd = rs.mean(), rs.std(ddof=1)
            if sd > 0:
                z.extend(((rs - m) / sd).tolist())
z = np.array(z)
rng = np.random.default_rng(0)
if len(z) > 400000:
    z = rng.choice(z, 400000, replace=False)
print(f"pooled standardized offers: {len(z)}")
print(f"skew {stats.skew(z):.3f}  excess-kurt {stats.kurtosis(z):.3f}")

# ---- 2. fit candidate families -------------------------------------------
cands = {
    "normal":       stats.norm,
    "logistic":     stats.logistic,
    "laplace":      stats.laplace,
    "skew-normal":  stats.skewnorm,
    "GEV":          stats.genextreme,
    "Gumbel (min)": stats.gumbel_l,
}
rows = []
fitted = {}
for name, dist in cands.items():
    par = dist.fit(z)
    ll = np.sum(dist.logpdf(z, *par))
    k = len(par)
    aic = 2 * k - 2 * ll
    ks = stats.kstest(z, dist.cdf, args=par).statistic
    rows.append((name, aic, ks, par))
    fitted[name] = (dist, par)
rows.sort(key=lambda r: r[1])
best = rows[0][0]
print("\nfamily          AIC(rel)     KS       params")
a0 = rows[0][1]
for name, aic, ks, par in rows:
    ps = ", ".join(f"{p:.3f}" for p in par)
    print(f"  {name:<13} {aic-a0:>9.0f}   {ks:.4f}   [{ps}]")
print(f"\nbest fit: {best}")

# ---- 3. implied standardized marginal-value curve d_G(m) -----------------
def dG_from_samples(dist, par, M=8, S=400000):
    x = dist.rvs(*par, size=(S, M), random_state=1)
    mx = np.maximum.accumulate(np.sort(x, axis=1)[:, ::-1] * 0 + np.maximum.accumulate(x, axis=1), axis=1)
    Vm = np.maximum.accumulate(x, axis=1).mean(axis=0)   # E[max of first m]
    return np.diff(Vm, prepend=Vm[0] - (Vm[1] - Vm[0]))  # d_G(m), m=1..M (d_G(1)~level)

M = 6
curves = {}
for name in ("normal", best if best != "normal" else "skew-normal"):
    dist, par = fitted[name]
    x = dist.rvs(*par, size=(600000, M), random_state=2)
    Vm = np.maximum.accumulate(x, axis=1).mean(axis=0)
    dG = np.diff(Vm)                                      # d_G(2..M)
    curves[name] = dG / dG[0]                             # normalize to d_G(2)=1
print("\nimplied marginal-value curve d_G(m)/d_G(2), m=2..6:")
for name, c in curves.items():
    print(f"  {name:<13} " + "  ".join(f"{v:.3f}" for v in c))

# empirical (paper) shortfall means -> normalized marginal shape for comparison
sf = {1: 23.75, 2: 10.05, 3: 5.80, 4: 3.57, 5: 2.12}     # mean Delta_i(m), bps (results/paper.json)
demp = {m: sf[m-1] - sf[m] for m in range(2, 6)}          # empirical delta(m)
base = demp[2]
print("  empirical      " + "  ".join(f"{demp[m]/base:.3f}" for m in range(2, 6)))

# ---- 4. figure: recovered shape (QQ / density) + fitted best -------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42})
fig, (axd, axq) = plt.subplots(1, 2, figsize=(7.2, 3.0))
xs = np.linspace(z.min(), np.quantile(z, 0.999), 400)
axd.hist(z, bins=140, density=True, color="#c9d5e5", edgecolor="none")
for name, col in [("normal", "#999999"), (best, "#1f4e79")]:
    dist, par = fitted[name]
    axd.plot(xs, dist.pdf(xs, *par), color=col, lw=1.4, label=name)
axd.set_xlim(np.quantile(z, 0.002), np.quantile(z, 0.999))
axd.set_xlabel("standardized offer $z=(r-\\mu_i)/\\sigma_i$")
axd.set_ylabel("density")
axd.set_title("(a) recovered shape $G$ vs fits")
axd.legend(frameon=False)
# QQ vs normal and vs best
probs = np.linspace(0.005, 0.995, 200)
eq = np.quantile(z, probs)
axq.plot(stats.norm.ppf(probs), eq, color="#999999", lw=1.3, label="vs normal")
d, p = fitted[best]
axq.plot(d.ppf(probs, *p), eq, color="#1f4e79", lw=1.3, label=f"vs {best}")
lim = [min(eq[0], -3), max(eq[-1], 3)]
axq.plot(lim, lim, ":", color="#444444", lw=0.8)
axq.set_xlabel("model quantile")
axq.set_ylabel("empirical quantile")
axq.set_title("(b) QQ: normal vs best fit")
axq.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shape.pdf"))
print("\nwrote figures/fig_shape.pdf")
