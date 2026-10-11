"""Illustrative diagram of the model's two core objects, the expected-execution
curve V_i(m) and the marginal-value curve delta_i(m), computed exactly from the
model of Section 3 (expected maxima of i.i.d. capacities) at two dispersion levels.
Schematic: the location-scale shape is the normal; the two scales sigma are chosen
so the magnitudes resemble the paper's deep and thin liquidity strata. Standalone;
does not touch the collection pipeline. Writes figures/fig_model.pdf."""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(HERE, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.titlesize": 8, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def e_max_normal(m, grid=None):
    """E[max of m i.i.d. N(0,1)] = m * int x phi(x) Phi(x)^{m-1} dx (numeric)."""
    if grid is None:
        grid = np.linspace(-7, 7, 20001)
    phi = np.exp(-grid**2 / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(grid / math.sqrt(2)))
    return m * np.trapz(grid * phi * Phi**(m - 1), grid)

M = 12
ms = np.arange(1, M + 1)
e = np.array([e_max_normal(m) for m in ms])            # standardized expected maxima
# scales chosen so cumulative value V(M)-V(1) ~ deep 14 bps, thin 96 bps (paper magnitudes)
BPS = 1e4
sig = {"deep": 14.0 / (BPS * (e[-1] - e[0])) if e[-1] != e[0] else 0,
       "thin": 96.0 / (BPS * (e[-1] - e[0]))}
COL = {"deep": "#1f4e79", "thin": "#a6611a"}
LAB = {"deep": r"deep pair (small $\sigma_i$)", "thin": r"thin pair (large $\sigma_i$)"}

fig, (axV, axd) = plt.subplots(1, 2, figsize=(7.2, 3.1))

# ---- Panel (a): V_i(m), expected execution above a lone bidder (bps) ----
for k in ("thin", "deep"):
    V = sig[k] * (e - e[0]) * BPS                       # V_i(m) - V_i(1), in bps
    axV.plot(ms, V, "o-", color=COL[k], lw=1.4, ms=3.5, label=LAB[k])
axV.set_xlabel(r"competitors admitted, $m$")
axV.set_ylabel(r"$V_i(m)-V_i(1)$  (bps of execution)")
axV.set_title(r"(a) expected execution  $V_i(m)=\mathbb{E}[\max\{R_{i1},\dots,R_{im}\}]$")
axV.set_xticks(ms)
axV.legend(frameon=False, loc="center right")
axV.annotate("concave: each added\ncompetitor adds less",
             xy=(8, sig["thin"] * (e[7] - e[0]) * BPS), xytext=(4.6, 26),
             color="#555555", fontsize=7,
             arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

# ---- Panel (b): delta_i(m), marginal value of the m-th competitor (bps) ----
for k in ("thin", "deep"):
    d = sig[k] * (e - np.concatenate(([e[0]], e[:-1]))) * BPS   # delta_i(m), bps
    d[0] = np.nan                                       # delta defined for m>=2
    axd.plot(ms, d, "o-", color=COL[k], lw=1.4, ms=3.5, label=LAB[k])
axd.set_xlabel(r"competitor index, $m$")
axd.set_ylabel(r"$\delta_i(m)=V_i(m)-V_i(m{-}1)$  (bps)")
axd.set_title(r"(b) marginal value  $\delta_i(m)$")
axd.set_xticks(ms)
axd.legend(frameon=False, loc="upper right")
axd.annotate("front-loaded; scales\n" r"with dispersion $\sigma_i$",
             xy=(3, sig["thin"] * (e[2] - e[1]) * BPS), xytext=(4.8, 20),
             color="#555555", fontsize=7,
             arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

fig.tight_layout()
out = os.path.join(FIGURES, "fig_model.pdf")
fig.savefig(out)
print("wrote", out)
# report the plotted magnitudes for the caption/text
for k in ("deep", "thin"):
    d2 = sig[k] * (e[1] - e[0]) * BPS
    d5 = sig[k] * (e[4] - e[3]) * BPS
    print(f"  {k}: delta(2)={d2:.1f} bps, delta(5)={d5:.2f} bps, cumV(12)-V(1)={sig[k]*(e[-1]-e[0])*BPS:.1f} bps")
