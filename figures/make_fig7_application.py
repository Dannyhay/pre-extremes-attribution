"""Figure 7: attribution in two observed heatwaves (ERA5-Land / ERA5).

(a),(b) observed 2-m temperature and root-zone soil moisture through each event, with
        the day-of-year soil-moisture climatology of the baseline summers (event year excluded).
(c)     factual and counterfactual exceedance probability at the peak hour against threshold.
(d)     risk ratio against threshold: direct counterfactual with its 90% year-block bootstrap
        band, first-order adjoint, and (2010) the flat-extrapolation alternative.
(e)     first-order shortfall against base rate, with the Gaussian mean-shift prediction.
Reads application/results/attribution_*.json and bootstrap_*.json and data/site*.csv.
Run from figures/.
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

C = ["#0072B2", "#D55E00", "#009E73", "#7A5195"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
W2 = 7.0
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "lines.linewidth": 1.3, "legend.frameon": False,
    "figure.dpi": 200, "savefig.pad_inches": 0.02,
})
A = os.environ.get("APP_RESULTS", "../application/results")
DATA = os.environ.get("APP_DATA", "../data")


def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def panel(ax, letter, dx=-0.16):
    ax.text(dx, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


SITES = {"trappes": dict(csv="site.csv", year=2003, label="Trappes, 2003", u_mark=37, c=C[2]),
         "voronezh": dict(csv="site_2010.csv", year=2010, label="Voronezh, 2010", u_mark=38, c=C[3])}
res = {k: json.load(open(f"{A}/attribution_{k}_linear.json")) for k in SITES}
flat = {k: json.load(open(f"{A}/attribution_{k}_flat.json")) for k in SITES}
boot = {k: json.load(open(f"{A}/bootstrap_{k}_linear.json")) for k in SITES}

fig = plt.figure(figsize=(W2, 4.9), layout="constrained")
fig.get_layout_engine().set(w_pad=0.055, h_pad=0.04, wspace=0.03)
gs = fig.add_gridspec(2, 6)
top = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
ax_p, ax_rr, ax_sf = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])

# ---------------------------------------------------------------- (a),(b)
for k, (key, meta) in enumerate(SITES.items()):
    ax = top[k]; r = res[key]
    df = pd.read_csv(f"{DATA}/{meta['csv']}", parse_dates=["time"]).set_index("time")
    w0, w1 = pd.Timestamp(r["t0"]), pd.Timestamp(r["t1"])
    lo, hi = w0 - pd.Timedelta(days=17), w1 + pd.Timedelta(days=12)
    seg = df.loc[lo:hi]
    base = df[df.index.month.isin([6, 7, 8]) & (df.index.year != meta["year"])]
    clim = base.groupby(base.index.dayofyear)["SM"].mean()
    seg_clim = pd.Series(seg.index.dayofyear, index=seg.index).map(clim)
    ax.plot(seg.index, seg["T"], color=C[1], lw=0.5, alpha=0.85)
    ax.axhline(meta["u_mark"], color=INK, lw=0.7, ls=":")
    ax.text(seg.index[-1], meta["u_mark"] + 0.7, f"$u={meta['u_mark']}$", fontsize=6.5, ha="right")
    ax.axvspan(w0, w1, color=GRID, alpha=0.55, zorder=0, lw=0)
    ax.plot([w1], [df.loc[w1, "T"]], "v", color=INK, ms=3.2, clip_on=False)
    if k == 0:
        ax.set_ylabel("$T$  (°C)", color=C[1])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", colors=C[1]); ax.set_ylim(5, 41)
    ax2 = ax.twinx()
    ax2.plot(seg.index, seg["SM"], color=C[0], lw=1.1)
    ax2.plot(seg.index, seg_clim.values, color=C[0], lw=0.9, ls="--", alpha=0.75)
    if k == 1:
        ax2.set_ylabel("$S$  (m$^3$m$^{-3}$)", color=C[0])
    else:
        ax2.tick_params(axis="y", labelright=False)
    ax2.tick_params(axis="y", colors=C[0]); ax2.spines["top"].set_visible(False); ax2.set_ylim(0.12, 0.40)
    ax.spines["top"].set_visible(False)
    ax.set_title(meta["label"], fontsize=8, pad=3)
    ticks = pd.date_range(lo.ceil("D"), hi, freq="10D")
    ax.set_xticks(ticks); ax.set_xticklabels([d.strftime("%d %b") for d in ticks])
    panel(ax, "ab"[k], dx=-0.17)
top[0].text(0.025, 0.10, "soil moisture", transform=top[0].transAxes, fontsize=6.2, color=C[0])
top[0].text(0.025, 0.02, "climatology (dashed)", transform=top[0].transAxes, fontsize=6.2, color=C[0], alpha=0.85)

# ---------------------------------------------------------------- (c)
for key, meta in SITES.items():
    rows = res[key]["thresholds"]; u = [r["u"] for r in rows]
    ax_p.plot(u, [r["P_rb"] for r in rows], "-o", color=meta["c"], ms=2.4)
    ax_p.plot(u, [r["P_cf_rb"] for r in rows], "--o", color=meta["c"], ms=2.4, mfc="white", mew=0.7)
ax_p.set_yscale("log"); ax_p.set_ylim(5e-4, 1.5)
ax_p.set_xlabel("threshold $u$  (°C)"); ax_p.set_ylabel("$P_u$")
ax_p.text(0.04, 0.14, "factual (solid)", transform=ax_p.transAxes, fontsize=6.0)
ax_p.text(0.04, 0.05, "climatological soil (dashed)", transform=ax_p.transAxes, fontsize=6.0)
tidy(ax_p); panel(ax_p, "c", dx=-0.30)

# ---------------------------------------------------------------- (d)
for key, meta in SITES.items():
    rows = res[key]["thresholds"]; u = np.array([r["u"] for r in rows])
    b = boot[key]
    rr_b = np.array([[rep["rr"][str(x)] for x in u] for rep in b[1:]])
    ax_rr.fill_between(u, np.percentile(rr_b, 5, 0), np.percentile(rr_b, 95, 0), color=meta["c"], alpha=0.15, lw=0)
    ax_rr.plot(u, [r["RR_rb"] for r in rows], "-o", color=meta["c"], ms=2.4, label=f"{meta['label']}, direct")
    ax_rr.plot(u, [r["RR_adjoint"] for r in rows], "--s", color=meta["c"], ms=2.2, alpha=0.8,
               label=f"{meta['label']}, first order")
fr = flat["voronezh"]["thresholds"]
ax_rr.plot([r["u"] for r in fr], [r["RR_rb"] for r in fr], ":", color=C[3], lw=1.1, label="Voronezh, flat extrapolation")
ax_rr.set_yscale("log"); ax_rr.axhline(1.0, color=MUTED, lw=0.7, ls=":")
ax_rr.set_xlabel("threshold $u$  (°C)"); ax_rr.set_ylabel("risk ratio $P_u/P_u^{\\rm cf}$")
ax_rr.legend(fontsize=5.4, loc="upper left", handlelength=1.8, labelspacing=0.25)
tidy(ax_rr); panel(ax_rr, "d", dx=-0.30)

# ---------------------------------------------------------------- (e)
for key, meta in SITES.items():
    r = res[key]; rows = r["thresholds"]; t = r["terminal"]
    p = np.array([x["P_rb"] for x in rows]); sh = 100 * np.array([x["shortfall_rb"] for x in rows])
    ax_sf.plot(p, sh, "o", color=meta["c"], ms=3.0, label=meta["label"])
    z = np.linspace(-3.5, 3.2, 300); d = t["shift"] / t["sd_total"]
    ex = norm.logsf(z) - norm.logsf(z + d); fo = norm.pdf(z) / norm.sf(z) * d
    ax_sf.plot(norm.sf(z), 100 * (1 - fo / ex), "-", color=meta["c"], lw=0.9, alpha=0.8)
ax_sf.set_xscale("log"); ax_sf.set_xlim(8e-3, 1.02); ax_sf.set_ylim(0, 100)
ax_sf.set_xlabel("base rate $P_u$"); ax_sf.set_ylabel("first-order shortfall  (%)")
ax_sf.text(0.03, 0.93, "curves: Gaussian mean shift", transform=ax_sf.transAxes, fontsize=6.0, color=MUTED)
ax_sf.legend(fontsize=6.0, loc="center left")
tidy(ax_sf); panel(ax_sf, "e", dx=-0.30)

fig.savefig("fig7_application.pdf", bbox_inches="tight")
print("wrote fig7_application.pdf")
