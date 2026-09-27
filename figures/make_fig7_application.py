"""Figure 7: application to two observed mega-heatwaves (ERA5-Land / ERA5).

Panels (a),(b): observed temperature and root-zone soil moisture through each
event, with the soil-moisture climatology for the same calendar days.
Panel (c): attributed risk ratio versus threshold, adjoint and direct.
Panel (d): relative shortfall of the first-order adjoint against the direct
counterfactual, versus base rate, with the testbed band of Sec. VI.
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#0072B2", "#D55E00", "#009E73", "#7A5195"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
W1, W2 = 3.375, 7.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "figure.dpi": 200, "savefig.pad_inches": 0.02,
})

def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

def panel(ax, letter, dx=-0.16):
    ax.text(dx, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")

A = "../application/results"          # sweep_all.json
DATA = "../data"                       # site.csv, site_2010.csv from data/download_era5_sites_arco_v2.py
FIG = "."
os.makedirs(FIG, exist_ok=True)
sweep = json.load(open(f"{A}/sweep_all.json"))

SITES = {
    # w0,w1 are the ACTUAL attribution window: t0 to the hour of the observed
    # peak, which attribute_event.run() truncates the search window to.
    "trappes":  dict(csv="site.csv",      year=2003,
                     w0="2003-08-01 00:00", w1="2003-08-06 15:00",
                     label="Trappes, 2003", u_mark=35),
    "voronezh": dict(csv="site_2010.csv", year=2010,
                     w0="2010-08-01 00:00", w1="2010-08-05 12:00",
                     label="Voronezh, 2010", u_mark=37),
}

fig = plt.figure(figsize=(W2, 4.9), layout="constrained")
fig.get_layout_engine().set(w_pad=0.055, h_pad=0.04, wspace=0.03)
gs = fig.add_gridspec(2, 6)
axs = np.empty((2, 2), dtype=object)
axs[0, 0] = fig.add_subplot(gs[0, 0:3])
axs[0, 1] = fig.add_subplot(gs[0, 3:6])
ax_prob = fig.add_subplot(gs[1, 0:2])
ax_rr   = fig.add_subplot(gs[1, 2:4])
ax_sf   = fig.add_subplot(gs[1, 4:6])

# ---------------------------------------------- (a),(b) observed events
for k, (key, meta) in enumerate(SITES.items()):
    ax = axs[0, k]
    df = pd.read_csv(f"{DATA}/{meta['csv']}", parse_dates=["time"]).set_index("time")
    # plot a window that brackets the event so the drawdown is visible
    lo = pd.Timestamp(meta["w0"]) - pd.Timedelta(days=17)
    hi = pd.Timestamp(meta["w1"]) + pd.Timedelta(days=12)
    seg = df.loc[lo:hi]
    # same construction as attribute_event.py: day-of-year mean over all
    # baseline summers (the event year is not excluded there)
    clim = df.groupby(df.index.dayofyear)["SM"].mean()
    seg_clim = pd.Series(seg.index.dayofyear, index=seg.index).map(clim)

    ax.plot(seg.index, seg["T"], color=C[1], lw=0.5, alpha=0.85)
    ax.axhline(meta["u_mark"], color=INK, lw=0.7, ls=":")
    ax.text(seg.index[-1], meta["u_mark"] + 0.7, f"$u={meta['u_mark']}$",
            fontsize=6.5, color=INK, ha="right")
    ax.axvspan(pd.Timestamp(meta["w0"]), pd.Timestamp(meta["w1"]),
               color=GRID, alpha=0.55, zorder=0, lw=0)
    ax.plot([pd.Timestamp(meta["w1"])], [df.loc[meta["w1"], "T"]], "v",
            color=INK, ms=3.2, clip_on=False)
    if k == 0:
        ax.set_ylabel("T  (°C)", color=C[1])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", colors=C[1])
    ax.set_ylim(5, 41)

    ax2 = ax.twinx()
    ax2.plot(seg.index, seg["SM"], color=C[0], lw=1.1)
    ax2.plot(seg.index, seg_clim.values, color=C[0], lw=0.9, ls="--", alpha=0.75)
    if k == 1:
        ax2.set_ylabel("SM  (m$^3$m$^{-3}$)", color=C[0])
    else:
        ax2.tick_params(axis="y", labelright=False)
    ax2.tick_params(axis="y", colors=C[0])
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0.12, 0.40)
    ax.spines["top"].set_visible(False)
    ax.set_title(meta["label"], fontsize=8, color=INK, pad=3)
    ax.set_xticks(pd.date_range(lo.ceil("D"), hi, freq="10D"))
    ax.set_xticklabels([d.strftime("%d %b") for d in
                        pd.date_range(lo.ceil("D"), hi, freq="10D")])
    panel(ax, "ab"[k], dx=-0.17)

axs[0, 0].text(0.025, 0.10, "SM observed", transform=axs[0, 0].transAxes,
               fontsize=6.2, color=C[0])
axs[0, 0].text(0.025, 0.02, "SM climatology (dashed)", transform=axs[0, 0].transAxes,
               fontsize=6.2, color=C[0], alpha=0.85)

# ---------------------------------------------- (c) factual vs counterfactual P
ax = ax_prob
for key, meta, c in (("trappes", SITES["trappes"], C[0]),
                     ("voronezh", SITES["voronezh"], C[1])):
    d = sweep[key]
    u = [r["u"] for r in d]
    ax.plot(u, [r["P_base"] for r in d], "-o", color=c, ms=2.6)
    ax.plot(u, [r["clim"]["P_cf"] for r in d], "--o", color=c, ms=2.6,
            markerfacecolor="white", markeredgewidth=0.7, alpha=0.9)
ax.set_yscale("log")
ax.set_xlabel("threshold $u$  (°C)")
ax.set_ylabel("$P_u$")
ax.text(0.04, 0.16, "observed soil (solid)", transform=ax.transAxes, fontsize=6.0)
ax.text(0.04, 0.06, "climatological soil (dashed)", transform=ax.transAxes,
        fontsize=6.0)
tidy(ax); panel(ax, "c", dx=-0.30)

# ---------------------------------------------- (d) risk ratio vs threshold
ax = ax_rr
for key, meta, c in (("trappes", SITES["trappes"], C[0]),
                     ("voronezh", SITES["voronezh"], C[1])):
    d = sweep[key]
    u = [r["u"] for r in d]
    ax.plot(u, [r["clim"]["risk_ratio_direct"] for r in d], "-o", color=c,
            ms=2.6, label=f"{meta['label']}, direct")
    ax.plot(u, [r["clim"]["risk_ratio_adjoint"] for r in d], "--s", color=c,
            ms=2.6, alpha=0.75, label=f"{meta['label']}, adjoint")
ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
ax.set_xlabel("threshold $u$  (°C)")
ax.set_ylabel("risk ratio")
tidy(ax); panel(ax, "d", dx=-0.30)
ax.legend(fontsize=5.6, loc="upper left", ncol=1, handlelength=1.6,
          labelspacing=0.25)

# ---------------------------------------------- (e) shortfall vs base rate
ax = ax_sf
for key, meta, c in (("trappes", SITES["trappes"], C[0]),
                     ("voronezh", SITES["voronezh"], C[1])):
    d = sweep[key]
    p = np.array([r["P_base"] for r in d])
    adj = np.array([r["clim"]["dlogP_adjoint"] for r in d])
    dir_ = np.array([r["clim"]["dlogP_direct"] for r in d])
    short = 100 * (1 - adj / dir_)          # % by which first order falls short
    # where the effect itself is near zero the ratio is noise-dominated
    res = np.abs(dir_) >= 0.05
    ax.plot(p, short, "-", color=c, lw=1.3, label=meta["label"])
    ax.plot(p[res], short[res], "o", color=c, ms=2.8)
    ax.plot(p[~res], short[~res], "o", color=c, ms=2.8,
            markerfacecolor="white", markeredgewidth=0.8)
# testbed band: first-order shortfall across the five thresholds of testbed/adjoint_accuracy.py
_ad = json.load(open("../testbed/adjoint.json"))
_r = [v["ratio_mean"] for v in _ad["replicates"].values()]
lo_b, hi_b = 100 * (1 - max(_r)), 100 * (1 - min(_r))
ax.axhspan(lo_b, hi_b, color=C[2], alpha=0.16, lw=0, zorder=0)
ax.text(0.03, 0.5 * (lo_b + hi_b), "testbed, Sec. VI", fontsize=6.2, color=C[2],
        va="center", transform=ax.get_yaxis_transform())
ax.set_xscale("log")
ax.set_xlabel("base rate $P_u$")
ax.set_ylabel("shortfall  (%)")
ax.set_ylim(-6, 38)
tidy(ax); panel(ax, "e", dx=-0.30)
ax.legend(fontsize=6.0, loc="upper right")

fig.savefig(f"{FIG}/fig7_application.pdf", bbox_inches="tight")
print("wrote", f"{FIG}/fig7_application.pdf")

# numbers quoted in the text
for key in SITES:
    d = sweep[key]
    p = np.array([r["P_base"] for r in d])
    adj = np.array([r["clim"]["dlogP_adjoint"] for r in d])
    dir_ = np.array([r["clim"]["dlogP_direct"] for r in d])
    print(f"\n{key}: SM0_vs_clim={d[0]['SM0_vs_clim']:.4f}  "
          f"tau_S={d[0]['tau_S_days']:.1f} d  tau_T={d[0]['tau_T_days']:.2f} d  "
          f"r2_T={d[0]['r2_T']:.3f}")
    print("  u      P_base   P_cf    RR_adj  RR_dir  shortfall%")
    for r, s in zip(d, 100 * (1 - adj / dir_)):
        print(f"  {r['u']:>4.0f}  {r['P_base']:7.4f} {r['clim']['P_cf']:7.4f}  "
              f"{r['clim']['risk_ratio_adjoint']:6.3f} "
              f"{r['clim']['risk_ratio_direct']:6.3f}  {s:6.1f}")
