"""Figure: the higher-order (finite response time) repair works in Smirnov's
setting and does not survive saturation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ftir_linear import ftir as ftir_exact, lkif_analytic

C = ["#0072B2", "#D55E00", "#009E73", "#7A5195"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "figure.dpi": 200, "savefig.pad_inches": 0.02, "mathtext.fontset": "cm",
})

def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

def panel(ax, s, dx=-0.18):
    ax.text(dx, 1.04, f"({s})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")

R = json.load(open("ftir_testbed.json"))
S = json.load(open("ftir_seeds.json"))

fig = plt.figure(figsize=(7.0, 2.35), layout="constrained")
fig.get_layout_engine().set(w_pad=0.06, h_pad=0.03)
ax = fig.subplots(1, 3)

# ---- (a) Smirnov's linear system: the repair working --------------------
al, G = 1.0, 2.0
tg = np.linspace(0.02, 3.0, 150)
for byx, col, ls in ((0.0, C[0], "-"), (-1.0, C[1], (0, (4, 2)))):
    L, st = ftir_exact(al, al, 1.0, byx, G, G, tg)
    lk = lkif_analytic(1.0, st["Cxx"], st["Cxy"], st["Cyy"])
    ax[0].plot(tg, L, color=col, ls=ls,
               label=rf"$\beta_{{yx}}={byx:g}$,  LKIF$\,={lk:.2f}$")
ax[0].annotate("LKIF exactly zero,\nFTIR retains 50%", xy=(0.85, 0.055),
               xytext=(1.25, 0.082), fontsize=7, color=C[1],
               arrowprops=dict(arrowstyle="->", color=C[1], lw=0.8))
ax[0].set_xlabel(r"response time  $\alpha t$")
ax[0].set_ylabel(r"FTIR  $L^{(t)}_{Y\to X}$")
ax[0].set_title("Smirnov's linear system", fontsize=7.5, color=MUTED, pad=3)
ax[0].legend(loc="upper right", handlelength=1.8)
ax[0].set_ylim(0, 0.135)
tidy(ax[0]); panel(ax[0], "a")

# ---- (b) the testbed ----------------------------------------------------
cols = {"transitional": C[2], "wet": C[0], "very dry": C[1]}
for name in ("transitional", "wet", "very dry"):
    t = np.array(R[name]["ts_curve"]); L = np.abs(np.array(R[name]["L_curve"]))
    ax[1].plot(0.25 * t, np.maximum(L, 1e-7), "-o", color=cols[name],
               ms=2.6, label=name)
sd = np.std(np.abs(S["very dry"]), ddof=1)
mu = np.mean(np.abs(S["very dry"]))
ax[1].axhspan(max(mu - sd, 1e-7), mu + sd, color=C[1], alpha=0.16, lw=0)
ax[1].text(0.62, mu * 1.9, "seed spread", fontsize=6.5, color=C[1])
ax[1].set_yscale("log"); ax[1].set_ylim(2e-6, 3)
ax[1].set_xlabel(r"response time  $\alpha_T t$")
ax[1].set_ylabel(r"$|L^{(t)}_{U\to T}|$")
ax[1].set_title("TV-CSM testbed", fontsize=7.5, color=MUTED, pad=3)
ax[1].legend(loc="lower right", handlelength=1.6)
tidy(ax[1]); panel(ax[1], "b")

# ---- (c) how every measure ranks very dry against transitional ----------
E = json.load(open("testbed_escort.json"))
A = json.load(open("auconi.json"))
al_grid = np.array(E["alphas"])
vd = np.array(E["regimes"]["very dry"]["T_alpha"])
tr = np.array(E["regimes"]["transitional"]["T_alpha"])
i1 = int(np.argmin(np.abs(al_grid - 1)))
ratios = [
    ("LKIF", vd[i1] / tr[i1], C[0]),
    ("R\u00e9nyi", float((vd / tr).max()), C[0]),
    ("FTIR", mu / abs(R["transitional"]["peak"]), C[3]),
    ("Auconi", A["retention"]["very dry"], C[3]),
    ("level", E["level_ratio_dry_over_trans"], C[1]),
]
xs = np.arange(len(ratios))
ax[2].bar(xs, [r[1] for r in ratios], 0.64, color=[r[2] for r in ratios])
ax[2].axhline(1.0, color=INK, lw=1.0)
ax[2].text(-0.45, 1.35, "parity", fontsize=6.5, color=INK, ha="left")
ax[2].axhline(0.50, color=MUTED, lw=0.9, ls=(0, (2, 2)))
ax[2].text(1.6, 0.20, "Smirnov's repair\nin his own setting", fontsize=5.8,
           color=MUTED, ha="center", va="top")
for x, (_, v, _) in zip(xs, ratios):
    ax[2].text(x, v * (0.42 if v > 1 else 1.75),
               f"{v:.1e}".replace("e-0", "e\u2212") if v < 0.01 else f"{v:.1f}",
               ha="center", fontsize=6.4,
               color="white" if v > 1 else INK,
               fontweight="bold" if v > 1 else "normal")
ax[2].set_yscale("log"); ax[2].set_ylim(4e-5, 40)
ax[2].set_xticks(xs); ax[2].set_xticklabels([r[0] for r in ratios], fontsize=7)
ax[2].set_ylabel("very dry / transitional")
ax[2].set_title("every repair ranks it the same way", fontsize=7.5,
                color=MUTED, pad=3)
tidy(ax[2]); panel(ax[2], "c")

fig.savefig("fig_ftir.pdf"); fig.savefig("fig_ftir.png")
print("ratios:", [(r[0].replace(chr(10)," "), float(f"{r[1]:.3g}")) for r in ratios])
print("wrote fig_ftir.{pdf,png}")
