"""Figures for the escort result. Styled to match the manuscript figures."""
import json
import numpy as np
import matplotlib
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
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "figure.dpi": 200, "savefig.pad_inches": 0.02, "mathtext.fontset": "cm",
})

def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

def panel(ax, s, dx=-0.17):
    ax.text(dx, 1.04, f"({s})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")

E = json.load(open("testbed_escort.json"))
S = json.load(open("stress_results.json"))
A = np.array(E["alphas"]); As = np.array(S["alphas"])
REG = ["wet", "transitional", "very dry"]
RC = {"wet": C[0], "transitional": C[2], "very dry": C[1]}

# ===================================================== FIGURE 1: the theory
fig = plt.figure(figsize=(W2, 2.35), layout="constrained")
fig.get_layout_engine().set(w_pad=0.05, h_pad=0.03)
ax = fig.subplots(1, 3)

# (a) escort densities in the very dry regime
r = E["regimes"]["very dry"]
y = np.array(r["y"]); rho = np.array(r["rho"])
ax[0].plot(y, rho / rho.max(), color=INK, lw=1.6, label=r"$\rho$  ($\alpha=1$)")
for a, ls, col in [(0.2, (0, (4, 2)), C[0]), (5.0, (0, (1, 1.5)), C[3])]:
    w = np.exp(a * np.log(np.maximum(rho, 1e-300)))
    w /= np.trapezoid(w, y)
    ax[0].plot(y, w / w.max(), color=col, lw=1.4, ls=ls,
               label=rf"$\rho_\alpha$, $\alpha={a:g}$")
u = r["mean_T"] + 2.2 * r["sd_T"]
ax[0].axvline(u, color=C[1], lw=1.2)
ax[0].annotate("threshold $u$", xy=(u, 0.30), xytext=(u + 0.30, 0.62),
               fontsize=7, color=C[1], ha="left",
               arrowprops=dict(arrowstyle="->", color=C[1], lw=0.8))
ax[0].set_xlim(r["mean_T"] - 4.5, r["mean_T"] + 4.5); ax[0].set_ylim(0, 1.52)
ax[0].set_xlabel("$y$"); ax[0].set_ylabel("weight (scaled)")
ax[0].legend(loc="upper left", handlelength=1.5, ncol=1, borderpad=0.1, labelspacing=0.25)
tidy(ax[0], grid=None); panel(ax[0], "a")

# (b) level vs slope in the very dry regime
j = np.array(r["j"]); dj = np.array(r["dj"])
m = (y > r["mean_T"] - 4.5) & (y < r["mean_T"] + 4.5)
ax[1].plot(y[m], j[m], color=C[1], lw=1.8)
ax[1].set_ylim(0, 3.4); ax[1].set_xlabel("$y$")
ax[1].set_ylabel(r"level  $j(y)$", color=C[1])
ax[1].tick_params(axis="y", colors=C[1])
a2 = ax[1].twinx()
a2.plot(y[m], np.abs(dj[m]), color=C[0], lw=1.8, ls=(0, (4, 2)))
a2.set_ylabel(r"slope  $|\partial_y j|$", color=C[0])
a2.tick_params(axis="y", colors=C[0]); a2.set_ylim(0, 3.4e-4)
a2.spines["top"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].text(0.5, 0.62, r"$j\simeq 2.99$", transform=ax[1].transAxes,
           fontsize=8, color=C[1], ha="center")
ax[1].text(0.5, 0.12, r"$|\partial_y j|\lesssim 10^{-4}$", transform=ax[1].transAxes,
           fontsize=8, color=C[0], ha="center")
panel(ax[1], "b")

# (c) the ranking, slope-based family vs level
vd = np.array(E["regimes"]["very dry"]["T_alpha"])
tr = np.array(E["regimes"]["transitional"]["T_alpha"])
ax[2].axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
ax[2].axhline(E["level_ratio_dry_over_trans"], color=C[1], lw=1.8)
ax[2].text(0.35, E["level_ratio_dry_over_trans"] * 1.45,
           r"by LEVEL: $\max|J_{\rm src}|$", fontsize=7, color=C[1])
ax[2].plot(A, vd / tr, "-o", color=C[0], ms=3, mfc=C[0])
ax[2].text(0.6, 6e-4, r"by SLOPE: $T^{(\alpha)}$, every $\alpha$",
           fontsize=7, color=C[0])
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_ylim(1e-4, 40)
ax[2].set_xlabel("R\u00e9nyi index $\\alpha$")
ax[2].set_ylabel("very dry / transitional")
tidy(ax[2]); panel(ax[2], "c")
fig.savefig("fig_escort_theory.pdf"); fig.savefig("fig_escort_theory.png")
plt.close(fig)

# ============================================ FIGURE 2: the estimator tests
fig = plt.figure(figsize=(W2, 4.5), layout="constrained")
fig.get_layout_engine().set(w_pad=0.06, h_pad=0.12)
gs = fig.add_gridspec(2, 2)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
       fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# (a) raw RCMI on the testbed
for n in REG:
    axs[0].plot(As, S["testbed_rcmi"][n]["I_obs"], "-o", color=RC[n], ms=2.6, label=n)
axs[0].set_xscale("log"); axs[0].set_yscale("symlog", linthresh=1e-4)
axs[0].set_xlabel(r"$\alpha$"); axs[0].set_ylabel("RCMI, raw value")
axs[0].legend(loc="upper right", handlelength=1.6)
axs[0].set_title("raw values invert the ranking", fontsize=7.5, color=MUTED, pad=3)
tidy(axs[0]); panel(axs[0], "a")

# (b) surrogate z on the testbed -- the decisive panel
for n in REG:
    axs[1].plot(As, np.abs(S["testbed_rcmi"][n]["z"]) + 1e-2, "-o",
                color=RC[n], ms=2.6, label=n)
axs[1].axhline(2.0, color=C[1], lw=0.9, ls=(0, (2, 2)))
axs[1].text(0.115, 2.5, "significance", fontsize=6.5, color=C[1])
axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xlabel(r"$\alpha$"); axs[1].set_ylabel("|z| against surrogates")
axs[1].set_title("against its own null, nothing survives in the dry regime",
                 fontsize=7.5, color=MUTED, pad=3)
tidy(axs[1]); panel(axs[1], "b")

# (c) B1 vs B2: who detects the tail driver X4
meth = ["RCMI\n(best $\\alpha$)", "nonlinear\ncond.-mean"]
b1, b2 = S["variants"]["sign"], S["variants"]["random"]
def arr(v):
    return [np.abs(np.array(v["z"])[3]).max(), abs(v["nl_z"][3])]
x = np.arange(2); w = 0.34
axs[2].bar(x - w / 2, arr(b1), w, color=C[3], label="B1  mean response")
axs[2].bar(x + w / 2, arr(b2), w, color=C[1], label="B2  variance only")
axs[2].axhline(2.0, color=MUTED, lw=0.9, ls=(0, (2, 2)))
axs[2].text(1.44, 2.5, "significance", fontsize=6.5, color=MUTED, ha="right")
axs[2].set_yscale("log"); axs[2].set_ylim(0.8, 3000); axs[2].set_xlim(-0.6, 1.6)
axs[2].set_xticks(x); axs[2].set_xticklabels(meth, fontsize=7)
axs[2].set_ylabel("detection strength  |z|")
axs[2].legend(loc="upper right", handlelength=1.2)
axs[2].set_xlabel("linear LKIF on X4:  $p=%.3f$ (B1),   $p=%.3f$ (B2)"
                  % (b1["lkif_p"][3], b2["lkif_p"][3]), fontsize=7, color=MUTED)
axs[2].set_title("only RCMI sees a variance-only driver", fontsize=7.5,
                 color=MUTED, pad=3)
tidy(axs[2]); panel(axs[2], "c")

# (d) seed robustness of Experiment B as written
sr = S["seed_robustness"]
zz = [r["best_z"] for r in sr]
cols = [C[1] if abs(v) > 2 else MUTED for v in zz]
axs[3].bar(np.arange(len(zz)), zz, 0.62, color=cols)
for h in (2, -2):
    axs[3].axhline(h, color=C[1], lw=0.9, ls=(0, (2, 2)))
axs[3].set_xticks(np.arange(len(zz)))
axs[3].set_xticklabels([str(r["n_inj"]) for r in sr], fontsize=6.8)
axs[3].set_xlabel("injections realised (8 seeds)")
axs[3].set_ylabel("strongest z for X4, any $\\alpha$")
axs[3].set_title("Experiment B detects X4 in 2 of 8 seeds", fontsize=7.5,
                 color=MUTED, pad=3)
tidy(axs[3]); panel(axs[3], "d")

fig.savefig("fig_escort_estimator.pdf"); fig.savefig("fig_escort_estimator.png")
plt.close(fig)
print("wrote fig_escort_theory.{pdf,png} and fig_escort_estimator.{pdf,png}")
