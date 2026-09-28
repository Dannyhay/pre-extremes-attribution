"""
Figures 1, 2, 4, 5 and 6 of the manuscript from the JSON outputs of
stationary_regimes.py, traversal.py, adjoint_accuracy.py and envelope.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import g, REGIMES, logistic

C = ["#0072B2", "#D55E00", "#009E73", "#7A5195", "#000000"]
INK, MUTED = "#1a1a1a", "#5c5c5c"
W1, W2 = 3.375, 7.0
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "lines.linewidth": 1.3,
    "legend.frameon": False, "figure.dpi": 200, "savefig.pad_inches": 0.02,
})


def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)


st = json.load(open("stationary.json"))
tr = json.load(open("traversal.json"))
ad = json.load(open("adjoint.json"))
env = json.load(open("envelope.json"))
names = ["wet", "transitional", "very dry"]
col = dict(zip(names, [C[0], C[2], C[1]]))

# ------------------------------------------------------------------ Fig. 1
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), layout="constrained")
S = np.linspace(0.0, 1.0, 400)
axs[0].plot(S, g(S), color=INK)
eps = 1e-4
dg = np.abs((g(S + eps) - g(S - eps)) / (2 * eps))
axs[1].plot(S, dg, color=INK)
for n in names:
    s0 = st[n]["S_mean"]
    axs[0].plot(s0, g(s0), "o", color=col[n], ms=5, label=n)
    axs[1].plot(s0, abs((g(s0 + eps) - g(s0 - eps)) / (2 * eps)), "o", color=col[n], ms=5)
axs[0].set_xlabel("soil moisture $S$"); axs[0].set_ylabel("$g(S)$"); axs[0].legend(loc="upper right")
axs[1].set_xlabel("soil moisture $S$"); axs[1].set_ylabel("$|g'(S)|$")
x = np.arange(3)
Tl = [st[n]["T_liang"] for n in names]; Js = [st[n]["max_abs_Jsrc"] for n in names]
axs[2].bar(x - 0.18, Tl, 0.34, color=C[0], label="Liang flow $T^{(1)}$")
axs[2].bar(x + 0.18, Js, 0.34, color=C[1], label=r"$\max_y|J_{\rm src}|$")
axs[2].set_yscale("log"); axs[2].set_xticks(x, names); axs[2].legend(loc="upper left")
for a, l in zip(axs, "abc"):
    tidy(a); label(a, f"({l})")
fig.savefig("fig1_level_slope.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Fig. 2
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), layout="constrained")
r = st["transitional"]; y = np.array(r["y"])
tot = np.array(r["J_self"]) + np.array(r["J_src"]) + np.array(r["J_diff"])
axs[0].plot(y, r["J_src"], color=C[1], label=r"$J_{\rm src}$")
axs[0].plot(y, r["J_self"], color=C[0], label=r"$J_{\rm self}$")
axs[0].plot(y, r["J_diff"], color=C[2], label=r"$J_{\rm diff}$")
axs[0].plot(y, tot, color=INK, lw=1.6, label="total")
axs[0].set_xlim(21, 31); axs[0].set_xlabel("$y$ (target, °C)"); axs[0].set_ylabel("current (d$^{-1}$)")
lo, hi = axs[0].get_ylim(); axs[0].set_ylim(lo, hi + 0.35 * (hi - lo))
axs[0].legend(loc="upper left", ncol=4, handlelength=1.2, columnspacing=0.8)
for n in names:
    rr = st[n]
    axs[1].plot(rr["y"], rr["J_exc"], color=col[n], label=n)
axs[1].set_xlim(18, 36); axs[1].set_xlabel("$y$ (°C)"); axs[1].set_ylabel(r"$J_{\rm exc}$ (d$^{-1}$)")
axs[1].legend(loc="lower right")
t = np.array(tr["t"]); rec = tr["records"]["29.0"]
P = np.array(rec["P"]); J = np.array(rec["total"])
dPdt = np.gradient(P, t)
axs[2].plot(t, J, color=C[1], label="$J(u,t)$")
axs[2].plot(t[::40], dPdt[::40], "o", color=INK, ms=2.5, label="$dP_u/dt$")
axs[2].set_xlabel("time (d)"); axs[2].set_ylabel("rate (d$^{-1}$)"); axs[2].legend(loc="upper left")
for a, l in zip(axs, "abc"):
    tidy(a); label(a, f"({l})")
fig.savefig("fig2_decomposition.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Fig. 4
fig, axs = plt.subplots(2, 1, figsize=(W1, 3.9), layout="constrained")
rec = tr["records"]["29.0"]
for key, c_, lab in (("src", C[1], r"$\int J_{\rm src}$"), ("self", C[0], r"$\int J_{\rm self}$"),
                     ("diff", C[2], r"$\int J_{\rm diff}$")):
    cum = np.concatenate([[0], np.cumsum(0.5 * (np.array(rec[key][1:]) + np.array(rec[key][:-1])) * np.diff(t))])
    axs[0].plot(t, cum, color=c_, label=lab)
axs[0].plot(t, np.array(rec["P"]) - rec["P"][0], color=INK, lw=1.6, label=r"$\Delta P_u$")
axs[0].set_xlabel("time (d)"); axs[0].set_ylabel("integrated current"); axs[0].legend(loc="upper left", ncol=2)
rec = tr["records"]["30.0"]
eff = np.array(rec["P"]) - np.array(rec["P_cf"])
cumexc = np.concatenate([[0], np.cumsum(0.5 * (np.array(rec["exc"][1:]) + np.array(rec["exc"][:-1])) * np.diff(t))])
axs[1].plot(t, eff, color=INK, lw=1.6, label="true effect $P_u-P_u^{\\rm cf}$")
axs[1].plot(t, cumexc, color=C[1], label=r"$\int J_{\rm exc}\,dt$")
axs[1].set_xlabel("time (d)"); axs[1].set_ylabel("probability"); axs[1].legend(loc="upper left")
for a, l in zip(axs, "ab"):
    tidy(a); label(a, f"({l})")
fig.savefig("fig4_nonattribution.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Fig. 5
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), layout="constrained")
US = ad["thresholds"]; EG = np.array(ad["eps_grid"]); k25 = int(np.argmin(abs(EG - ad["eps_main"])))
Ps = [ad["sweep"][str(u)][k25]["P"] for u in US]
ra = [ad["sweep"][str(u)][k25]["dP_adj"] / ad["sweep"][str(u)][k25]["dP_true"] for u in US]
rr = [ad["sweep"][str(u)][k25]["dlogP_adj"] / ad["sweep"][str(u)][k25]["dlogP_true"] for u in US]
axs[0].semilogx(Ps, ra, "s-", color=C[1]); axs[0].axhline(1, color=MUTED, lw=0.6, ls=":")
axs[0].set_xlabel("base rate $P_u$"); axs[0].set_ylabel(r"first order / truth, $\delta P_u$")
axs[1].semilogx(Ps, rr, "o-", color=C[0]); axs[1].axhline(1, color=MUTED, lw=0.6, ls=":")
axs[1].set_ylim(0, max(1.3, max(ra) * 0 + 1.3))
axs[1].set_xlabel("base rate $P_u$"); axs[1].set_ylabel(r"first order / truth, $\delta\log P_u$")
wa = [max(abs(ad["sweep"][str(u)][k]["dP_adj"] / ad["sweep"][str(u)][k]["dP_true"] - 1) for u in US) for k in range(len(EG))]
wr = [max(abs(ad["sweep"][str(u)][k]["dlogP_adj"] / ad["sweep"][str(u)][k]["dlogP_true"] - 1) for u in US) for k in range(len(EG))]
axs[2].loglog(EG, wa, "s-", color=C[1], label=r"$\delta P_u$")
axs[2].loglog(EG, wr, "o-", color=C[0], label=r"$\delta\log P_u$")
axs[2].set_xlabel(r"perturbation $\varepsilon$"); axs[2].set_ylabel("worst-case relative error"); axs[2].legend()
for a, l in zip(axs, "abc"):
    tidy(a); label(a, f"({l})")
fig.savefig("fig5_adjoint.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Fig. 6
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), layout="constrained")
rhos = [d["rho"] for d in env["decoy"]]
axs[0].plot(rhos, [max(np.abs(d["omitted"])) for d in env["decoy"]], "s-", color=C[1], label="true driver omitted")
axs[0].plot(rhos, [max(np.abs(d["both"])) for d in env["decoy"]], "o-", color=C[0], label="both drivers included")
axs[0].set_xlabel(r"correlation $\rho$ with true driver"); axs[0].set_ylabel("decoy share of true attribution")
axs[0].legend(loc="upper left")
ns = [d["n_inc"] for d in env["sample_size"]]
EU = env["thresholds"]
for i, u in enumerate(EU):
    m = [d["mean"][str(u)] for d in env["sample_size"]]; s = [d["sd"][str(u)] for d in env["sample_size"]]
    axs[1].errorbar(np.array(ns) * (1 + 0.04 * (i - 1.5)), m, yerr=s, fmt="o-", ms=2.5, lw=0.9,
                    color=C[i % 4], label=f"$u={u}$")
axs[1].set_xscale("log"); axs[1].axhline(0, color=MUTED, lw=0.6, ls=":")
axs[1].set_xlabel("increment samples"); axs[1].set_ylabel("error vs true generator")
lo, hi = axs[1].get_ylim(); axs[1].set_ylim(lo - 0.45 * (hi - lo), hi); axs[1].legend(ncol=2, loc="lower right")
w = 0.25
for i, (world, lab) in enumerate((("slow_memory", "hidden slow memory"),
                                  ("state_dependent", "state-dependent coupling"),
                                  ("multiplicative", "multiplicative noise"))):
    o = 100 * np.array(env["misspec"][world]["over_attribution"])
    axs[2].bar(np.arange(len(EU)) + (i - 1) * w, o, w, color=C[i], label=lab)
axs[2].axhline(0, color=MUTED, lw=0.6)
axs[2].set_xticks(np.arange(len(EU)), [f"{u}" for u in EU]); axs[2].set_xlabel("threshold $u$")
axs[2].set_ylabel("over-attribution vs control (%)"); axs[2].legend(loc="upper left")
for a, l in zip(axs, "abc"):
    tidy(a); label(a, f"({l})")
fig.savefig("fig6_envelope.pdf", bbox_inches="tight")
print("figures written")
