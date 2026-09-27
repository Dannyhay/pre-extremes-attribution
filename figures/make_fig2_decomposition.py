"""Figure 2 (flux decomposition), redrawn 25 Sep 2026.

The original generating script was not archived. The curves below were recovered
exactly from the vector paths of the published fig2_decomposition.pdf
(matplotlib output, same points matplotlib wrote), mapped back to data
coordinates through the tick positions, and saved to fig2_curves.json. Only the
legend placement changes: in (b) the legend sat on the 'transitional' peak, and
in (a) it sat on the J_self curve.
"""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "fig2_decomposition_original.pdf"
C = ["#0072B2", "#D55E00", "#009E73"]; INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False,
    "mathtext.fontset": "dejavuserif",
})

def extract():
    """Curves are taken from the exact drawing objects of the original PDF, by index.
    Legend swatches (2-point lines inside the legend box) are excluded by construction."""
    import pymupdf
    pg = pymupdf.open(SRC)[0]
    dr = pg.get_drawings()
    PANELS = {  # (px, value) tick anchors read from the original axes
      "a": dict(x=(59.2, 20, 130.0, 30), y=(63.6, 0.0, 31.1, 0.2)),
      "b": dict(x=(238.7, 20, 330.0, 35), y=(68.0, 0.0, 36.1, 0.05)),
      "c": dict(x=(382.9, 0, 495.4, 60), y=(110.7, 0.0, 49.5, 0.15)),
    }
    INDEX = {"a_blue": 22, "a_orange": 23, "a_green": 24, "a_black": 25,
             "b_blue": 54, "b_orange": 55, "b_green": 56,
             "c_black_w": 78, "c_orange": 79}
    def lin(p0, v0, p1, v1): return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)
    out = {}
    for tag, i in INDEX.items():
        P = PANELS[tag[0]]; fx = lin(*P["x"]); fy = lin(*P["y"])
        items = dr[i]["items"]
        pts = [items[0][1]] + [it[-1] for it in items]          # start + each segment end
        out[tag] = [{"x": [round(fx(q.x), 6) for q in pts], "y": [round(fy(q.y), 7) for q in pts]}]
    return out

if __name__ == "__main__":
    if not os.path.exists("fig2_curves.json"):
        json.dump(extract(), open("fig2_curves.json", "w"), indent=0)
    D = json.load(open("fig2_curves.json"))
    def seg(tag):
        s = D[tag]; return np.concatenate([q["x"] for q in s]), np.concatenate([q["y"] for q in s])

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.05), layout="constrained")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.5, zorder=0); a.set_axisbelow(True)

    # (a) decomposition of the current
    a = ax[0]
    a.plot(*seg("a_blue"),   color=C[0], lw=1.3, label=r"$J_{\rm self}$")
    a.plot(*seg("a_orange"), color=C[1], lw=1.3, label=r"$J_{\rm src}$")
    a.plot(*seg("a_green"),  color=C[2], lw=1.3, label=r"$J_{\rm diff}$")
    a.plot(*seg("a_black"),  color=INK,  lw=1.6, label=r"total $J$")
    a.set_xlim(17.10, 34.58); a.set_ylim(-0.341, 0.395)   # headroom above J_src peak (0.22) for the legend
    a.set_xticks([20, 25, 30]); a.set_yticks(np.round(np.arange(-0.3, 0.31, 0.1), 1))
    a.set_xlabel(r"target $y$"); a.set_ylabel("current")
    a.legend(loc="upper left", ncol=4, handlelength=1.1, handletextpad=0.4, columnspacing=0.7, borderaxespad=0.15)

    # (b) conditional excess by regime
    b = ax[1]
    b.axhline(0, color=MUTED, lw=0.6, zorder=1)
    b.plot(*seg("b_blue"),   color=C[0], lw=1.3, label="wet")
    b.plot(*seg("b_orange"), color=C[1], lw=1.3, label="transitional")
    b.plot(*seg("b_green"),  color=C[2], lw=1.3, label="very dry")
    b.set_xlim(15.83, 36.15); b.set_ylim(-0.0794, 0.0830)
    b.set_xticks([20, 25, 30, 35]); b.set_yticks([-0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075])
    b.set_xlabel(r"target $y$"); b.set_ylabel(r"$J_{\rm exc}$")
    b.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)

    # (c) exceedance-flux identity check
    c = ax[2]
    c.plot(*seg("c_black_w"), color="#9a9a9a", lw=2.2, label=r"$dP_u/dt$ (finite diff.)")
    c.plot(*seg("c_orange"),  color=C[1], lw=1.1, label=r"$J(u,t)$")
    c.set_xlim(-2.99, 62.99); c.set_ylim(-0.0199, 0.2348)
    c.set_xticks([0, 20, 40, 60]); c.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    c.set_xlabel("time"); c.set_ylabel(r"rate at $u=29.0$")
    c.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)

    for a, l in zip(ax, "abc"):
        a.text(-0.20, 1.04, f"({l})", transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.savefig("fig2_decomposition.pdf"); fig.savefig("fig2_decomposition_check.png", dpi=220)
    print("series:", {k: sum(len(q["x"]) for q in v) for k, v in D.items()})
