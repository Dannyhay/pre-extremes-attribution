"""
The saturated regime and the choice of reference (Sec. VI F; Table II).

In the very dry regime g(S) is at its maximum and almost constant, so the entropy
transfer vanishes.  Three perturbations of the source forcing are compared over the
five-day window of Sec. VI D, starting from each stationary regime:

  ensemble mean : delta F = gbar(t) - g(S)      (conditional excess, Eq. perturbfield in the testbed)
  climatological: delta F = g_clim  - g(S)      (fixed reference: mean forcing of the transitional regime)
  removal       : delta F = 0       - g(S)      (full removal of the source)

For each, the first-order relative response A/P_u and the exact change in log P_u at
eps = 0.25 and eps = 1 are computed from the Gaussian mixture (exact given soil paths),
together with the prediction of the Gaussian mean-shift formula of Sec. IV B.
Writes saturation.json.
"""
import json
import numpy as np
from scipy.stats import norm
from common import *

N = 200_000
DT = 0.05
T_WIN = 5.0
TARGET_P = [0.3, 0.1, 0.03, 0.01, 0.003]
EPS = [0.25, 1.0]


def window(regime, seed):
    rng = np.random.default_rng(seed)
    M, gs, U = stationary_mixture(REGIMES[regime], N, rng)
    nu = logit(REGIMES[regime])
    t = np.arange(0.0, T_WIN + DT / 2, DT)
    a_u = np.exp(-TH_U * DT); s_u = SD_U * np.sqrt(1 - a_u * a_u)
    a_t = np.exp(-TH_T * DT); w_t = (1 - a_t) / TH_T
    t1 = t[-1]
    G = [gs]
    for k in range(len(t) - 1):
        U = nu + a_u * (U - nu) + s_u * rng.standard_normal(N)
        g_new = g(logistic(U))
        M = T_REF + a_t * (M - T_REF) + w_t * 0.5 * (G[-1] + g_new)
        G.append(g_new)
    G = np.array(G)                                    # (len(t), N)
    ak = np.exp(-TH_T * (t1 - t))                      # kernel at each step
    # trapezoid weights consistent with the forward recursion
    wk = np.zeros(len(t))
    for k in range(len(t) - 1):
        wk[k] += 0.5 * w_t * np.exp(-TH_T * (t1 - t[k + 1]))
        wk[k + 1] += 0.5 * w_t * np.exp(-TH_T * (t1 - t[k + 1]))
    return M, G, wk


def h(z):
    return norm.pdf(z) / norm.sf(z)


def evaluate(M1, D, u, eps_list):
    z = (u - M1) / S_COND
    Q = norm.sf(z); P = Q.mean()
    A = (norm.pdf(z) * D / S_COND).mean()
    out = dict(u=float(u), P=float(P), first_order=float(A / P))
    for e in eps_list:
        Pe = norm.sf((u - M1 - e * D) / S_COND).mean()
        out[f"exact_{e}"] = float(np.log(Pe) - np.log(P)) if Pe > 0 else -np.inf
        out[f"linear_{e}"] = float(e * A / P)
    return out


def meanshift_prediction(M1, D, u, e):
    """Gaussian mean-shift formula: first order -h(z) e delta against the exact log change."""
    sd = np.sqrt(M1.var() + S_COND ** 2)
    z = (u - M1.mean()) / sd
    delta = -e * D.mean() / sd                         # removal moves the threshold up by delta
    exact = norm.logsf(z + delta) - norm.logsf(z)
    first = -h(z) * delta
    return float(1 - first / exact)


stat = json.load(open("stationary.json"))
g_clim = None
res = {}
for regime, seed in (("transitional", 11), ("very dry", 12)):
    M1, G, wk = window(regime, seed)
    if regime == "transitional":
        g_clim = float(G.mean())
    refs = {"ensemble mean": G.mean(1, keepdims=True), "climatological": g_clim, "removal": 0.0}
    grid = np.arange(20.0, 40.0, 0.02)
    Pg = np.array([norm.sf((u - M1) / S_COND).mean() for u in grid])
    US = [float(np.round(grid[np.argmin(np.abs(np.log(Pg) - np.log(p)))], 2)) for p in TARGET_P]
    rows = {}
    for name, ref in refs.items():
        dF = ref - G
        D = wk @ dF                                    # integrated, kernel-weighted perturbation per path
        rr = []
        for u in US:
            r = evaluate(M1, D, u, EPS)
            for e in EPS:
                r[f"shortfall_{e}"] = float(1 - r[f"linear_{e}"] / r[f"exact_{e}"]) if abs(r[f"exact_{e}"]) > 1e-12 else None
                r[f"meanshift_shortfall_{e}"] = meanshift_prediction(M1, D, u, e)
            rr.append(r)
        rows[name] = dict(D_mean=float(D.mean()), D_sd=float(D.std()), rows=rr)
        if regime == "very dry":
            rows[name]["u29"] = evaluate(M1, D, 29.0, EPS)
    res[regime] = dict(T_liang=stat[regime]["T_liang"], M_mean=float(M1.mean()), M_sd=float(M1.std()),
                       g_mean=float(G.mean()), g_sd=float(G.std()), thresholds=US, refs=rows)
    print(f"\n{regime}: Liang flow {stat[regime]['T_liang']:.2e}  g mean {G.mean():.3f} sd {G.std():.4f}  "
          f"M(t1) mean {M1.mean():.2f} sd {M1.std():.3f}  thresholds {US}")
    for name, v in rows.items():
        print(f"  {name:15s} mean integrated dF {v['D_mean']:+.3f} (sd {v['D_sd']:.3f})")
        for r in v["rows"]:
            print(f"     u={r['u']:.2f} P={r['P']:.4f}  first-order dlogP/deps={r['first_order']:+.4f} | "
                  f"eps=.25 exact {r['exact_0.25']:+.4f} shortfall {100*(r['shortfall_0.25'] or 0):5.1f}% (mean-shift {100*r['meanshift_shortfall_0.25']:5.1f}%) | "
                  f"eps=1 exact {r['exact_1.0']:+.3f} shortfall {100*(r['shortfall_1.0'] or 0):5.1f}% (mean-shift {100*r['meanshift_shortfall_1.0']:5.1f}%)")
        if "u29" in v:
            r = v["u29"]
            print(f"     u=29 P={r['P']:.6f} first-order {r['first_order']:+.2e}  eps=1 exact {r['exact_1.0']:+.3f}")
res["g_clim"] = g_clim
json.dump(res, open("saturation.json", "w"), indent=1)
