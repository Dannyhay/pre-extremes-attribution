"""
Stationary comparisons (Secs. II D, VI A-B; Fig. 1; Fig. 2(a),(b)).

For the wet, transitional and very dry equilibria:
  * Liang information flow T^(1) = int d_y j rho dy, and independently in its
    integrated-by-parts form -int j d_y rho dy (the bridge identity, Eq. T3);
  * total current and its three terms (Eq. split) and max_y |J_src|;
  * the conditional-excess current (Eq. T2);
  * an independent-source check: a source with nonzero mean, independent of T.
Writes stationary.json.
"""
import json
import numpy as np
from common import *

rng = np.random.default_rng(1)
N = 400_000
out = {}
for name, S_star in REGIMES.items():
    M, gs, U = stationary_mixture(S_star, N, rng)
    y = np.linspace(M.min() - 7 * S_COND, M.max() + 7 * S_COND, 3001)
    rho, j, dj, drho = mixture_fields(M, gs, y)
    T_a = np.trapezoid(dj * rho, y)                 # int d_y j rho
    T_b = -np.trapezoid(j * drho, y)                # by parts: -int j d_y rho
    J_self = -TH_T * (y - T_REF) * rho
    J_src = j * rho
    J_diff = -0.5 * SIG_T ** 2 * drho
    J_tot = J_self + J_src + J_diff
    J_exc = (j - gs.mean()) * rho
    out[name] = dict(
        S_star=S_star, T_liang=float(T_a), T_liang_byparts=float(T_b),
        bridge_rel_err=float(abs(T_a - T_b) / abs(T_a)),
        max_abs_Jsrc=float(np.abs(J_src).max()),
        max_abs_Jtot=float(np.abs(J_tot).max()),
        max_abs_Jself=float(np.abs(J_self).max()),
        mean_T=float(np.trapezoid(y * rho, y)),
        g_mean=float(gs.mean()), S_mean=float(logistic(U).mean()),
        y=y[::10].tolist(), J_self=J_self[::10].tolist(), J_src=J_src[::10].tolist(),
        J_diff=J_diff[::10].tolist(), J_exc=J_exc[::10].tolist(), rho=rho[::10].tolist(),
        j=j[::10].tolist(), dj=dj[::10].tolist())
    print(f"{name:13s} T^(1)={T_a:.4g}  bridge rel.err={out[name]['bridge_rel_err']:.1e}  "
          f"max|J_src|={out[name]['max_abs_Jsrc']:.4f}  max|J_total|={out[name]['max_abs_Jtot']:.1e}  "
          f"<T>={out[name]['mean_T']:.2f}")

# independent source with nonzero mean: X independent of the soil path, b = 1.
# Independence is imposed exactly by pairing every soil path with every X draw
# (product measure), so E[X | T=y] = mean(X) identically.
M, gs, _ = stationary_mixture(REGIMES["transitional"], 50_000, rng)
X = 2.0 + rng.standard_normal(200)
y = np.linspace(M.min() - 7 * S_COND, M.max() + 7 * S_COND, 3001)
rho, _, _, _ = mixture_fields(M, gs, y)
jX = np.full_like(y, X.mean())                 # (sum_i phi_i)(sum_k X_k)/(n_i n_k rho)
exc = (jX - X.mean()) * rho
out["independent_source"] = dict(max_abs_Jexc=float(np.abs(exc).max()),
                                 max_abs_Jmean=float(np.abs(X.mean() * rho).max()))
print("independent source: max|J_exc| =", out["independent_source"]["max_abs_Jexc"],
      " max|J_mean| =", out["independent_source"]["max_abs_Jmean"])
json.dump(out, open("stationary.json", "w"))
