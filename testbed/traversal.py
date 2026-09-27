"""
Nonstationary wet-to-dry traversal (Secs. III, VI B-C; Fig. 2(c); Fig. 4).

nu(t) moves linearly from logit(0.74) (wet) to logit(0.04) (very dry) over
[T0_RAMP, T1_RAMP] and is held there afterwards.  The ensemble starts from the
stationary wet mixture.  Along the traversal we record, for five thresholds,
  * P_u(t) and the current J(u,t) with its terms (exceedance-flux identity, Eq. T1);
  * the cumulative budget (Eq. cumulative) and the integrated source and self terms;
  * a ground-truth counterfactual in which g(S) is replaced by its ensemble mean
    at each time, and the integrated conditional-excess current (Eq. T2).
Writes traversal.json.
"""
import json
import numpy as np
from common import *

rng = np.random.default_rng(2)
N = 200_000
DT = 0.05
T_END = 160.0
T0_RAMP, T1_RAMP = 10.0, 130.0
NU_WET, NU_DRY = logit(0.74), logit(0.04)
US = [28.0, 29.0, 30.0, 31.0, 32.0]


def nu_of_t(t):
    f = np.clip((t - T0_RAMP) / (T1_RAMP - T0_RAMP), 0.0, 1.0)
    return NU_WET + f * (NU_DRY - NU_WET)


t = np.arange(0.0, T_END + DT / 2, DT)
M, gs, U = stationary_mixture(REGIMES["wet"], N, rng)
M_cf = M.copy()                                  # counterfactual conditional means
a_u = np.exp(-TH_U * DT); s_u = SD_U * np.sqrt(1 - a_u * a_u)
a_t = np.exp(-TH_T * DT); w_t = (1 - a_t) / TH_T

rec = {u: {k: [] for k in ("P", "P_cf", "self", "src", "diff", "mean", "exc", "total")} for u in US}
gbar = []
for k, tk in enumerate(t):
    gbar.append(gs.mean())
    for u in US:
        c = currents_at(u, M, gs)
        r = rec[u]
        r["P"].append(float(sf((u - M) / S_COND).mean()))
        r["P_cf"].append(float(sf((u - M_cf) / S_COND).mean()))
        for key in ("self", "src", "diff", "mean", "exc", "total"):
            r[key].append(float(c[key]))
    if k == len(t) - 1:
        break
    nu = nu_of_t(tk)
    U = nu + a_u * (U - nu) + s_u * rng.standard_normal(N)
    g_new = g(logistic(U))
    M = T_REF + a_t * (M - T_REF) + w_t * 0.5 * (gs + g_new)
    M_cf = T_REF + a_t * (M_cf - T_REF) + w_t * 0.5 * (gs.mean() + g_new.mean())
    gs = g_new

out = dict(t=t.tolist(), gbar=gbar, thresholds=US, N=N, dt=DT,
           ramp=[T0_RAMP, T1_RAMP], records={str(u): rec[u] for u in US})

# ---- exceedance-flux identity at 17 times, five thresholds
idx = np.linspace(20, len(t) - 21, 17).astype(int)
errs = []
for u in US:
    P = np.array(rec[u]["P"]); J = np.array(rec[u]["total"])
    dPdt = (P[idx + 1] - P[idx - 1]) / (2 * DT)
    ok = np.abs(J[idx]) > 1e-4
    errs += list(np.abs(dPdt[ok] - J[idx][ok]) / np.abs(J[idx][ok]))
out["identity_rel_err"] = dict(median=float(np.median(errs)), max=float(np.max(errs)), n=len(errs))
print(f"dP/dt = J(u,t): median rel err {np.median(errs):.2e}, max {np.max(errs):.2e} (n={len(errs)})")

# ---- cumulative budget and non-attribution
summ = {}
for u in US:
    r = {k: np.array(v) for k, v in rec[u].items()}
    dP = r["P"][-1] - r["P"][0]
    I = {k: float(np.trapezoid(r[k], t)) for k in ("self", "src", "diff", "exc", "mean")}
    budget_err = abs(I["self"] + I["src"] + I["diff"] - dP)
    effect = r["P"] - r["P_cf"]
    kpk = int(np.argmax(np.abs(effect)))
    exc_to_peak = float(np.trapezoid(r["exc"][:kpk + 1], t[:kpk + 1]))
    summ[str(u)] = dict(dP=float(dP), int_self=I["self"], int_src=I["src"], int_diff=I["diff"],
                        int_exc=I["exc"], budget_abs_err=float(budget_err),
                        peak_effect=float(effect[kpk]), t_peak=float(t[kpk]),
                        exc_integrated_to_peak=exc_to_peak)
    print(f"u={u:.0f}: dP={dP:.3f}  int J_src={I['src']:+.2f}  int J_self={I['self']:+.2f}  "
          f"int J_diff={I['diff']:+.2f}  budget err={budget_err:.1e}  "
          f"true peak effect={effect[kpk]:.3f} at t={t[kpk]:.1f}  int J_exc to peak={exc_to_peak:.3f}")
out["summary"] = summ
json.dump(out, open("traversal.json", "w"))
