"""
Two diagnostics of the fitted-generator pipeline of envelope.py (Sec. VI E).

1. Residual autocorrelation: hidden slow memory leaves the squared-residual
   diagnostic untouched but appears as a small, persistent autocorrelation of the
   drift residuals at lags of days.  Computed for every world at 1e5 increments.
2. Sampling step: the fitted relaxation rate against the sampling interval DT_OBS
   in the correctly specified world, compared with the finite-lag value
   (1 - exp(-theta dt))/dt expected of an increment regression.
3. Basis: on the same three training sets (DT_OBS = 0.1), the fitted relaxation rate
   with the true source function g(S) as regressor and with 14, 28 and 56 tent knots.
Writes diagnostics.json.
"""
import json
import numpy as np
import envelope as E
from common import *


def fit_residuals(obs, drivers, n_inc, dt_obs):
    T = obs["T"]
    dT = ((T[1:] - T[:-1]) / dt_obs).T                     # (chunk, time)
    rows = [-T[:-1].T]
    for j, name in enumerate(drivers):
        x = obs[name][:-1].T
        k = np.quantile(x, np.linspace(0.002, 0.998, E.KNOTS))
        B = E.tent(x, k)
        rows.append(B if j == 0 else B[..., 1:])
    X = np.concatenate([r[..., None] if r.ndim == 2 else r for r in rows], axis=-1)
    nch, nt, p = X.shape
    Xf, yf = X.reshape(-1, p)[:n_inc], dT.reshape(-1)[:n_inc]
    beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    res = (dT - X @ beta)                                  # (chunk, time), contiguous in time
    return float(beta[0]), res


def acf(res, lag):
    a, b = res[:, :-lag].ravel(), res[:, lag:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


out = {"acf": {}, "dt": {}}
rng = np.random.default_rng(21)
LAGS = {"0.1 d": 1, "1 d": 10, "2 d": 20, "5 d": 50}
for world in ("control", "slow_memory", "state_dependent", "multiplicative"):
    obs, per = E.training_series_fast(world, 100_000, rng)
    th, res = fit_residuals(obs, ["S"], 100_000, E.DT_OBS)
    r = {k: acf(res, L) for k, L in LAGS.items()}
    # portmanteau over lags 1-5 d, in units of its null standard error
    n = res.size
    lb = n * sum(acf(res, L) ** 2 for L in range(10, 51))
    out["acf"][world] = dict(theta_fit=th, acf=r, ljung_box_1to5d=float(lb), n=int(n))
    print(f"{world:16s} theta_fit={th:.3f}  acf " + "  ".join(f"{k}:{v:+.3f}" for k, v in r.items())
          + f"   Q(1-5 d)={lb:.0f} (41 lags; 99% null ~64)")

for dt_obs in (0.02, 0.05, 0.1, 0.2, 0.5):
    E.DT_OBS = dt_obs
    ths = []
    for rep in range(3):
        obs, per = E.training_series_fast("control", 100_000, np.random.default_rng(100 + rep))
        f = E.Fit(obs, per, ["S"], 100_000)
        ths.append(f.theta)
    fin = (1 - np.exp(-TH_T * dt_obs)) / dt_obs
    out["dt"][str(dt_obs)] = dict(theta_fit=float(np.mean(ths)), theta_sd=float(np.std(ths)), finite_lag=float(fin))
    print(f"DT_OBS={dt_obs}: fitted theta {np.mean(ths):.4f} +- {np.std(ths):.4f}   finite-lag value {fin:.4f}   true 0.25")
E.DT_OBS = 0.1
out["basis"] = {}
sets = [E.training_series_fast("control", 100_000, np.random.default_rng(100 + rep)) for rep in range(3)]
th_true = []
for obs, per in sets:
    T = obs["T"]; dT = ((T[1:] - T[:-1]) / E.DT_OBS).T.reshape(-1)[:100_000]
    X = np.column_stack([-T[:-1].T.reshape(-1), np.ones(T[:-1].size), g(obs["S"][:-1]).T.reshape(-1)])[:100_000]
    b, *_ = np.linalg.lstsq(X, dT, rcond=None); th_true.append(b[0])
out["basis"]["true_g"] = dict(theta=float(np.mean(th_true)), sd=float(np.std(th_true)))
print(f"true g(S) regressor: theta {np.mean(th_true):.4f} +- {np.std(th_true):.4f}")
for knots in (14, 28, 56):
    E.KNOTS = knots
    ths = [E.Fit(obs, per, ["S"], 100_000).theta for obs, per in sets]
    out["basis"][str(knots)] = dict(theta=float(np.mean(ths)), sd=float(np.std(ths)))
    print(f"tent basis, {knots} knots: theta {np.mean(ths):.4f} +- {np.std(ths):.4f}")
E.KNOTS = 14
json.dump(out, open("diagnostics.json", "w"), indent=1)
