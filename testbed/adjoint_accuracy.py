"""
Accuracy of the adjoint response against the ground-truth counterfactual
(Secs. IV B, V C, VI D; Fig. 5).

Window [0, T_WIN] (T_WIN = 5 d, comparable to the 4.5-5.6 d lead times of Sec. VII A) starting from the stationary transitional mixture.  The
perturbation removes the state dependence of the source,
    delta F = eps * (gbar(t) - g(S)),   gbar(t) = ensemble mean of g at t,
(Eq. perturbfield with the ensemble-mean reference).

Ground truth: because T is conditionally linear, the perturbed conditional
means are exact, M_eps = M + eps * int a(t) (gbar - g) dt, and
P_eps = mean Phibar((u - M_eps(t1))/s).

Adjoint (Eq. response with the semi-analytic backward solution, Eq. semianalytic):
the expectation of d_y psi over the Gaussian conditional law of T_t collapses to
a(t) phi((M(t1) - u)/s)/s, so the per-path contribution is
    C_i = [int_0^t1 a(t) (gbar - g_i) dt] * phi((M_i(t1) - u)/s) / s,
A = mean C_i, standard error sd(C_i)/sqrt(N).
Writes adjoint.json.
"""
import json
import numpy as np
from common import *

N = 200_000
DT = 0.05
T_WIN = 5.0
N_ENS = 6
TARGET_P = [0.3, 0.1, 0.03, 0.01, 0.003]
EPS_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
EPS_MAIN = 0.25


def run_window(seed):
    rng = np.random.default_rng(seed)
    M, gs, U = stationary_mixture(REGIMES["transitional"], N, rng)
    nu = logit(REGIMES["transitional"])
    t = np.arange(0.0, T_WIN + DT / 2, DT)
    a_u = np.exp(-TH_U * DT); s_u = SD_U * np.sqrt(1 - a_u * a_u)
    a_t = np.exp(-TH_T * DT); w_t = (1 - a_t) / TH_T
    D = np.zeros(N)                                   # int a(t) (gbar - g_i) dt
    t1 = t[-1]
    for k in range(len(t) - 1):
        U_new = nu + a_u * (U - nu) + s_u * rng.standard_normal(N)
        g_new = g(logistic(U_new))
        # exact kernel over the step, trapezoid in the forcing
        fk = gs.mean() - gs
        fk1 = g_new.mean() - g_new
        ak = np.exp(-TH_T * (t1 - t[k])); ak1 = np.exp(-TH_T * (t1 - t[k + 1]))
        D += 0.5 * DT * (ak * fk + ak1 * fk1)
        M = T_REF + a_t * (M - T_REF) + w_t * 0.5 * (gs + g_new)
        gs, U = g_new, U_new
    return M, D                                        # M = M(t1)


def evaluate(M1, D, u, eps):
    P = sf((u - M1) / S_COND).mean()
    Pe = sf((u - (M1 + eps * D)) / S_COND).mean()
    C = D * phi((M1 - u) / S_COND) / S_COND
    A = C.mean(); se = C.std(ddof=1) / np.sqrt(len(C))
    return dict(P=float(P), P_eps=float(Pe), A=float(A), A_se=float(se),
                dP_true=float(Pe - P), dP_adj=float(eps * A),
                dlogP_true=float(np.log(Pe) - np.log(P)), dlogP_adj=float(eps * A / P),
                dlogP_se=float(eps * se / P))


ens = [run_window(100 + e) for e in range(N_ENS)]

# thresholds: fixed values giving the target base rates in the first ensemble
M1 = ens[0][0]
grid = np.arange(24.0, 36.0, 0.05)
Pg = np.array([sf((u - M1) / S_COND).mean() for u in grid])
US = [float(np.round(grid[np.argmin(np.abs(np.log(Pg) - np.log(p)))], 1)) for p in TARGET_P]
print("thresholds:", US)

out = dict(N=N, dt=DT, T_win=T_WIN, thresholds=US, eps_grid=EPS_GRID, eps_main=EPS_MAIN)

# eps sweep on ensemble 0
sweep = {str(u): [evaluate(*ens[0], u, e) for e in EPS_GRID] for u in US}
out["sweep"] = sweep
for u in US:
    r = sweep[str(u)]
    err_abs = [abs(x["dP_adj"] - x["dP_true"]) / abs(x["dP_true"]) for x in r]
    err_rel = [abs(x["dlogP_adj"] - x["dlogP_true"]) / abs(x["dlogP_true"]) for x in r]
    print(f"u={u}: P={r[0]['P']:.4f}  err@eps=0.01 abs {err_abs[0]:.3f} rel {err_rel[0]:.3f} | "
          f"eps=0.25 abs {err_abs[5]:.3f} rel {err_rel[5]:.3f} | eps=1 abs {err_abs[-1]:.2f} rel {err_rel[-1]:.2f}")

# reproducibility across independent ensembles at eps = 0.25
rep = {}
for u in US:
    rows = [evaluate(*e, u, EPS_MAIN) for e in ens]
    ratio = np.array([r["dlogP_adj"] / r["dlogP_true"] for r in rows])
    spread = np.std([r["A"] for r in rows], ddof=1)          # spread of A across ensembles
    mean_se = np.mean([r["A_se"] for r in rows])             # mean analytic standard error of A
    rep[str(u)] = dict(ratio_mean=float(ratio.mean()), ratio_sd=float(ratio.std(ddof=1)),
                       spread_over_se=float(spread / mean_se), rows=rows)
    print(f"u={u}: relative reading / truth = {ratio.mean():.3f} +- {ratio.std(ddof=1):.3f}   "
          f"spread/SE = {spread / mean_se:.2f}")
out["replicates"] = rep
json.dump(out, open("adjoint.json", "w"))
