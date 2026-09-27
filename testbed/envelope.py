"""
Operating envelope of the fitted-generator adjoint attribution
(Secs. V A, VI E; Fig. 6; Table I).

Every experiment follows the same pipeline.
 1. A long training series is simulated from a data-generating world, sampled
    every DT_OBS, and the target drift is fitted by regressing increments,
    dT/dt = -theta T + sum_j c_j(x_j) + residual, with a tent basis per driver
    (Eq. driftfit); sigma^2 = DT_OBS * Var(residual).
 2. An event ensemble is drawn from the world's stationary state.  The estimated
    attribution is the adjoint of Eq. (response) under the FITTED generator, with
    delta F = eps (cbar_j(t) - c_j(x_j)) (Eq. perturbfield, ensemble-mean
    reference), read as a relative change eps A / P.
 3. The ground truth is the relative change in P_u(t1) under the TRUE dynamics
    when the true source forcing g(S) is moved the same fraction eps of the way
    to its ensemble mean, simulated with common random numbers.
Worlds: correctly specified; inert decoy correlated with the true driver;
hidden slow memory; state-dependent coupling; multiplicative noise.
Writes envelope.json.
"""
import json
import numpy as np
from common import *

DT_FINE = 0.01
DT_OBS = 0.1
T_WIN = 5.0
EPS = 0.25
N_ENS = 100_000
NU = logit(REGIMES["transitional"])
KNOTS = 14

# ------------------------------------------------------------------ worlds
TH_H, SD_H = 0.02, 0.45            # hidden slow memory (timescale 50 d)
KAPPA = 0.04                       # state-dependent coupling g(S) (1 + kappa (T - 26)); keeps theta_T - kappa g > 0
LAMBDA = 0.8                       # multiplicative noise sigma(S) = sigma_T (1 + lambda (g(S)/1.5 - 1))


def drift_T(world, T, S, H):
    gS = g(S)
    if world == "state_dependent":
        src = gS * np.clip(1.0 + KAPPA * (T - 26.0), 0.2, 3.0)
    else:
        src = gS
    extra = H if world == "slow_memory" else 0.0
    return -TH_T * (T - T_REF) + src + extra, src


def sigma_T(world, S):
    if world == "multiplicative":
        return SIG_T * np.clip(1.0 + LAMBDA * (g(S) / 1.5 - 1.0), 0.2, None)
    return SIG_T


def step_U(U, h, rng, n):
    a = np.exp(-TH_U * h)
    return NU + a * (U - NU) + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(n)


def step_H(H, h, rng, n):
    a = np.exp(-TH_H * h)
    return a * H + SD_H * np.sqrt(1 - a * a) * rng.standard_normal(n)


# ------------------------------------------------------------------ training data
def training_series(world, n_inc, rng, rho_decoy=None):
    n_fine = int(round(n_inc * DT_OBS / DT_FINE))
    every = int(round(DT_OBS / DT_FINE))
    U = NU + SD_U * rng.standard_normal(1); T = np.array([26.0]); H = SD_H * rng.standard_normal(1)
    V = SD_U * rng.standard_normal(1)                          # decoy's independent part
    obs = {"T": [], "S": [], "U": [], "V": []}
    burn = int(200 / DT_FINE)
    for k in range(burn + n_fine + every):
        if k >= burn and (k - burn) % every == 0:
            obs["T"].append(T[0]); obs["S"].append(logistic(U[0])); obs["U"].append(U[0]); obs["V"].append(V[0])
        S = logistic(U)
        dr, _ = drift_T(world, T, S, H)
        T = T + dr * DT_FINE + sigma_T(world, S) * np.sqrt(DT_FINE) * rng.standard_normal(1)
        U = step_U(U, DT_FINE, rng, 1); H = step_H(H, DT_FINE, rng, 1)
        a = np.exp(-TH_U * DT_FINE)
        V = a * V + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(1)
    obs = {k: np.array(v) for k, v in obs.items()}
    if rho_decoy is not None:
        obs["D"] = rho_decoy * (obs["U"] - NU) + np.sqrt(1 - rho_decoy ** 2) * obs["V"]
    return obs


def training_series_fast(world, n_inc, rng, rho_decoy=None, n_chunks=50):
    """Same law as training_series, vectorised over independent chunks."""
    per = int(np.ceil(n_inc / n_chunks)) + 1
    every = int(round(DT_OBS / DT_FINE))
    n = n_chunks
    U = NU + SD_U * rng.standard_normal(n); T = np.full(n, 26.0); H = SD_H * rng.standard_normal(n)
    V = SD_U * rng.standard_normal(n)
    burn = int(200 / DT_FINE)
    rec = []
    a = np.exp(-TH_U * DT_FINE)
    for k in range(burn + per * every):
        if k >= burn and (k - burn) % every == 0:
            rec.append(np.stack([T, logistic(U), U, V]))
        S = logistic(U)
        dr, _ = drift_T(world, T, S, H)
        T = T + dr * DT_FINE + sigma_T(world, S) * np.sqrt(DT_FINE) * rng.standard_normal(n)
        U = step_U(U, DT_FINE, rng, n); H = step_H(H, DT_FINE, rng, n)
        V = a * V + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(n)
    R = np.array(rec)                                   # (per, 4, n_chunks)
    obs = {"T": R[:, 0], "S": R[:, 1], "U": R[:, 2], "V": R[:, 3]}
    if rho_decoy is not None:
        obs["D"] = rho_decoy * (obs["U"] - NU) + np.sqrt(1 - rho_decoy ** 2) * obs["V"]
    return obs, per


# ------------------------------------------------------------------ fitting
def tent(x, knots):
    B = np.zeros(x.shape + (knots.size,))
    for j, c in enumerate(knots):
        lo = knots[j - 1] if j > 0 else c - (knots[1] - knots[0])
        hi = knots[j + 1] if j < knots.size - 1 else c + (knots[-1] - knots[-2])
        B[..., j] = np.clip(np.minimum((x - lo) / (c - lo), (hi - x) / (hi - c)), 0, None)
    return B


class Fit:
    def __init__(self, obs, per, drivers, n_inc):
        T = obs["T"]                                        # (time, chunk)
        dT = ((T[1:] - T[:-1]) / DT_OBS).T.reshape(-1)       # chunk-major, no cross-chunk increments
        rows = [-T[:-1].T.reshape(-1)]
        self.knots = {}
        for j, name in enumerate(drivers):
            x = obs[name][:-1].T.reshape(-1)
            k = np.quantile(x, np.linspace(0.002, 0.998, KNOTS))
            self.knots[name] = k
            B = tent(x, k)
            rows.append(B if j == 0 else B[:, 1:])          # partition of unity: drop one
        X = np.column_stack(rows)[:n_inc]
        yv = dT[:n_inc]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res = yv - X @ beta
        self.theta = float(beta[0])
        self.sigma = float(np.sqrt(DT_OBS * res.var()))
        self.drivers = drivers
        self.coef = {}
        i = 1
        for j, name in enumerate(drivers):
            m = KNOTS if j == 0 else KNOTS - 1
            self.coef[name] = beta[i:i + m]; i += m
        state = obs["S"][:-1].T.reshape(-1)[:n_inc]
        self.het = float(np.corrcoef(res ** 2, state)[0, 1])
        self.n = n_inc

    def c(self, name, x):
        k = self.knots[name]
        B = tent(np.clip(x, k[0], k[-1]), k)
        if name != self.drivers[0]:
            B = B[..., 1:]
        return B @ self.coef[name]


# ------------------------------------------------------------------ event ensemble
def event_ensemble(world, rng, rho_decoy=None, n=N_ENS):
    """Stationary initial states, then the window [0, T_WIN] on the DT_OBS grid.
    Returns driver paths, initial T, and the true P and P_eps from the true dynamics."""
    U = NU + SD_U * rng.standard_normal(n); T = np.full(n, 26.0)
    H = SD_H * rng.standard_normal(n); V = SD_U * rng.standard_normal(n)
    a = np.exp(-TH_U * DT_FINE)
    for _ in range(int(60 / DT_FINE)):                           # burn-in (U, H start stationary; T relaxes in ~4 d)
        S = logistic(U)
        dr, _ = drift_T(world, T, S, H)
        T = T + dr * DT_FINE + sigma_T(world, S) * np.sqrt(DT_FINE) * rng.standard_normal(n)
        U = step_U(U, DT_FINE, rng, n); H = step_H(H, DT_FINE, rng, n)
        V = a * V + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(n)
    T0 = T.copy()
    every = int(round(DT_OBS / DT_FINE)); steps = int(round(T_WIN / DT_FINE))
    paths = {"S": [], "U": [], "V": []}
    # true factual and counterfactual share the noise
    Tf = T.copy(); Tc = T.copy()
    for k in range(steps + 1):
        if k % every == 0:
            paths["S"].append(logistic(U)); paths["U"].append(U.copy()); paths["V"].append(V.copy())
        if k == steps:
            break
        S = logistic(U)
        z = rng.standard_normal(n)
        drf, _ = drift_T(world, Tf, S, H)
        drc, srcc = drift_T(world, Tc, S, H)
        # counterfactual: the true source term moved eps of the way to its ensemble mean
        drc = drc + EPS * (srcc.mean() - srcc)
        sg = sigma_T(world, S) * np.sqrt(DT_FINE)
        Tf = Tf + drf * DT_FINE + sg * z
        Tc = Tc + drc * DT_FINE + sg * z
        U = step_U(U, DT_FINE, rng, n); H = step_H(H, DT_FINE, rng, n)
        V = a * V + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(n)
    paths = {k: np.array(v) for k, v in paths.items()}
    if rho_decoy is not None:
        paths["D"] = rho_decoy * (paths["U"] - NU) + np.sqrt(1 - rho_decoy ** 2) * paths["V"]
    return paths, T0, Tf, Tc


def truth(Tf, Tc, u):
    P = (Tf > u).mean(); Pc = (Tc > u).mean()
    return float(P), float(Pc), float(np.log(max(Pc, 1e-12)) - np.log(max(P, 1e-12)))


def adjoint_fitted(fit, paths, T0, u, attrib, theta=None, sigma=None, cfun=None):
    """Relative first-order attribution eps*A/P under a (fitted or true) generator.
    cfun(name, x) -> forcing component; attrib = driver whose state dependence is removed."""
    th = fit.theta if theta is None else theta
    sg = fit.sigma if sigma is None else sigma
    c = fit.c if cfun is None else cfun
    t = np.arange(paths["S"].shape[0]) * DT_OBS
    t1 = t[-1]
    F = sum(c(name, paths[name]) for name in (fit.drivers if fit is not None else ["S"]))
    kern = np.exp(-th * (t1 - t))[:, None]
    w = np.full(len(t), DT_OBS); w[0] = w[-1] = DT_OBS / 2
    const = 0.0 if fit is None else 0.0
    M1 = T0 * np.exp(-th * t1) + (w[:, None] * kern * F).sum(0)
    if fit is None:
        M1 = M1 + T_REF * (1 - np.exp(-TH_T * t1))
    s1 = sg * np.sqrt((1 - np.exp(-2 * th * t1)) / (2 * th))
    ca = c(attrib, paths[attrib])
    D = (w[:, None] * kern * (ca.mean(1, keepdims=True) - ca)).sum(0)
    P = sf((u - M1) / s1).mean()
    A = (D * phi((M1 - u) / s1) / s1).mean()
    return float(EPS * A / P), float(P)


def true_c(name, x):
    return g(x)


def run():
    rng = np.random.default_rng(7)
    out = {}
    # ---------------- control: thresholds from the true base rates
    obs, per = training_series_fast("control", 100_000, rng)
    fit_c = Fit(obs, per, ["S"], 100_000)
    paths, T0, Tf, Tc = event_ensemble("control", rng)
    grid = np.arange(24.0, 36.0, 0.1)
    Pg = np.array([(Tf > u).mean() for u in grid])
    US = [float(np.round(grid[np.argmin(np.abs(np.log(np.maximum(Pg, 1e-9)) - np.log(p)))], 1))
          for p in (0.1, 0.03, 0.01, 0.003)]
    out["thresholds"] = US
    print("thresholds", US, "fitted theta", fit_c.theta, "sigma", fit_c.sigma)

    def ratios(world, fit, paths, T0, Tf, Tc, attrib="S"):
        rows = []
        for u in US:
            P, Pc, dl = truth(Tf, Tc, u)
            est, Pfit = adjoint_fitted(fit, paths, T0, u, attrib)
            rows.append(dict(u=u, P_true=P, dlogP_true=dl, dlogP_est=est, P_fit=Pfit,
                             ratio=est / dl if dl != 0 else np.nan))
        return rows

    ctrl = ratios("control", fit_c, paths, T0, Tf, Tc)
    out["control"] = dict(theta=fit_c.theta, sigma=fit_c.sigma, het=fit_c.het, rows=ctrl)
    for r in ctrl:
        print(f"  control u={r['u']}: P={r['P_true']:.4f} true dlogP={r['dlogP_true']:+.4f} "
              f"est={r['dlogP_est']:+.4f} ratio={r['ratio']:.3f}")

    # ---------------- misspecifications
    out["misspec"] = {}
    for world in ("slow_memory", "state_dependent", "multiplicative"):
        obs, per = training_series_fast(world, 100_000, rng)
        fit = Fit(obs, per, ["S"], 100_000)
        p, t0, tf, tc = event_ensemble(world, rng)
        rows = ratios(world, fit, p, t0, tf, tc)
        over = [r["ratio"] / c["ratio"] - 1 for r, c in zip(rows, ctrl)]
        out["misspec"][world] = dict(theta=fit.theta, sigma=fit.sigma, het=fit.het, rows=rows,
                                     over_attribution=over)
        print(f"{world}: theta={fit.theta:.3f} het={fit.het:+.3f} over-attribution "
              + " ".join(f"{o:+.2f}" for o in over)
              + "  base rates " + " ".join(f"{r['P_true']:.3f}" for r in rows))

    # ---------------- decoy correlated with the true driver
    out["decoy"] = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        obs, per = training_series_fast("control", 100_000, rng, rho_decoy=rho)
        fit_d = Fit(obs, per, ["D"], 100_000)
        fit_b = Fit(obs, per, ["S", "D"], 100_000)
        p, t0, tf, tc = event_ensemble("control", rng, rho_decoy=rho)
        row = dict(rho=rho, omitted=[], both=[])
        for u in US:
            _, _, dl = truth(tf, tc, u)
            e_om, _ = adjoint_fitted(fit_d, p, t0, u, "D")
            e_bo, _ = adjoint_fitted(fit_b, p, t0, u, "D")
            row["omitted"].append(e_om / dl); row["both"].append(e_bo / dl)
        out["decoy"].append(row)
        print(f"decoy rho={rho}: share when S omitted {max(np.abs(row['omitted'])):.2f}, "
              f"with S included {max(np.abs(row['both'])):.3f}")

    # ---------------- sample size: fitted vs true generator
    p, t0, tf, tc = event_ensemble("control", rng)
    true_est = {u: adjoint_fitted(None, p, t0, u, "S", theta=TH_T, sigma=SIG_T, cfun=true_c)[0] for u in US}
    out["sample_size"] = []
    for n_inc in (5_000, 10_000, 20_000, 50_000, 100_000, 200_000):
        errs = {u: [] for u in US}
        for rep in range(5):
            obs, per = training_series_fast("control", n_inc, rng)
            fit = Fit(obs, per, ["S"], n_inc)
            for u in US:
                est, _ = adjoint_fitted(fit, p, t0, u, "S")
                errs[u].append(est / true_est[u] - 1)
        row = dict(n_inc=n_inc, mean={str(u): float(np.mean(errs[u])) for u in US},
                   sd={str(u): float(np.std(errs[u], ddof=1)) for u in US})
        out["sample_size"].append(row)
        print(f"n={n_inc}: rel. error vs true generator (mean +- sd over fits) "
              + " ".join(f"{np.mean(errs[u]):+.2f}+-{np.std(errs[u], ddof=1):.2f}" for u in US))
    json.dump(out, open("envelope.json", "w"), default=float)


if __name__ == "__main__":
    run()
