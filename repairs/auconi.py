"""
Auconi, Friedrich & Giansanti, EPL 135, 28002 (2021): the information response

    Gamma^{x->y}_tau = <d_tau^{x->y}> / c_x(eps)          (their Eq. 5)

with the fluctuation-response theorem (Eq. 8) giving it as a ratio of Fisher
informations, and, for Gaussian conditionals, the closed form (their Eq. 11)

    Gamma^{x->y}_tau = ( d<y_tau | x0,y0> / dx0 )^2 * sigma^2_{x0|y0}
                       / sigma^2_{y_tau | x0,y0}

Applied here with x = U (soil state) and y = T.  Note the numerator: it is the
SQUARED derivative of the target's conditional mean with respect to the source.
That is the same object the Liang flow averages and the same object Smirnov's
second-order response squares.  Under saturation it vanishes, and squaring it
makes the suppression quadratic rather than linear.

T_tau | U0, T0 = T_REF + a(T0 - T_REF) + I(tau),  a = exp(-TH_T tau),
so  d<T_tau|U0,T0>/dU0 = d<I(tau)|U0>/dU0, independent of T0, and
sigma^2_{T_tau|U0,T0} = Var(I(tau)|U0) + SIG_T^2 (1-a^2)/(2 TH_T).
"""
import json
import numpy as np

rng = np.random.default_rng(3)

TH_U, TH_T, T_REF = 0.08, 0.25, 20.0
SIG_T = np.sqrt(0.5) * 0.7
SD_U = 0.35
logistic = lambda u: 1.0 / (1.0 + np.exp(-u))
logit = lambda p: np.log(p / (1 - p))
g = lambda S: 3.0 / (1.0 + np.exp((S - 0.35) / 0.055))
REGIMES = {"wet": 0.74, "transitional": 0.35, "very dry": 0.04}


def soil_I(nu, U0, taus, dt, noise):
    """I(tau) along OU soil paths started at U0, with supplied noise."""
    taus = np.asarray(taus, float); nmax = int(round(taus.max() / dt))
    aU = np.exp(-TH_U * dt); sU = SD_U * np.sqrt(1 - aU * aU)
    decay = np.exp(-TH_T * dt)
    U = U0.copy(); gc = g(logistic(U)); I = np.zeros_like(U)
    want = {int(round(t / dt)): k for k, t in enumerate(taus)}
    out = [None] * len(taus)
    for n in range(1, nmax + 1):
        U = nu + aU * (U - nu) + sU * noise[:, n - 1]
        gn = g(logistic(U))
        I = decay * I + 0.5 * dt * (decay * gc + gn); gc = gn
        if n in want:
            out[want[n]] = I.copy()
    return out


def sigma2_U_given_T(nu, n=400_000, dt=0.05, burn=40_000):
    """Residual variance of U regressed on T in the stationary joint."""
    aU = np.exp(-TH_U * dt); sU = SD_U * np.sqrt(1 - aU * aU)
    aT = np.exp(-TH_T * dt); sT = SIG_T * np.sqrt((1 - aT * aT) / (2 * TH_T))
    w = (1 - aT) / TH_T
    U = nu; T = T_REF + float(g(logistic(nu))) / TH_T
    for _ in range(burn):
        U = nu + aU * (U - nu) + sU * rng.standard_normal()
        T = T_REF + aT * (T - T_REF) + w * g(logistic(U)) + sT * rng.standard_normal()
    Us = np.empty(n); Ts = np.empty(n)
    for i in range(n):
        U = nu + aU * (U - nu) + sU * rng.standard_normal()
        T = T_REF + aT * (T - T_REF) + w * g(logistic(U)) + sT * rng.standard_normal()
        Us[i] = U; Ts[i] = T
    A = np.column_stack([np.ones(n), Ts])
    b, *_ = np.linalg.lstsq(A, Us, rcond=None)
    return float(np.var(Us - A @ b)), float(np.corrcoef(Us, Ts)[0, 1])


def gamma(nu, taus, dt=0.01, n_paths=40_000, n_nodes=11, delta=1e-3):
    """Ensemble-averaged Gamma^{U->T}_tau, Auconi Eq. (11)."""
    s2_U_T, r = sigma2_U_given_T(nu)
    a = np.exp(-TH_T * np.asarray(taus, float))
    s2_brown = SIG_T ** 2 * (1 - a ** 2) / (2 * TH_T)
    nmax = int(round(np.asarray(taus, float).max() / dt))

    hx, hw = np.polynomial.hermite_e.hermegauss(n_nodes)
    u0s = nu + SD_U * hx; wts = hw / hw.sum()
    G = np.zeros(len(taus))
    for u0, wq in zip(u0s, wts):
        noise = rng.standard_normal((n_paths, nmax))   # common random numbers
        Ip = soil_I(nu, np.full(n_paths, u0 + delta), taus, dt, noise)
        Im = soil_I(nu, np.full(n_paths, u0 - delta), taus, dt, noise)
        I0 = soil_I(nu, np.full(n_paths, u0), taus, dt, noise)
        for k in range(len(taus)):
            dmean = (Ip[k].mean() - Im[k].mean()) / (2 * delta)
            s2_y = I0[k].var() + s2_brown[k]
            G[k] += wq * (dmean ** 2) * s2_U_T / s2_y
    return G, s2_U_T, r


if __name__ == "__main__":
    taus = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 12.0])
    out = {"taus": taus.tolist()}
    print("Auconi et al. information response Gamma^{U->T}_tau in the testbed")
    print("(their Eq. 11; numerator is the SQUARED derivative of the target's")
    print(" conditional mean with respect to the source)\n")
    print(f"{'regime':14s}" + "".join(f"{t:>11.0f}" for t in taus)
          + f"{'sd(U|T)':>10s}{'corr':>8s}")
    for name, S_star in REGIMES.items():
        nu = logit(S_star)
        G, s2, r = gamma(nu, taus)
        out[name] = dict(Gamma=G.tolist(), s2_U_given_T=s2, corr_UT=r)
        print(f"{name:14s}" + "".join(f"{v:>11.3e}" for v in G)
              + f"{np.sqrt(s2):>10.4f}{r:>8.3f}")

    tr = np.array(out["transitional"]["Gamma"])
    print("\nretention relative to the transitional regime:")
    for name in ("wet", "very dry"):
        v = np.array(out[name]["Gamma"]) / tr
        print(f"  {name:14s}" + "".join(f"{q:>11.2e}" for q in v))
    k = int(np.argmax(tr))
    out["peak_tau"] = float(taus[k])
    out["retention"] = {n: float(np.array(out[n]["Gamma"])[k] / tr[k])
                        for n in ("wet", "very dry")}
    print(f"\nat the transitional peak (tau={taus[k]:.0f}): "
          f"wet {out['retention']['wet']:.2e}, "
          f"very dry {out['retention']['very dry']:.2e}")
    json.dump(out, open("auconi.json", "w"))
    print("\nwrote auconi.json")
