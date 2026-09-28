"""
Fitted generator for the heatwave application (Sec. VII A).

Model, in model time (days), for hourly site series, with h the hour-of-day phase:

  dT = [-theta_T(h) T + c_S(S,h) + c_H(H,h) + c_Z(Z) + diurnal_T(h) + seasonal_T + a_T] dt + sigma_T(h) dW_T
  dS = [-theta_S S    + d_T(T)   + d_Z(Z)             + diurnal_S(h) + seasonal_S + a_S] dt + sigma_S    dW_S

T  ERA5-Land 2-m temperature [degC]; S ERA5-Land 0-100 cm soil moisture [m3 m-3];
Z  ERA5 500-hPa geopotential height [m]; H ERA5 850-hPa temperature [degC].
W_T and W_S are independent.  Z and H are exogenous and prescribed as observed.

Each driver component is piecewise linear on 14 evenly spaced knots between the
0.5th and 99.5th percentiles of that driver in the baseline summers.  Beyond the
outer knots it is continued linearly with the slope of the outermost segment
("linear", the default) or held constant ("flat"); the SAME rule is used when
fitting and when evaluating.  With diurnal modulation (default) the relaxation rate
and the soil and 850-hPa components of the temperature drift vary with hour of day
through two harmonics, c(x,h) = sum_l B_l(x) [v_l0 + sum_k a_lk sin kh + b_lk cos kh],
and the temperature noise amplitude is estimated separately for each hour of day.
diurnal_* and seasonal_* are two additive harmonics of hour of day and day of year.

The drift is fitted by least squares on increments between consecutive hours
within one summer (increments that span a gap, e.g. 31 Aug -> 1 Jun of the next
year, are excluded), and sigma^2 = dt Var(residual).  The model is integrated on
the same hourly step, so the simulated model is the fitted discrete model, and the
adjoint below uses the same discrete kernel: it is the exact derivative of what is
simulated.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

N_KNOTS = 14
N_DIURNAL = 2
N_SEASONAL = 2
MAX_LAG = 160          # hours of residual autocorrelation carried (longest window is 135 h)


# ------------------------------------------------------------------ bases
def basis(x, knots, tail):
    """Piecewise-linear basis on `knots` with the given tail rule (all m columns)."""
    x = np.asarray(x, float)
    m, dk = knots.size, knots[1] - knots[0]
    xi = np.clip(x, knots[0], knots[-1])
    B = np.zeros(x.shape + (m,))
    for j in range(m):
        B[..., j] = np.clip(1 - np.abs(xi - knots[j]) / dk, 0, None)
    if tail == "linear":
        hi = np.clip((x - knots[-1]) / dk, 0, None)
        lo = np.clip((knots[0] - x) / dk, 0, None)
        B[..., -1] += hi; B[..., -2] -= hi
        B[..., 0] += lo; B[..., 1] -= lo
    elif tail != "flat":
        raise ValueError(tail)
    return B


def harmonics(phase, n):
    phase = np.atleast_1d(np.asarray(phase, float))
    return np.column_stack([f(k * phase) for k in range(1, n + 1) for f in (np.sin, np.cos)])


def hvec(phase, modulated):
    """[1] or [1, sin h, cos h, sin 2h, cos 2h], shape (len, nh)."""
    phase = np.atleast_1d(np.asarray(phase, float))
    if not modulated:
        return np.ones((phase.size, 1))
    return np.column_stack([np.ones(phase.size), harmonics(phase, N_DIURNAL)])


class Component:
    """c(x, h) = sum_l B_l(x) (V @ hvec(h))_l, piecewise linear in x with the stated tail rule."""
    def __init__(self, knots, V, tail, modulated):
        self.knots, self.V, self.tail, self.mod = knots, V, tail, modulated
        self.dk = knots[1] - knots[0]

    def values(self, phase):
        return self.V @ hvec(phase, self.mod)[0]            # knot values at one hour phase

    def __call__(self, x, phase=0.0):
        """x array at a single hour phase (scalar)."""
        x = np.asarray(x, float)
        k, v = self.knots, self.values(phase)
        out = np.interp(x, k, v)
        if self.tail == "linear":
            out = out + np.where(x > k[-1], (x - k[-1]) * (v[-1] - v[-2]) / self.dk, 0.0) \
                      + np.where(x < k[0], (x - k[0]) * (v[1] - v[0]) / self.dk, 0.0)
        return out

    def along(self, x, phase):
        """x and phase arrays of equal length (one value per time step)."""
        B = basis(x, self.knots, self.tail)                  # (n, m)
        return np.einsum("nm,mh,nh->n", B, self.V, hvec(phase, self.mod))


# ------------------------------------------------------------------ data
def load(csv):
    df = pd.read_csv(csv, parse_dates=["time"]).set_index("time").sort_index()
    df = df.rename(columns={"SM": "S"})
    df["hour_phase"] = 2 * np.pi * df.index.hour / 24.0
    df["doy_phase"] = 2 * np.pi * df.index.dayofyear / 365.25
    return df


def increments(df, years, dt_h=1.0):
    """Rows of the baseline summers whose successor is exactly one step later."""
    jja = df.index.month.isin([6, 7, 8]) & df.index.year.isin(years)
    succ = df.index.to_series().shift(-1)
    good = jja & ((succ - df.index.to_series()) == pd.Timedelta(hours=dt_h)).values
    d = df[good]
    nxt = df.shift(-1)[good]
    dt = dt_h / 24.0
    return d, (nxt["T"].values - d["T"].values) / dt, (nxt["S"].values - d["S"].values) / dt, dt


def make_knots(x):
    return np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), N_KNOTS)


# ------------------------------------------------------------------ fitting
def fit_equation(y, own, drivers, knots, hour_phase, doy_phase, dt, tail,
                 modulated=(), mod_own=False, sigma_by_hour=False, weights=None, times=None, cache=None,
                 solver=None):
    hv_mod = hvec(hour_phase, True)
    own_cols = own[:, None] * (hv_mod if mod_own else np.ones((own.size, 1)))
    if cache is not None and "X" in cache:
        X, blocks, p = cache["X"], cache["blocks"], cache["p"]
    else:
        cols, blocks, p = [np.ones_like(own)[:, None], own_cols], {}, 1 + own_cols.shape[1]
        for name, x in drivers.items():
            B = basis(x, knots[name], tail)[:, 1:]               # value at the first knot fixed to 0
            if name in modulated:
                B = (B[:, :, None] * hv_mod[:, None, :]).reshape(len(x), -1)
            blocks[name] = (p, p + B.shape[1]); cols.append(B); p += B.shape[1]
        cols += [harmonics(hour_phase, N_DIURNAL), harmonics(doy_phase, N_SEASONAL)]
        X = np.column_stack(cols)
        if cache is not None:
            cache.update(X=X, blocks=blocks, p=p)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    if solver is not None:
        beta = solver(Xo, yo, ok)
    elif weights is None:
        beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    else:
        w = np.sqrt(weights[ok])
        beta, *_ = np.linalg.lstsq(Xo * w[:, None], yo * w, rcond=None)
    resid = yo - Xo @ beta
    comps = {}
    for name, (i, j) in blocks.items():
        nh = 1 + 2 * N_DIURNAL if name in modulated else 1
        V = np.vstack([np.zeros((1, nh)), beta[i:j].reshape(-1, nh)])
        comps[name] = Component(knots[name], V, tail, name in modulated)
    q = p
    own_coef = -beta[1:1 + own_cols.shape[1]]
    hr = np.round(hour_phase[ok] * 24 / (2 * np.pi)).astype(int) % 24
    wv = np.ones(resid.size) if weights is None else weights[ok]
    def wvar(r, ww):
        mu = np.average(r, weights=ww); return np.average((r - mu) ** 2, weights=ww)
    if sigma_by_hour:
        sig_h = np.array([np.sqrt(dt * wvar(resid[hr == h], wv[hr == h])) for h in range(24)])
    else:
        sig_h = np.full(24, np.sqrt(dt * wvar(resid, wv)))
    theta_mean = float(own_coef[0])
    rho = None
    if times is not None:
        zres = resid * np.sqrt(dt) / sig_h[hr]
        ser = pd.Series(np.nan, index=times)
        ser.loc[times[ok]] = zres
        full = ser.asfreq("h")
        v = full.values
        rho = np.ones(MAX_LAG + 1)
        for L in range(1, MAX_LAG + 1):
            a, b = v[:-L], v[L:]
            g = np.isfinite(a) & np.isfinite(b)
            rho[L] = np.corrcoef(a[g], b[g])[0, 1]
    return dict(rho=rho, theta_coef=own_coef, mod_own=mod_own, theta=theta_mean, intercept=float(beta[0]),
                comps=comps, diurnal=beta[q:q + 2 * N_DIURNAL], seasonal=beta[q + 2 * N_DIURNAL:],
                sigma_h=sig_h, sigma=float(np.sqrt(dt * wvar(resid, wv))),
                r2=float(1 - resid.var() / yo.var()), n=int(ok.sum()),
                het=float(np.corrcoef(resid ** 2, own[ok])[0, 1]), resid=resid, ok=ok, n_par=X.shape[1])


def theta_at(eq, phase):
    phase = np.atleast_1d(phase)
    if eq["mod_own"]:
        return hvec(phase, True) @ eq["theta_coef"]
    return np.full(phase.size, eq["theta_coef"][0])


def fit(df, years, tail="linear", knots=None, year_weights=None, diurnal_mod=True, colored=True, cache=None,
        solvers=None, rho=None):
    """Fit both equations on the summers in `years`.  year_weights: dict year -> multiplicity."""
    d, dT, dS, dt = increments(df, years)
    if knots is None:
        knots = {k: make_knots(d[k].values) for k in ("S", "Z", "H", "T")}
    w = None
    if year_weights is not None:
        w = np.array([year_weights.get(y, 0) for y in d.index.year], float)
    hp, dp = d["hour_phase"].values, d["doy_phase"].values
    mod = ("S", "H") if diurnal_mod else ()
    eqT = fit_equation(dT, d["T"].values, {"S": d["S"].values, "Z": d["Z"].values, "H": d["H"].values},
                       knots, hp, dp, dt, tail, modulated=mod, mod_own=diurnal_mod,
                       sigma_by_hour=diurnal_mod, weights=w, times=d.index if (colored and rho is None) else None,
                       cache=None if cache is None else cache.setdefault("T", {}),
                       solver=None if solvers is None else solvers["T"])
    if rho is not None:
        eqT["rho"] = rho
    eqS = fit_equation(dS, d["S"].values, {"T": d["T"].values, "Z": d["Z"].values},
                       knots, hp, dp, dt, tail, weights=w,
                       cache=None if cache is None else cache.setdefault("S", {}),
                       solver=None if solvers is None else solvers["S"])
    return dict(T=eqT, S=eqS, knots=knots, dt=dt, tail=tail, d=d, diurnal_mod=diurnal_mod, colored=colored)


def noise_corr(m, n):
    """Correlation matrix of the standardised temperature noise over n steps (identity if white)."""
    rho = m["T"]["rho"]
    if rho is None:
        return np.eye(n)
    R = rho[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    lam, U = np.linalg.eigh(R)
    lam = np.clip(lam, 1e-10, None)                  # nearest positive semi-definite
    R = (U * lam) @ U.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def color(m, white):
    """Map white standard normals (n, N) to noise with the fitted residual autocorrelation."""
    n = white.shape[0]
    if m["T"]["rho"] is None:
        return white
    L = np.linalg.cholesky(noise_corr(m, n) + 1e-12 * np.eye(n))
    return L @ white


# ------------------------------------------------------------------ event window
def window(df, t0, t1):
    w = (df.index >= pd.Timestamp(t0)) & (df.index <= pd.Timestamp(t1))
    return df[w]


def step_coefs(m, e):
    """Per-step quantities shared by every path: exogenous forcing, relaxation rate, noise amplitude."""
    eT, eS = m["T"], m["S"]
    hp, dp = e["hour_phase"].values, e["doy_phase"].values
    xT = (eT["intercept"] + eT["comps"]["Z"].along(e["Z"].values, hp) + eT["comps"]["H"].along(e["H"].values, hp)
          + harmonics(hp, N_DIURNAL) @ eT["diurnal"] + harmonics(dp, N_SEASONAL) @ eT["seasonal"])
    xS = (eS["intercept"] + eS["comps"]["Z"].along(e["Z"].values, hp)
          + harmonics(hp, N_DIURNAL) @ eS["diurnal"] + harmonics(dp, N_SEASONAL) @ eS["seasonal"])
    th = theta_at(eT, hp)
    hr = e.index.hour.values
    sg = eT["sigma_h"][hr]
    return xT, xS, th, sg


def soil_forcing(m, e, S):
    """c_S along the window: S has shape (n+1, N) or (n+1,)."""
    cS, hp = m["T"]["comps"]["S"], e["hour_phase"].values
    S = np.asarray(S, float)
    return np.array([cS(S[k], hp[k]) for k in range(len(hp))])


def simulate(m, e, T0, S0, noise, cS_ref=None, eps=0.0, S_path=None):
    """
    Integrate the fitted discrete model through window e (n = len(e)-1 hourly steps).
    noise: array (n, 2, N).  The soil forcing on T is c_S(S,h) + eps*(cS_ref - c_S(S,h)).
    S_path: (n+1, N) soil paths to impose instead of simulating soil (feedback off),
            or (n+1,) a single prescribed observed path.
    Returns T (n+1, N), S (n+1, N), cS (n+1, N) = c_S actually applied.
    """
    eT, eS, dt = m["T"], m["S"], m["dt"]
    n, N = noise.shape[0], noise.shape[2]
    xT, xS, th, sg = step_coefs(m, e)
    hp = e["hour_phase"].values
    cSf, dTf = eT["comps"]["S"], eS["comps"]["T"]
    T = np.full(N, float(T0)); S = np.full(N, float(S0))
    Ts = np.empty((n + 1, N)); Ss = np.empty((n + 1, N)); Cs = np.empty((n + 1, N))
    sq = np.sqrt(dt)
    zT = color(m, noise[:, 0, :])
    for k in range(n + 1):
        if S_path is not None:
            S = S_path[k] if S_path.ndim == 2 else np.full(N, S_path[k])
        Ts[k], Ss[k] = T, S
        c = cSf(S, hp[k])
        if eps != 0.0:
            c = c + eps * (cS_ref[k] - c)
        Cs[k] = c
        if k == n:
            break
        fT = -th[k] * T + c + xT[k]
        if S_path is None:
            fS = -eS["theta"] * S + dTf(T) + xS[k]
            S = np.clip(S + fS * dt + eS["sigma"] * sq * noise[k, 1], 0.01, 0.6)
        T = T + fT * dt + sg[k] * sq * zT[k]
    return Ts, Ss, Cs


def terminal_law(m, e, T0, Cs):
    """Conditional on each forcing path: T_n ~ N(M, s0^2) under the discrete model."""
    dt = m["dt"]
    xT, _, th, sg = step_coefs(m, e)
    n = Cs.shape[0] - 1
    q = 1 - th[:n] * dt
    after = np.ones(n)                                       # prod_{j>k} q_j
    for k in range(n - 2, -1, -1):
        after[k] = after[k + 1] * q[k + 1]
    w = after * dt
    M = np.prod(q) * T0 + w @ (Cs[:n] + xT[:n, None])
    a = sg[:n] * after
    s0 = np.sqrt(dt * a @ noise_corr(m, n) @ a)
    return M, s0, w


def adjoint(m, e, T0, Cs, dF, u):
    """
    First-order response A = dP_u/d eps of Eq. (response) with the semi-analytic backward
    solution, evaluated along each soil path and integrated over the Gaussian conditional
    law of T:  per-path C_i = phi((u - M_i)/s0) D_i / s0,  D_i = sum_k w_k dF_ik.
    Also returns P (Rao-Blackwellised) and the Monte Carlo standard error of A/P from the
    influence variable (C - A)/P - A (Q - P)/P^2 of the ratio.
    """
    M, s0, w = terminal_law(m, e, T0, Cs)
    D = w @ dF[:-1]
    z = (u - M) / s0
    Q = norm.sf(z)
    C = norm.pdf(z) * D / s0
    P, A = Q.mean(), C.mean()
    infl = (C - A) / P - A * (Q - P) / P ** 2
    return dict(A=float(A), P_rb=float(P), dlogP=float(A / P),
                dlogP_se=float(infl.std(ddof=1) / np.sqrt(len(C))),
                D_mean=float(D.mean()), M_mean=float(M.mean()), M_sd=float(M.std()), s0=float(s0))


def exact_shift_law(m, e, T0, Cs, dF, u, eps=1.0):
    """Exceedance probability with soil paths held fixed and the soil forcing shifted by eps*dF."""
    M, s0, w = terminal_law(m, e, T0, Cs)
    D = w @ dF[:-1]
    return float(norm.sf((u - M - eps * D) / s0).mean())
