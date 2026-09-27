"""
Faithful Python port of rcmi_toy_experiment_binning_only.m (D. Hagan, H-CEL).

Kept identical: the binning Renyi entropy estimator, the sum-form RCMI
    I_a(X;Y|Z) = H_a(X,Z) + H_a(Y,Z) - H_a(X,Y,Z) - H_a(Z),
the within-group (t, t+tau) triple construction, the circular-shift surrogate
with offset in [10, N-10], and the linear multivariate LKIF cofactor formula
with phase-randomised surrogates.

Added, for the stress test (all opt-in, none change the original numbers):
  * records the raw I_obs, and the surrogate mean/sd separately, so a peak in
    the z-score can be traced to signal or to collapse of the null spread;
  * records how often I_obs / I_null go negative (the .m assumes RCMI >= 0);
  * occupancy diagnostics (samples per occupied cell in 3-D);
  * a nonlinear conditional-mean information flow, so that the LKIF arm of the
    comparison is not restricted to the linear/Gaussian estimator.
"""
import numpy as np

# ---------------------------------------------------------------- estimators

def renyi_entropy_binning(X, alpha, n_bins):
    """Renyi entropy by equidistant binning. X is (N, d). Port of the .m."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    N, d = X.shape
    if N == 0:
        return np.nan
    idx = np.empty((N, d), dtype=np.int64)
    for k in range(d):
        col = X[:, k]
        lo, hi = col.min() - 1e-9, col.max() + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)
        b = np.digitize(col, edges) - 1
        idx[:, k] = np.clip(b, 0, n_bins - 1)
    lin = np.zeros(N, dtype=np.int64)
    for k in range(d):
        lin = lin * n_bins + idx[:, k]
    counts = np.bincount(lin, minlength=n_bins ** d).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    if abs(alpha - 1.0) < 1e-9:
        return float(-(p * np.log(p)).sum())
    return float(np.log((p ** alpha).sum()) / (1.0 - alpha))


def rcmi(X, Y, Z, alpha, n_bins):
    """Sum-form Renyi CMI, as in the .m (Palus et al. Eq. 7)."""
    H_XZ = renyi_entropy_binning(np.column_stack([X, Z]), alpha, n_bins)
    H_YZ = renyi_entropy_binning(np.column_stack([Y, Z]), alpha, n_bins)
    H_XYZ = renyi_entropy_binning(np.column_stack([X, Y, Z]), alpha, n_bins)
    H_Z = renyi_entropy_binning(Z.reshape(-1, 1), alpha, n_bins)
    return H_XZ + H_YZ - H_XYZ - H_Z


def occupancy(X, Y, Z, n_bins):
    """Samples per occupied 3-D cell, and occupied fraction."""
    M = np.column_stack([X, Y, Z]); N = len(X)
    idx = np.empty((N, 3), dtype=np.int64)
    for k in range(3):
        col = M[:, k]
        edges = np.linspace(col.min() - 1e-9, col.max() + 1e-9, n_bins + 1)
        idx[:, k] = np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)
    lin = (idx[:, 0] * n_bins + idx[:, 1]) * n_bins + idx[:, 2]
    occ = np.bincount(lin, minlength=n_bins ** 3)
    nz = (occ > 0).sum()
    return dict(n=N, cells=n_bins ** 3, occupied=int(nz),
                per_occupied=float(N / nz), frac_occupied=float(nz / n_bins ** 3))


# ------------------------------------------------------------------- triples

def build_triples(x_source, y, groups, tau):
    N = len(y)
    t = np.arange(N - tau)
    ok = groups[t] == groups[t + tau]
    i = t[ok]
    return x_source[i], y[i + tau], y[i]


def rcmi_with_surrogate(x_source, y, groups, tau, alpha, n_bins, n_surr, seed):
    rng = np.random.default_rng(seed)
    Xs, Yf, Yp = build_triples(x_source, y, groups, tau)
    if len(Xs) < 30:
        return dict(I_obs=np.nan, mu=np.nan, sd=np.nan, z=np.nan, p=np.nan,
                    n_neg_null=0, I_obs_neg=False)
    I_obs = rcmi(Xs, Yf, Yp, alpha, n_bins)
    N_src = len(x_source)
    I_null = np.empty(n_surr)
    for b in range(n_surr):
        shift = int(rng.integers(10, N_src - 10 + 1))
        xs, yf, yp = build_triples(np.roll(x_source, shift), y, groups, tau)
        I_null[b] = rcmi(xs, yf, yp, alpha, n_bins)
    mu, sd = I_null.mean(), I_null.std(ddof=0)
    sd_eff = max(sd, 1e-12)
    return dict(I_obs=I_obs, mu=float(mu), sd=float(sd),
                z=float((I_obs - mu) / sd_eff),
                p=float((I_null >= I_obs).mean()),
                n_neg_null=int((I_null < 0).sum()), I_obs_neg=bool(I_obs < 0))


def alpha_sweep(predictors, y, groups, lags, alphas, n_surr, n_bins):
    npred, nal = len(predictors), len(alphas)
    out = {k: np.full((npred, nal), np.nan) for k in
           ("I_obs", "mu", "sd", "z", "p", "n_neg_null")}
    for j in range(npred):
        for a in range(nal):
            r = rcmi_with_surrogate(predictors[j], y, groups, lags[j],
                                    alphas[a], n_bins, n_surr,
                                    seed=1000 * (j + 1) + 17 * (a + 1))
            for k in out:
                out[k][j, a] = r[k]
    return out


# ---------------------------------------------------------------- LKIF (port)

def _tendency(x, groups, dt=1.0):
    same = groups[:-1] == groups[1:]
    idx = np.flatnonzero(same)
    return (x[idx + 1] - x[idx]) / dt, idx


def lkif_normalized(X, target_idx, dt, groups):
    d = X.shape[1]
    dxt, valid = _tendency(X[:, target_idx], groups, dt)
    if len(dxt) < d + 1:
        return np.full(d, np.nan), np.full(d, np.nan)
    Xa = X[valid]
    P = np.cov(Xa, rowvar=False)
    P_ext = np.array([np.cov(Xa[:, k], dxt)[0, 1] for k in range(d)])
    detP = np.linalg.det(P)
    if abs(detP) < 1e-14:
        return np.full(d, np.nan), np.full(d, np.nan)
    T = np.zeros(d)
    for j in range(d):
        s = 0.0
        for k in range(d):
            minor = np.delete(np.delete(P, j, axis=0), k, axis=1)
            s += ((-1) ** (j + k)) * np.linalg.det(minor) * P_ext[k]
        T[j] = s / detP * P[target_idx, j] / P[target_idx, target_idx]
    a, *_ = np.linalg.lstsq(Xa, dxt, rcond=None)
    res = dxt - Xa @ a
    dof = max(Xa.shape[0] - Xa.shape[1], 1)
    h_noise = (res @ res / dof) * dt / (2 * P[target_idx, target_idx])
    Z = np.abs(T).sum() + abs(h_noise)
    tau = T / Z if Z > 1e-15 else np.zeros(d)
    return tau, T


def phase_randomize(x, rng):
    n = len(x)
    X = np.fft.fft(x)
    mag = np.abs(X)
    ph = np.zeros(n)
    if n % 2 == 0:
        npos = n // 2 - 1
        rp = 2 * np.pi * rng.random(npos)
        ph[1:n // 2] = rp
        ph[n // 2 + 1:] = -rp[::-1]
    else:
        npos = (n - 1) // 2
        rp = 2 * np.pi * rng.random(npos)
        ph[1:(n + 1) // 2] = rp
        ph[(n + 1) // 2:] = -rp[::-1]
    return np.real(np.fft.ifft(mag * np.exp(1j * ph)))


def lkif_for_predictors(predictors, y, groups, lags, n_surr=100):
    npred = len(predictors)
    N = len(y); max_lag = max(lags)
    t = np.arange(N - max_lag)
    idx = t[groups[t] == groups[t + max_lag]]
    Xf = np.zeros((len(idx), npred + 1))
    for j in range(npred):
        Xf[:, j] = predictors[j][idx + (max_lag - lags[j])]
    Xf[:, -1] = y[idx + max_lag]
    gf = groups[idx + max_lag]
    tgt = npred
    tau_o, T_o = lkif_normalized(Xf, tgt, 1.0, gf)
    taus, ps = np.full(npred, np.nan), np.full(npred, np.nan)
    for j in range(npred):
        rng = np.random.default_rng(7000 + j)
        Tn = np.empty(n_surr)
        for b in range(n_surr):
            Xb = Xf.copy()
            Xb[:, j] = phase_randomize(Xf[:, j], rng)
            _, Tb = lkif_normalized(Xb, tgt, 1.0, gf)
            Tn[b] = Tb[j]
        taus[j] = tau_o[j]
        ps[j] = float((np.abs(Tn) >= abs(T_o[j])).mean())
    return dict(tau=taus, p=ps)


# ------------------------------- nonlinear conditional-mean flow (added) -----
# Fair-comparison arm: the CSM-style object, T = E_rho[d_y j] with
# j(y) = E[X_src | Y = y], estimated nonparametrically.  Unlike the linear
# cofactor LKIF this can represent a sigmoid or threshold response.

def nonlinear_flow(x_source, y, groups, tau, n_knots=12, n_surr=100, seed=0):
    """E_rho[ d_y j ] with j(y)=E[x_src|Y_future=y] from a tent-basis fit,
    plus a circular-shift surrogate test (same null as the RCMI arm)."""
    rng = np.random.default_rng(seed)

    def stat(xs):
        Xs, Yf, _ = build_triples(xs, y, groups, tau)
        knots = np.quantile(Yf, np.linspace(0, 1, n_knots))
        knots = np.unique(knots)
        if len(knots) < 4:
            return np.nan
        B = np.maximum(0.0, 1.0 - np.abs(
            (Yf[:, None] - knots[None, :]) / np.diff(knots).mean()))
        B = np.column_stack([np.ones(len(Yf)), B])
        coef, *_ = np.linalg.lstsq(B, Xs, rcond=None)
        grid = np.linspace(knots[0], knots[-1], 400)
        Bg = np.maximum(0.0, 1.0 - np.abs(
            (grid[:, None] - knots[None, :]) / np.diff(knots).mean()))
        Bg = np.column_stack([np.ones(len(grid)), Bg])
        jg = Bg @ coef
        dj = np.gradient(jg, grid)
        dens, edges = np.histogram(Yf, bins=60, density=True)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        w = np.interp(grid, ctr, dens, left=0, right=0)
        if np.trapezoid(w, grid) <= 0:
            return np.nan
        return float(np.trapezoid(dj * w, grid) / np.trapezoid(w, grid))

    obs = stat(x_source)
    null = np.array([stat(np.roll(x_source, int(rng.integers(10, len(x_source) - 9))))
                     for _ in range(n_surr)])
    sd = max(null.std(ddof=0), 1e-12)
    return dict(obs=obs, mu=float(null.mean()), sd=float(null.std(ddof=0)),
                z=float((obs - null.mean()) / sd),
                p=float((np.abs(null) >= abs(obs)).mean()))
