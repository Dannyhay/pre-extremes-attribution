"""Diagnostic: run the Monte-Carlo FTIR pipeline on a LINEAR system where the
exact Gaussian answer is available from ftir_linear.ftir()."""
import math
import numpy as np
from ftir_linear import ftir as ftir_exact, lkif_analytic, dirs as dirs_lin

rng = np.random.default_rng(11)

# linear testbed:  dT = [-aT*T + c*U]dt + sT dW ,  dU = -aU*U dt + sU dW
aT, aU = 0.25, 0.08
sT = np.sqrt(0.5) * 0.7
SD_U = 0.35
sU = SD_U * np.sqrt(2 * aU)
C = 1.3                      # coupling coefficient k_xy


def stationary(n=400_000, dt=0.05, burn=40_000):
    a_u = np.exp(-aU * dt); s_u = SD_U * np.sqrt(1 - a_u * a_u)
    a_t = np.exp(-aT * dt); s_t = sT * np.sqrt((1 - a_t * a_t) / (2 * aT))
    w = (1 - a_t) / aT
    U = 0.0; T = 0.0
    for _ in range(burn):
        U = a_u * U + s_u * rng.standard_normal()
        T = a_t * T + w * C * U + s_t * rng.standard_normal()
    Us = np.empty(n); Ts = np.empty(n)
    for i in range(n):
        U = a_u * U + s_u * rng.standard_normal()
        T = a_t * T + w * C * U + s_t * rng.standard_normal()
        Us[i] = U; Ts[i] = T
    return Us, Ts


def forward_I(U0, ts, dt, noise=None):
    ts = np.asarray(ts, float); nmax = int(round(ts.max() / dt))
    a_u = np.exp(-aU * dt); s_u = SD_U * np.sqrt(1 - a_u * a_u)
    decay = np.exp(-aT * dt)
    U = U0.copy(); gc = C * U; I = np.zeros_like(U)
    want = {int(round(t / dt)): k for k, t in enumerate(ts)}
    out = [None] * len(ts)
    for n in range(1, nmax + 1):
        U = a_u * U + s_u * (noise[:, n-1] if noise is not None
                             else rng.standard_normal(U.shape))
        gn = C * U
        I = decay * I + 0.5 * dt * (decay * gc + gn); gc = gn
        if n in want:
            out[want[n]] = I.copy()
    return out


def mix_logpdf(x, M, s):
    acc = np.zeros_like(x)
    for i in range(0, len(M), 40_000):
        z = (x[:, None] - M[None, i:i + 40_000]) / s
        acc += np.exp(-0.5 * z * z).sum(1)
    return np.log(np.maximum(acc / (s * np.sqrt(2 * np.pi)) / len(M), 1e-300))


def ftir_mc(ts, dt, n_paths=20_000, n_nodes=15, n_grid=420, cond_k=5000):
    Ust, Tst = stationary()
    a_t = np.exp(-aT * np.asarray(ts, float))
    s_t = sT * np.sqrt((1 - a_t ** 2) / (2 * aT))
    pick = rng.choice(len(Ust), n_paths, replace=False)
    nmax = int(round(np.asarray(ts, float).max() / dt))
    shared = rng.standard_normal((n_paths, nmax))   # common random numbers
    I_dd = forward_I(Ust[pick], ts, dt, shared); T0_dd = Tst[pick]
    hx, hw = np.polynomial.hermite_e.hermegauss(n_nodes)
    u0s = SD_U * hx; wts = hw / hw.sum()
    L = np.zeros(len(ts))
    for u0, wq in zip(u0s, wts):
        I_s = forward_I(np.full(n_paths, u0), ts, dt, shared)
        T0_star = T0_dd          # SAME draws -> the two mixtures agree at t=0
        sel = np.argsort(np.abs(Ust - u0))[:cond_k]
        T0_w = Tst[sel[rng.integers(0, cond_k, n_paths)]]
        for k in range(len(ts)):
            s = s_t[k]
            Ms = a_t[k] * T0_star + I_s[k]
            Mw = a_t[k] * T0_w + I_s[k]
            Md = a_t[k] * T0_dd + I_dd[k]
            lo = min(Ms.min(), Mw.min(), Md.min()) - 5 * s
            hi = max(Ms.max(), Mw.max(), Md.max()) + 5 * s
            x = np.linspace(lo, hi, n_grid)
            lps, lpd, lpw = mix_logpdf(x, Ms, s), mix_logpdf(x, Md, s), mix_logpdf(x, Mw, s)
            L[k] += wq * np.trapezoid(np.exp(lpw) * (lps - lpd), x)
    return L


def dirs(ts, L, deg=4):
    p = np.arange(1, deg + 1)
    X = ts[:, None] ** p[None, :] / np.array([math.factorial(int(q)) for q in p])
    c, *_ = np.linalg.lstsq(X, L, rcond=None)
    return c[0], c[1]


ts = np.round(np.linspace(0.008, 0.16, 12), 6)

L_ex, st = ftir_exact(aT, aU, C, 0.0, sT ** 2, sU ** 2, ts)
lk = lkif_analytic(C, st["Cxx"], st["Cxy"], st["Cyy"])
c1e, c2e = dirs_lin(lambda tt: ftir_exact(aT, aU, C, 0.0, sT ** 2, sU ** 2, tt)[0],
                    tmax=0.16, deg=4, npts=40)
print(f"EXACT   : r={st['r']:+.4f}  LKIF={lk:+.6f}   c1={c1e:+.6f}  c2={c2e:+.5f}")
print(f"          L(t) at the test grid: {np.array2string(L_ex, precision=5)}")

L_mc = ftir_mc(ts, dt=0.001)
c1m, c2m = dirs(ts, L_mc)
print(f"\nMONTE   : c1={c1m:+.6f}  c2={c2m:+.5f}")
print(f"          L(t) at the test grid: {np.array2string(L_mc, precision=5)}")
print(f"\nratio MC/exact :  c1 {c1m/c1e:+.4f}   c2 {c2m/c2e:+.4f}")
print(f"mean |L_mc - L_ex| / max|L_ex| = {np.abs(L_mc-L_ex).mean()/np.abs(L_ex).max():.4f}")
