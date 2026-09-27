"""
Shared definitions for the soil-moisture--temperature testbed of Sec. VI A.

    dU = -theta_U [U - nu(t)] dt + sigma_U dW_U ,   S = logistic(U)
    dT = [-theta_T (T - T_ref) + g(S)] dt + sigma_T dW_T
    g(S) = 3 / (1 + exp[(S - 0.35)/0.055])

Conditional on a soil path, T is Gaussian with mean M(t) solving
dM/dt = -theta_T (M - T_ref) + g(S(t)) and, once started from stationarity,
constant variance s^2 = sigma_T^2 / (2 theta_T).  The target density is
therefore an exact Gaussian mixture over an ensemble of soil paths, and every
current, probability and adjoint below is evaluated from that mixture without
density estimation.
"""
import numpy as np
from scipy.special import ndtr

TH_U, TH_T = 0.08, 0.25
T_REF = 20.0
SIG_T = np.sqrt(0.5) * 0.7
SD_U = 0.35
SIG_U = SD_U * np.sqrt(2 * TH_U)
S_COND = SIG_T / np.sqrt(2 * TH_T)            # conditional sd of T given a soil path
SQ2PI = np.sqrt(2 * np.pi)

REGIMES = {"wet": 0.74, "transitional": 0.35, "very dry": 0.04}


def g(S):
    return 3.0 / (1.0 + np.exp((S - 0.35) / 0.055))


def logistic(u):
    return 1.0 / (1.0 + np.exp(-u))


def logit(p):
    return np.log(p / (1.0 - p))


def phi(z):
    return np.exp(-0.5 * z * z) / SQ2PI


def sf(z):
    """Standard normal survival function."""
    return ndtr(-z)


# ------------------------------------------------------------------ soil paths
def soil_paths(nu_of_t, t, n, rng, U0=None):
    """Exact OU transitions of U on the grid t for a time-varying equilibrium.

    nu_of_t(t) gives the equilibrium; within a step nu is held at the step's
    start value.  If U0 is None, paths start from the stationary law at nu(t[0]).
    Returns U with shape (len(t), n).
    """
    U = np.empty((len(t), n))
    U[0] = (nu_of_t(t[0]) + SD_U * rng.standard_normal(n)) if U0 is None else U0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        a = np.exp(-TH_U * h)
        nu = nu_of_t(t[k])
        U[k + 1] = nu + a * (U[k] - nu) + SD_U * np.sqrt(1 - a * a) * rng.standard_normal(n)
    return U


def conditional_mean(G, t, M0, forcing_ref=None, eps=0.0):
    """Integrate dM/dt = -theta_T (M - T_ref) + G + eps (ref - G) exactly
    for piecewise-linear forcing on the grid (trapezoid in the exponential kernel).

    G has shape (len(t), n); forcing_ref is None or shape (len(t),) or (len(t), n).
    """
    F = G if forcing_ref is None else G + eps * (forcing_ref - G)
    M = np.empty_like(F)
    M[0] = M0
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        a = np.exp(-TH_T * h)
        w = (1 - a) / TH_T
        M[k + 1] = T_REF + a * (M[k] - T_REF) + w * 0.5 * (F[k] + F[k + 1])
    return M


def stationary_mixture(S_star, n, rng, dt=0.05, burn=3000, run=1200):
    """Stationary (M_i, g_i) pairs for a fixed equilibrium nu = logit(S_star)."""
    nu = logit(S_star)
    U = nu + SD_U * rng.standard_normal(n)
    a = np.exp(-TH_U * dt)
    s_step = SD_U * np.sqrt(1 - a * a)
    for _ in range(burn):
        U = nu + a * (U - nu) + s_step * rng.standard_normal(n)
    decay = np.exp(-TH_T * dt)
    w = (1 - decay) / TH_T                     # exact kernel weight for dM/dt
    I = np.zeros(n)
    g_old = g(logistic(U))
    for _ in range(run):
        U = nu + a * (U - nu) + s_step * rng.standard_normal(n)
        g_new = g(logistic(U))
        I = decay * I + w * 0.5 * (g_old + g_new)
        g_old = g_new
    return T_REF + I, g_old, U


def mixture_fields(M, gs, y, chunk=20_000):
    """rho(y), j(y)=E[g|T=y], d_y j(y), d_y rho(y) from the Gaussian mixture."""
    A = np.zeros_like(y); B = np.zeros_like(y)
    Ap = np.zeros_like(y); Bp = np.zeros_like(y)
    for i in range(0, len(M), chunk):
        Mc, gc = M[i:i + chunk], gs[i:i + chunk]
        z = (y[:, None] - Mc[None, :]) / S_COND
        p = phi(z) / S_COND
        dp = p * (Mc[None, :] - y[:, None]) / S_COND ** 2
        B += p.sum(1); A += (p * gc[None, :]).sum(1)
        Bp += dp.sum(1); Ap += (dp * gc[None, :]).sum(1)
    n = len(M)
    rho = B / n
    j = A / np.maximum(B, 1e-300)
    dj = (Ap * B - A * Bp) / np.maximum(B, 1e-300) ** 2
    return rho, j, dj, Bp / n


def currents_at(u, M, gs):
    """J_self, J_src, J_diff and J_exc at the threshold u for one time slice."""
    p = phi((u - M) / S_COND) / S_COND
    rho = p.mean()
    drho = (p * (M - u) / S_COND ** 2).mean()
    J_self = -TH_T * (u - T_REF) * rho
    J_src = (p * gs).mean()
    J_diff = -0.5 * SIG_T ** 2 * drho
    J_mean = gs.mean() * rho
    return dict(rho=rho, self=J_self, src=J_src, diff=J_diff,
                mean=J_mean, exc=J_src - J_mean, total=J_self + J_src + J_diff)


def exceedance(u, M):
    return sf((u - M) / S_COND).mean(axis=-1)
