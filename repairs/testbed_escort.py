"""
Escort-weighting test in the TV-CSM soil-moisture / temperature testbed.

Claim under test
----------------
For the source-additive drift class, the Renyi-alpha information flow is

    T^(alpha)_{X->Y} = \int  d_y j(y)  rho_alpha(y) dy ,   rho_alpha = rho^a / \int rho^a

i.e. the SAME local response d_y j, averaged against an escort density instead of
rho itself.  alpha = 1 recovers the Liang flow (manuscript Eq. 8).

Consequence: every member of the family differentiates j.  No alpha makes the
measure read the LEVEL j(u) at a threshold.

Testbed (manuscript Eq. 19):
    dU = -theta_U (U - nu) dt + sigma_U dW_U ,  S = logistic(U)
    dT = [-theta_T (T - T_ref) + g(S)] dt + sigma_T dW_T
    g(S) = 3 / (1 + exp((S - 0.35)/0.055))

Source-additive with X = g(S), b = 1, so
    j(y) = E[g(S) | T = y]
    d_y j = local response  (manuscript's D)

Representation: conditional on a soil path, T is Gaussian.  We therefore
represent rho as an exact Gaussian mixture over an ensemble of soil paths -
no density estimation anywhere, and the tail stays smooth.
"""
import json
import numpy as np

rng = np.random.default_rng(20260920)

# ---- testbed parameters (manuscript Sec. VI A) ----------------------------
TH_U, TH_T = 0.08, 0.25
T_REF = 20.0
SIG_T = np.sqrt(0.5) * 0.7
SD_U = 0.35
SIG_U = SD_U * np.sqrt(2 * TH_U)

S_COND = SIG_T / np.sqrt(2 * TH_T)          # stationary conditional sd of T

def g(S):
    return 3.0 / (1.0 + np.exp((S - 0.35) / 0.055))

def logistic(u):
    return 1.0 / (1.0 + np.exp(-u))

def logit(p):
    return np.log(p / (1 - p))

# three equilibria marked in manuscript Fig. 1(a)
REGIMES = {"wet": 0.74, "transitional": 0.35, "very dry": 0.04}


def simulate_mixture(S_star, n_paths=400_000, dt=0.25, burn=600, run=240):
    """Stationary Gaussian-mixture representation of (T, g(S)).

    Returns per-path conditional mean M_i of T and the concurrent source
    value g_i = g(S_i).  Conditional sd of T is S_COND for every path.
    """
    nu = logit(S_star)
    U = nu + SD_U * rng.standard_normal(n_paths)       # start at stationarity
    a = np.exp(-TH_U * dt)
    s_step = SD_U * np.sqrt(1 - a * a)
    for _ in range(burn):
        U = nu + a * (U - nu) + s_step * rng.standard_normal(n_paths)

    # I(t) = \int_{-inf}^{t} e^{-theta_T (t-s)} g(S_s) ds, built by recursion
    decay = np.exp(-TH_T * dt)
    I = np.zeros(n_paths)
    for _ in range(run):
        U = nu + a * (U - nu) + s_step * rng.standard_normal(n_paths)
        I = decay * I + g(logistic(U)) * dt
    M = T_REF + I
    return M, g(logistic(U))


def mixture_fields(M, gsrc, y, chunk=20_000):
    """rho(y), j(y) = E[g|T=y] and d_y j(y), exactly from the mixture."""
    A = np.zeros_like(y); B = np.zeros_like(y)
    Ap = np.zeros_like(y); Bp = np.zeros_like(y)
    for i in range(0, len(M), chunk):
        Mc, gc = M[i:i + chunk], gsrc[i:i + chunk]
        z = (y[:, None] - Mc[None, :]) / S_COND
        phi = np.exp(-0.5 * z * z) / (S_COND * np.sqrt(2 * np.pi))
        dphi = phi * (Mc[None, :] - y[:, None]) / S_COND**2
        B += phi.sum(1); A += (phi * gc[None, :]).sum(1)
        Bp += dphi.sum(1); Ap += (dphi * gc[None, :]).sum(1)

    rho = B / len(M)
    j = A / B
    dj = (Ap * B - A * Bp) / B**2
    return rho, j, dj


def renyi_flow(y, rho, dj, alpha):
    """T^(alpha) = \int dj * rho_alpha,  rho_alpha \propto rho^alpha."""
    w = np.exp(alpha * np.log(np.maximum(rho, 1e-300)))
    Z = np.trapezoid(w, y)
    if not np.isfinite(Z) or Z <= 0:
        return np.nan
    return float(np.trapezoid(dj * w / Z, y))


ALPHAS = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])

out = {"alphas": ALPHAS.tolist(), "regimes": {}}
print(f"conditional sd of T = {S_COND:.4f}\n")

for name, S_star in REGIMES.items():
    M, gs = simulate_mixture(S_star)
    lo, hi = M.min() - 6 * S_COND, M.max() + 6 * S_COND
    y = np.linspace(lo, hi, 4000)
    rho, j, dj = mixture_fields(M, gs, y)

    T_alpha = [renyi_flow(y, rho, dj, a) for a in ALPHAS]
    T_shannon = renyi_flow(y, rho, dj, 1.0)

    Jsrc = j * rho
    k = int(np.argmax(np.abs(Jsrc)))

    # escort spread: sd of rho_alpha, to show where the weight sits
    esc_sd = []
    for a in ALPHAS:
        w = np.exp(a * np.log(np.maximum(rho, 1e-300)))
        w = w / np.trapezoid(w, y)
        mu = np.trapezoid(y * w, y)
        esc_sd.append(float(np.sqrt(np.trapezoid((y - mu) ** 2 * w, y))))

    out["regimes"][name] = dict(
        S_star=S_star, T_alpha=T_alpha, T_shannon=T_shannon,
        max_Jsrc=float(np.abs(Jsrc).max()), j_at_maxJ=float(j[k]),
        y_mode=float(y[np.argmax(rho)]), rho_max=float(rho.max()),
        mean_T=float(np.trapezoid(y * rho, y)),
        sd_T=float(np.sqrt(np.trapezoid((y - np.trapezoid(y * rho, y))**2 * rho, y))),
        escort_sd=esc_sd,
        max_abs_dj=float(np.abs(dj).max()),
        y=y[::4].tolist(), rho=rho[::4].tolist(), j=j[::4].tolist(), dj=dj[::4].tolist(),
    )
    print(f"{name:13s} S*={S_star:.2f}  T^(1)={T_shannon:.6g}   "
          f"max|J_src|={np.abs(Jsrc).max():.4f}   mean T={out['regimes'][name]['mean_T']:.2f}  "
          f"sd T={out['regimes'][name]['sd_T']:.2f}")

print("\nmanuscript targets:  T^(1) ~ 9e-4 (wet), 0.2267 (transitional), 7.8e-5 (very dry)")
print("                     max|J_src| = 0.0030, 0.2243, 1.7029\n")

print("alpha sweep, T^(alpha):")
hdr = "  alpha " + "".join(f"{a:>10.2f}" for a in ALPHAS)
print(hdr)
for name in REGIMES:
    r = out["regimes"][name]
    print(f"  {name:11s}" + "".join(f"{v:>10.3g}" for v in r["T_alpha"]))

print("\nratio  very dry / transitional  (slope-based ranking):")
vd = np.array(out["regimes"]["very dry"]["T_alpha"])
tr = np.array(out["regimes"]["transitional"]["T_alpha"])
print("         " + "".join(f"{v:>10.3g}" for v in vd / tr))
lvl = out["regimes"]["very dry"]["max_Jsrc"] / out["regimes"]["transitional"]["max_Jsrc"]
print(f"\nsame ranking by LEVEL (max|J_src| dry / transitional) = {lvl:.2f}")

out["level_ratio_dry_over_trans"] = lvl
json.dump(out, open("escort/testbed_escort.json", "w"))
