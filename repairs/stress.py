"""
Three stress tests.

(i)   Seed robustness of Experiment B as written.
(ii)  Two redesigned injectors that separate WHY a tail driver is detectable:
        B1  sign-following injection -> the mechanism lives in the conditional
            MEAN response, so a conditional-mean method should also see it;
        B2  random-sign injection    -> the conditional mean is unchanged and
            only the conditional VARIANCE responds.  A conditional-mean method
            (LKIF / CSM / the manuscript's d_y j) MUST fail by construction.
      If RCMI detects B2 and the mean-based methods do not, that is a genuine
      capability of RCMI that the escort theorem does not cover.
(iii) Dan's own binning RCMI estimator applied to the manuscript testbed, with
      a sample size large enough to remove the estimation objection, to see
      whether any alpha ranks the very dry regime above the transitional one.
"""
import json
import numpy as np
from rcmi import alpha_sweep, lkif_for_predictors, nonlinear_flow, occupancy

ALPHAS = np.array([0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95,
                   1.05, 1.20, 1.50, 1.80, 2.20, 2.60, 3.00])
out = {"alphas": ALPHAS.tolist()}


def ar(n, phi, sig, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sig * rng.standard_normal()
    return x


def make_B(seed, N=5000, thr=3.0, val=1.8, mode="sign"):
    rng = np.random.default_rng(seed)
    x1 = ar(N, 0.70, 1.0, rng); x2 = ar(N, 0.85, 0.5, rng)
    x3 = rng.standard_normal(N); x4 = rng.standard_normal(N)
    t = np.zeros(N)
    for k in range(4, N):
        t[k] = (0.60 * x1[k - 1] + 0.50 * x2[k - 4] + 0.40 * t[k - 1]
                + 0.70 * rng.standard_normal())
    sd0 = t.std()
    hit = np.abs(x4[:-1]) > thr
    idx = np.flatnonzero(hit) + 1
    if mode == "sign":                      # mean response present
        t[idx] = np.sign(x4[idx - 1]) * val
    elif mode == "random":                  # variance-only response
        t[idx] = rng.choice([-1.0, 1.0], size=len(idx)) * val
    return [x1, x2, x3, x4], t, len(idx), sd0


# ---------------------------------------------------- (i) seed robustness
print("=" * 72)
print("(i) Experiment B exactly as written, across 8 seeds")
print("=" * 72)
print("  seed   n_inj   best |z| for X4   alpha at best   X4 flagged by |z|>2?")
rows = []
for sd in range(8):
    pred, t, ninj, sd0 = make_B(1000 + sd)
    grp = np.ones(len(t), dtype=int)
    sw = alpha_sweep([pred[3]], t, grp, [1], ALPHAS, 100, 8)
    z = np.array(sw["z"])[0]
    k = int(np.argmax(np.abs(z)))
    rows.append(dict(seed=1000 + sd, n_inj=ninj, best_z=float(z[k]),
                     best_alpha=float(ALPHAS[k]), flagged=bool(abs(z[k]) > 2)))
    print(f"  {1000+sd}   {ninj:5d}      {z[k]:+7.2f}          {ALPHAS[k]:.2f}"
          f"           {'YES' if abs(z[k])>2 else 'no'}"
          f"{'  (WRONG SIGN)' if z[k] < 0 and abs(z[k])>2 else ''}")
out["seed_robustness"] = rows

# ------------------------------------- (ii) mean-response vs variance-only
print("\n" + "=" * 72)
print("(ii) Redesigned injectors, N=20000, |X4|>2, injected value = 4 sd")
print("=" * 72)
variants = {}
for mode in ("sign", "random"):
    pred, t, ninj, sd0 = make_B(7, N=20000, thr=2.0, val=4 * 1.55, mode=mode)
    grp = np.ones(len(t), dtype=int)
    names, lags = ["X1", "X2", "X3", "X4"], [1, 4, 1, 1]
    sw = alpha_sweep(pred, t, grp, lags, ALPHAS, 100, 8)
    lk = lkif_for_predictors(pred, t, grp, lags)
    nl = [nonlinear_flow(pred[j], t, grp, lags[j], seed=31 + j) for j in range(4)]
    z = np.array(sw["z"])
    lab = ("B1  sign-following  (mean response present)" if mode == "sign"
           else "B2  random-sign    (variance-only response)")
    print(f"\n  {lab}   injections={ninj} ({100*ninj/20000:.1f}%)")
    print("    pred " + "".join(f"{a:>7.2f}" for a in ALPHAS))
    for j, nm in enumerate(names):
        print(f"    {nm:4s} " + "".join(f"{v:>7.1f}" for v in z[j]))
    print("    linear LKIF tau:", [f"{v:+.4f}" for v in lk["tau"]],
          "p:", [f"{v:.3f}" for v in lk["p"]])
    print("    nonlinear flow z:", [f"{v:+.2f}" for v in [r['z'] for r in nl]],
          "p:", [f"{v:.3f}" for v in [r['p'] for r in nl]])
    variants[mode] = dict(names=names, n_inj=ninj, z=z.tolist(),
                          I_obs=sw["I_obs"].tolist(),
                          lkif_tau=lk["tau"].tolist(), lkif_p=lk["p"].tolist(),
                          nl_z=[r["z"] for r in nl], nl_p=[r["p"] for r in nl])
out["variants"] = variants

# -------------------------- (iii) RCMI estimator on the manuscript testbed
print("\n" + "=" * 72)
print("(iii) Dan's binning RCMI applied to the TV-CSM testbed, N=200000")
print("=" * 72)
TH_U, TH_T, T_REF = 0.08, 0.25, 20.0
SIG_T = np.sqrt(0.5) * 0.7
SD_U = 0.35
SIG_U = SD_U * np.sqrt(2 * TH_U)
g = lambda S: 3.0 / (1.0 + np.exp((S - 0.35) / 0.055))
logistic = lambda u: 1.0 / (1.0 + np.exp(-u))
logit = lambda p: np.log(p / (1 - p))

def simulate(S_star, N=200_000, dt=0.25, burn=4000, seed=5):
    rng = np.random.default_rng(seed)
    nu = logit(S_star)
    U, T = nu, T_REF + g(S_star) / TH_T
    aU = np.exp(-TH_U * dt); sU = SD_U * np.sqrt(1 - aU * aU)
    for _ in range(burn):
        U = nu + aU * (U - nu) + sU * rng.standard_normal()
        T += (-TH_T * (T - T_REF) + g(logistic(U))) * dt + SIG_T * np.sqrt(dt) * rng.standard_normal()
    Us = np.empty(N); Ts = np.empty(N)
    for i in range(N):
        U = nu + aU * (U - nu) + sU * rng.standard_normal()
        T += (-TH_T * (T - T_REF) + g(logistic(U))) * dt + SIG_T * np.sqrt(dt) * rng.standard_normal()
        Us[i] = U; Ts[i] = T
    return g(logistic(Us)), Ts

tb = {}
for name, S_star in [("wet", 0.74), ("transitional", 0.35), ("very dry", 0.04)]:
    X, T = simulate(S_star)
    grp = np.ones(len(T), dtype=int)
    sw = alpha_sweep([X], T, grp, [1], ALPHAS, 60, 8)
    tb[name] = dict(I_obs=np.array(sw["I_obs"])[0].tolist(),
                    z=np.array(sw["z"])[0].tolist(),
                    sd_T=float(T.std()), mean_T=float(T.mean()))
    print(f"\n  {name:13s} mean T={T.mean():6.2f}  sd T={T.std():.2f}")
    print("    alpha  " + "".join(f"{a:>9.2f}" for a in ALPHAS))
    print("    RCMI   " + "".join(f"{v:>9.4f}" for v in tb[name]["I_obs"]))

print("\n  RCMI ratio  very dry / transitional  (should exceed 1 if alpha")
print("  rescues the saturated regime; the LEVEL ratio max|J_src| is 7.8):")
vd = np.array(tb["very dry"]["I_obs"]); tr = np.array(tb["transitional"]["I_obs"])
print("         " + "".join(f"{v:>9.3f}" for v in vd / tr))
out["testbed_rcmi"] = tb

json.dump(out, open("stress_results.json", "w"))
print("\nwrote stress_results.json")
