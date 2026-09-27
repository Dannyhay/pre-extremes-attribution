"""
Event-conditional adjoint attribution of an exceedance probability to soil
moisture, in the style of Miralles et al. (2014) for the 2003 European heatwave.

INPUT  a CSV with an hourly (or 6-hourly) time index and columns
         T     2-m air temperature at the site        [degC]
         SM    root-zone soil moisture (daily is fine; will be interpolated)
         Z     a circulation control, e.g. Z500 anomaly or MSLP         [any unit]
         H     (optional) a boundary-layer heat-memory proxy, e.g. T850 [degC]
       Columns are standardised internally; the drift model is fitted on the
       JJA baseline years and the event window is attributed.

WHAT IS COMPUTED
  1. A joint SDE for (T, SM) with Z (and H, if present) exogenous:
         dT  = [-th_T T + c_S(SM) + c_Z(Z) + c_H(H) + diurnal + seasonal] dt + s_T dW
         dSM = [-th_S SM + d_T(T) + d_Z(Z) + seasonal]                 dt + s_S dW
     fitted from increments on the baseline summers with a tent basis.
  2. From the observed state at t0, an ensemble of N paths of (T, SM) is
     simulated under the fitted SDE with the OBSERVED exogenous forcing.
  3. The adjoint attribution of  log P(T(t1) > u)  to the state-dependent
     soil-moisture forcing, using the semi-analytic backward solution (T is
     conditionally linear given the SM, Z, H paths).  Two perturbations:
        excess : c_S(SM) -> ensemble-mean c_S           (conditional excess)
        clim   : c_S(SM) -> c_S(climatological SM)      (Miralles-style)
  4. A direct counterfactual on the same fitted model and the same noise, as a
     check on the first-order adjoint (they should agree to ~10-20%).

The error budget of the companion paper applies: ~15% under-attribution from
first order under correct specification, and upward bias under omitted
drivers or unmodelled slow memory.  The H column exists because multi-day
boundary-layer heat accumulation IS the hidden-slow-memory failure mode.
"""
import argparse, json, numpy as np, pandas as pd
from scipy.stats import norm


# ------------------------------------------------------------------ basis
def tent(x, knots):
    B = np.zeros(x.shape + (knots.size,))
    for j, c in enumerate(knots):
        lo = knots[j - 1] if j > 0 else c - (knots[1] - knots[0])
        hi = knots[j + 1] if j < knots.size - 1 else c + (knots[-1] - knots[-2])
        B[..., j] = np.clip(np.minimum((x - lo) / (c - lo), (hi - x) / (hi - c)), 0, None)
    return B[..., 1:]                                  # drop one: intercept is separate


def harmonics(phase, n):
    return np.column_stack([f(k * phase) for k in range(1, n + 1)
                            for f in (np.sin, np.cos)])


class Component:
    """
    A fitted piecewise-linear component c(x) with LINEAR TAILS: beyond the
    outermost knots it continues with the slope of the last interior segment.
    An event is by construction at the edge of the training support; a basis
    that returns zero there makes the forcing vanish exactly when it matters.
    """
    def __init__(self, knots, coef): self.knots, self.coef = knots, coef
    def __call__(self, x):
        x = np.asarray(x, float); k = self.knots
        xin = np.clip(x, k[0], k[-1])
        val = tent(xin, k) @ self.coef
        # tail slopes from the outermost interior segments
        dk = k[1] - k[0]
        s_hi = (tent(np.array([k[-1]]), k) @ self.coef - tent(np.array([k[-2]]), k) @ self.coef) / dk
        s_lo = (tent(np.array([k[1]]), k) @ self.coef - tent(np.array([k[0]]), k) @ self.coef) / dk
        return val + np.where(x > k[-1], (x - k[-1]) * s_hi, 0.0) \
                   + np.where(x < k[0], (x - k[0]) * s_lo, 0.0)


# ------------------------------------------------------------------ fitting
def fit_equation(target_incr, own, drivers, hour_phase, doy_phase, dt,
                 n_knots=14, n_diurnal=2, n_seasonal=2):
    """
    Regress d(target)/dt on [1, own, tent(driver_1), ..., diurnal, seasonal].
    Returns theta, sigma, dict of Component per driver, harmonic coefs, intercept.
    """
    cols = [np.ones_like(own), own]
    blocks, comps = {}, {}
    p = 2
    for name, x in drivers.items():
        kn = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), n_knots)
        B = tent(x, kn)
        blocks[name] = (p, p + B.shape[1], kn); cols.append(B); p += B.shape[1]
    Hd = harmonics(hour_phase, n_diurnal); cols.append(Hd); pd0, p = p, p + Hd.shape[1]
    Hs = harmonics(doy_phase, n_seasonal); cols.append(Hs); ps0, p = p, p + Hs.shape[1]
    X = np.column_stack(cols)
    ok = np.isfinite(X).all(1) & np.isfinite(target_incr)
    beta, *_ = np.linalg.lstsq(X[ok], target_incr[ok], rcond=None)
    resid = target_incr[ok] - X[ok] @ beta
    theta = -beta[1]
    sigma = np.sqrt(max(dt * resid.var(), 1e-12))
    for name, (i, j, kn) in blocks.items():
        comps[name] = Component(kn, beta[i:j])
    return dict(theta=float(theta), sigma=float(sigma), intercept=float(beta[0]),
                comps=comps, diurnal=beta[pd0:pd0 + Hd.shape[1]],
                seasonal=beta[ps0:ps0 + Hs.shape[1]],
                n_diurnal=n_diurnal, n_seasonal=n_seasonal,
                r2=float(1 - resid.var() / target_incr[ok].var()),
                n=int(ok.sum()), het=float(np.corrcoef(resid ** 2, own[ok])[0, 1]))


def forcing(eq, drivers_now, hour_phase, doy_phase):
    """Total non-relaxation drift c(.) at given driver values (vectorised over paths)."""
    out = eq["intercept"] + np.zeros_like(next(iter(drivers_now.values())))
    for name, comp in eq["comps"].items():
        out = out + comp(drivers_now[name])
    out = out + harmonics(np.atleast_1d(hour_phase), eq["n_diurnal"]) @ eq["diurnal"]
    out = out + harmonics(np.atleast_1d(doy_phase), eq["n_seasonal"]) @ eq["seasonal"]
    return out


# ------------------------------------------------------------------ event
def simulate_event(eqT, eqS, ev, N, rng, cS_override=None):
    lo, hi = ev["SM_bounds"]
    prescribed = ev.get("SM_obs") is not None
    """
    Simulate (T, SM) from the observed state at t0 with exogenous Z (and H)
    prescribed as observed.  cS_override(SM_path, k) -> replacement for c_S.
    Returns T (nt, N), SM (nt, N), and the per-step c_S actually applied.
    """
    nt, dt = len(ev["t"]), ev["dt"]
    T = np.full(N, ev["T0"])
    S = np.full(N, ev["SM_obs"][0] if prescribed else ev["SM0"])
    Tst = np.empty((nt, N)); Sst = np.empty((nt, N)); Cs = np.empty((nt, N))
    sq = np.sqrt(dt)
    for k in range(nt):
        Tst[k], Sst[k] = T, S
        ex = {"Z": np.full(N, ev["Z"][k])}
        if "H" in eqT["comps"]: ex["H"] = np.full(N, ev["H"][k])
        cS = eqT["comps"]["SM"](S) if cS_override is None else cS_override(S, k)
        Cs[k] = cS
        fT = (-eqT["theta"] * T + eqT["intercept"] + cS
              + sum(eqT["comps"][n](ex[n]) for n in ex)
              + harmonics(np.atleast_1d(ev["hour"][k]), eqT["n_diurnal"]) @ eqT["diurnal"]
              + harmonics(np.atleast_1d(ev["doy"][k]), eqT["n_seasonal"]) @ eqT["seasonal"])
        dS = {"T": T, "Z": ex["Z"]}
        fS = (-eqS["theta"] * S + forcing(eqS, dS, ev["hour"][k], ev["doy"][k]))
        if k < nt - 1:
            zT, zS = ev["noise"][k]
            T = T + fT * dt + eqT["sigma"] * sq * zT
            S = (np.full(N, ev["SM_obs"][k + 1]) if prescribed
                 else np.clip(S + fS * dt + eqS["sigma"] * sq * zS, lo, hi))
    return Tst, Sst, Cs


def adjoint_attribution(eqT, ev, Tst, Sst, Cs, u, dF):
    """
    int E_t[ dF * d_T psi ] dt  with the semi-analytic backward solution.
    The full non-relaxation forcing along each path is reconstructed so that
    the terminal law given the path is exact.
    """
    nt, N = Tst.shape; dt = ev["dt"]; th, sg = eqT["theta"], eqT["sigma"]
    # total forcing along each path (SM term + exogenous + harmonics)
    Ctot = np.empty((nt, N))
    for k in range(nt):
        ex = {"Z": np.full(N, ev["Z"][k])}
        if "H" in eqT["comps"]: ex["H"] = np.full(N, ev["H"][k])
        Ctot[k] = (eqT["intercept"] + Cs[k]
                   + sum(eqT["comps"][n](ex[n]) for n in ex)
                   + harmonics(np.atleast_1d(ev["hour"][k]), eqT["n_diurnal"]) @ eqT["diurnal"]
                   + harmonics(np.atleast_1d(ev["doy"][k]), eqT["n_seasonal"]) @ eqT["seasonal"])
    I = np.zeros((nt, N)); tk = ev["t"]
    for k in range(nt - 2, -1, -1):
        h = tk[k + 1] - tk[k]; ah = np.exp(-th * h)
        I[k] = np.exp(-th * (tk[-1] - tk[k + 1])) * (1 - ah) / th * 0.5 * (Ctot[k] + Ctot[k + 1]) + I[k + 1]
    per = np.zeros(N); w = np.gradient(tk)
    for k in range(nt):
        a = np.exp(-th * (tk[-1] - tk[k]))
        sd = np.sqrt(max(sg ** 2 * (1 - a ** 2) / (2 * th), 1e-12))
        M = Tst[k] * a + I[k]
        per += w[k] * dF[k] * norm.pdf((M - u) / sd) * a / sd
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(N))


# ------------------------------------------------------------------ main
def run(csv, t0, t1, u, baseline, exclude_event_year, N, seed, out, prescribe_sm=False):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv, parse_dates=["time"]).set_index("time").sort_index()
    df["SM"] = df["SM"].interpolate("time")
    dt_h = (df.index[1] - df.index[0]) / pd.Timedelta(hours=1)
    dt = dt_h / 24.0                                     # model time in days
    hour = 2 * np.pi * df.index.hour / 24.0
    doy = 2 * np.pi * df.index.dayofyear / 365.25

    # ---- baseline fit on JJA
    jja = df.index.month.isin([6, 7, 8]) & df.index.year.isin(baseline)
    if exclude_event_year:
        jja &= df.index.year != pd.Timestamp(t0).year
    d = df[jja]
    nxt = df.shift(-1)[jja]
    same = (nxt.index + pd.Timedelta(hours=dt_h) == nxt.index + pd.Timedelta(hours=dt_h))  # placeholder
    dT = (nxt["T"].values - d["T"].values) / dt
    dS = (nxt["SM"].values - d["SM"].values) / dt
    hp, dp = hour[jja], doy[jja]
    drvT = {"SM": d["SM"].values, "Z": d["Z"].values}
    if "H" in df: drvT["H"] = d["H"].values
    eqT = fit_equation(dT, d["T"].values, drvT, hp, dp, dt)
    eqS = fit_equation(dS, d["SM"].values, {"T": d["T"].values, "Z": d["Z"].values}, hp, dp, dt)
    tauT = 1 / eqT["theta"]

    # ---- event window
    w = (df.index >= pd.Timestamp(t0)) & (df.index <= pd.Timestamp(t1))
    t_peak = df.loc[w, "T"].idxmax()
    w = (df.index >= pd.Timestamp(t0)) & (df.index <= t_peak)
    e = df[w]
    print(f"[event]  window {pd.Timestamp(t0)} -> observed peak {t_peak} ({e['T'].iloc[-1]:.1f} degC)")
    ev = dict(t=np.arange(w.sum()) * dt, dt=dt, T0=e["T"].iloc[0], SM0=e["SM"].iloc[0],
              Z=e["Z"].values, H=e["H"].values if "H" in df else None,
              hour=hour[w], doy=doy[w],
              SM_bounds=(float(np.nanmin(d["SM"])), float(np.nanmax(d["SM"]))),
              SM_obs=e["SM"].values if prescribe_sm else None)
    ev["noise"] = rng.standard_normal((len(ev["t"]) - 1, 2, N))
    Tst, Sst, Cs = simulate_event(eqT, eqS, ev, N, rng)
    cS_fit = eqT["comps"]["SM"]
    print(f"[fit T]  theta={eqT['theta']:.3f}/day  tau={1/eqT['theta']:.2f} d  sigma={eqT['sigma']:.2f}  "
          f"R2={eqT['r2']:.3f}  n={eqT['n']}")
    print(f"[fit SM] theta={eqS['theta']:.4f}/day  tau={1/eqS['theta']:.1f} d  sigma={eqS['sigma']:.4f}")
    smq = np.nanpercentile(d["SM"], [0.5, 5, 50, 95])
    print("[c_S]    at SM pct 0.5/5/50/95 =", np.round(cS_fit(smq) - cS_fit(smq[2]), 2),
          " (relative to median SM)")
    print(f"[event]  T0={ev['T0']:.1f}  SM0={ev['SM0']:.3f}  obs T(t1)={e['T'].iloc[-1]:.1f}  "
          f"sim SM mean over window={Sst.mean():.3f}  sim T(t1) mean={Tst[-1].mean():.1f}  "
          f"sim T(t1) 99pct={np.percentile(Tst[-1],99):.1f}")
    P_base = float((Tst[-1] > u).mean())
    if P_base == 0.0:
        raise SystemExit(f"no exceedances of u={u} at t1 in the baseline ensemble; "
                         f"ensemble t1 mean {Tst[-1].mean():.1f}, 99th pct {np.percentile(Tst[-1],99):.1f}")
    P_obs_exceeded = bool(e["T"].iloc[-1] > u)

    # climatological SM on the event calendar days
    clim = df[df.index.month.isin([6, 7, 8]) & df.index.year.isin(baseline)]
    sm_clim_doy = clim.groupby(clim.index.dayofyear)["SM"].mean()
    sm_clim = sm_clim_doy.reindex(e.index.dayofyear).values

    results = dict(mode="prescribed SM" if prescribe_sm else "simulated SM",
                   theta_T=eqT["theta"], tau_T_days=tauT, sigma_T=eqT["sigma"],
                   r2_T=eqT["r2"], n_fit_T=eqT["n"], het_T=eqT["het"],
                   theta_S=eqS["theta"], tau_S_days=1 / eqS["theta"],
                   P_base=P_base, obs_exceeded=P_obs_exceeded, u=u,
                   SM0_vs_clim=float(ev["SM0"] - sm_clim[0]))
    cS = eqT["comps"]["SM"]
    for label, cs_cf in (("excess", lambda S, k: np.full_like(S, cS(S).mean())),
                         ("clim", lambda S, k: np.full_like(S, cS(sm_clim[k])))):
        dF = np.array([cs_cf(Sst[k], k) - Cs[k] for k in range(len(ev["t"]))])
        A, se = adjoint_attribution(eqT, ev, Tst, Sst, Cs, u, dF)
        # direct counterfactual on the same fitted model and the same noise
        Tcf, _, _ = simulate_event(eqT, eqS, ev, N, rng, cS_override=cs_cf)
        P_cf = float((Tcf[-1] > u).mean())
        results[label] = dict(
            A=A, A_se=se,
            dlogP_adjoint=A / P_base,
            dlogP_direct=float(np.log(max(P_cf, 1e-9)) - np.log(max(P_base, 1e-9))),
            P_cf=P_cf,
            dT_mean_direct=float(Tst[-1].mean() - Tcf[-1].mean()),
            # sign convention: the perturbation REMOVES the SM effect, so the
            # source's contribution is minus the response
            contribution_logP=-A / P_base,
            risk_ratio_adjoint=float(np.exp(A / P_base) ** -1),
            risk_ratio_direct=float(P_base / max(P_cf, 1e-9)))
    json.dump(results, open(out, "w"), indent=2, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--t0", default="2003-08-01 00:00")
    ap.add_argument("--t1", default="2003-08-14 23:00", help="end of search window for the peak"); ap.add_argument("--u", type=float, default=38.0)
    ap.add_argument("--baseline", default="1979-2022"); ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--include-event-year", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="attribution.json")
    ap.add_argument("--prescribe-sm", action="store_true",
                    help="use the observed SM path through the event instead of simulating it")
    a = ap.parse_args()
    y0, y1 = map(int, a.baseline.split("-"))
    r = run(a.csv, a.t0, a.t1, a.u, list(range(y0, y1 + 1)), not a.include_event_year,
            a.N, a.seed, a.out, a.prescribe_sm)
    print(json.dumps(r, indent=2, default=float))
