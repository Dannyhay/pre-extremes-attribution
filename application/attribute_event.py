"""
Attribution of an observed heatwave's exceedance probability to soil moisture
(Sec. VII A).  One fit, one factual ensemble and one counterfactual ensemble per
site; every threshold is evaluated on the same ensembles.

Estimand, for each threshold u:
    P_u = P( T(t1) > u | T(t0), S(t0) observed;  Z, H on [t0, t1] prescribed as observed )
under the fitted generator, with t1 the hour of the observed maximum over 1-15 August
(selected retrospectively and then held fixed).  The counterfactual replaces the fitted
soil forcing c_S(S_t) by c_S evaluated at the day-of-year soil-moisture climatology of
the baseline summers (event year excluded); the reference is fixed and does not depend
on eps.  Risk ratio RR = P_u / P_u^cf.

Outputs (JSON): fit summary, support diagnostics, residual autocorrelation, and per
threshold: direct counterfactual (full coupled model), exact soil-fixed values,
first-order adjoint, feedback check, flat-tail sensitivity and prescribed-soil variant.

usage: python attribute_event.py trappes|voronezh [--N 200000] [--seed 0]
"""
import argparse, json, numpy as np, pandas as pd
from scipy.stats import norm
import model as M

SITES = {
    "trappes":  dict(csv="site.csv", year=2003, t0="2003-08-01 00:00", search="2003-08-15 23:00",
                     thresholds=[31, 32, 33, 34, 35, 36, 37, 38, 39], lat=48.8, lon=2.0),
    "voronezh": dict(csv="site_2010.csv", year=2010, t0="2010-08-01 00:00", search="2010-08-15 23:00",
                     thresholds=[32, 33, 34, 35, 36, 37, 38, 39, 40], lat=51.7, lon=39.2),
}
BASELINE = range(1979, 2023)


def residual_acf(df, eq, d, lags_h=(1, 6, 24, 48, 120)):
    """Autocorrelation of the fitted T-equation residuals, pairing only rows exactly k hours apart."""
    r = pd.Series(np.nan, index=df.index)
    r.loc[d.index[eq["ok"]]] = eq["resid"]
    out = {}
    for k in lags_h:
        a, b = r.values, r.shift(-k).values
        ok = np.isfinite(a) & np.isfinite(b)
        out[str(k)] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    # squared residuals (conditional-variance memory)
    out_sq = {}
    for k in lags_h:
        a, b = r.values ** 2, r.shift(-k).values ** 2
        ok = np.isfinite(a) & np.isfinite(b)
        out_sq[str(k)] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out, out_sq


def event_setup(df, site):
    s = SITES[site]
    srch = M.window(df, s["t0"], s["search"])
    t1 = srch["T"].idxmax()
    e = M.window(df, s["t0"], t1)
    years = [y for y in BASELINE if y != s["year"]]
    clim_df = df[df.index.month.isin([6, 7, 8]) & df.index.year.isin(years)]
    clim_doy = clim_df.groupby(clim_df.index.dayofyear)["S"].mean()
    sm_clim = clim_doy.reindex(e.index.dayofyear).values
    # spread of soil moisture across baseline years at t0 (same calendar hour)
    at_t0 = clim_df[(clim_df.index.month == 8) & (clim_df.index.day == 1) & (clim_df.index.hour == 0)]["S"]
    return e, t1, years, sm_clim, at_t0


def run_site(site, N=200_000, seed=0, tail="linear", extras=True, diurnal_mod=True):
    s = SITES[site]
    df = M.load(s["csv"])
    e, t1, years, sm_clim, at_t0 = event_setup(df, site)
    m = M.fit(df, years, tail=tail, diurnal_mod=diurnal_mod)
    n = len(e) - 1
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 2, N))
    T0, S0 = float(e["T"].iloc[0]), float(e["S"].iloc[0])
    ref = M.soil_forcing(m, e, sm_clim)                            # fixed reference forcing, per step

    Tf, Sf, Cf = M.simulate(m, e, T0, S0, noise)                   # factual, full coupled model
    Tc, Sc, Cc = M.simulate(m, e, T0, S0, noise, cS_ref=ref, eps=1.0)       # counterfactual
    Tc_nf, _, _ = M.simulate(m, e, T0, S0, noise, cS_ref=ref, eps=1.0, S_path=Sf)  # soil held fixed
    dF = ref[:, None] - Cf                                         # perturbation field along paths

    kn = m["knots"]
    out = dict(site=site, year=s["year"], tail=tail, N=N, seed=seed,
               t0=str(e.index[0]), t1=str(t1), lead_days=float(n / 24), T_obs_t1=float(e["T"].iloc[-1]),
               fit=dict(theta_T=m["T"]["theta"], tau_T=1 / m["T"]["theta"], sigma_T=m["T"]["sigma"],
                        r2_T=m["T"]["r2"], het_T=m["T"]["het"], n_T=m["T"]["n"],
                        theta_S=m["S"]["theta"], tau_S=1 / m["S"]["theta"], sigma_S=m["S"]["sigma"],
                        r2_S=m["S"]["r2"], n_excluded_gaps=int(len(df[df.index.month.isin([6, 7, 8])
                                                                   & df.index.year.isin(years)]) - m["T"]["n"])),
               S0=S0, S0_minus_clim=float(S0 - sm_clim[0]),
               S0_std_anom=float((S0 - at_t0.mean()) / at_t0.std()),
               support=dict(
                   S_knots=[float(kn["S"][0]), float(kn["S"][-1])], S_train_min=float(m["d"]["S"].min()),
                   H_knots=[float(kn["H"][0]), float(kn["H"][-1])], H_train_max=float(m["d"]["H"].max()),
                   T_train_max=float(m["d"]["T"].max()),
                   frac_S_below_knots_factual=float(np.mean(Sf < kn["S"][0])),
                   frac_S_below_knots_clim=float(np.mean(sm_clim < kn["S"][0])),
                   frac_hours_H_above_knots=float(np.mean(e["H"].values > kn["H"][-1])),
                   frac_T_above_knots_factual=float(np.mean(Tf > kn["T"][-1])),
                   S_obs_range=[float(e["S"].min()), float(e["S"].max())],
                   H_obs_max=float(e["H"].max())),
               mean_warming_t1=float(Tf[-1].mean() - Tc[-1].mean()),
               feedback_max_dS=float(np.abs(Sc - Sf).max()),
               mean_forcing_shift=float(-dF[:-1].mean()))
    acf, acf_sq = residual_acf(df, m["T"], m["d"])
    out["resid_acf_T"], out["resid_acf_sq_T"] = acf, acf_sq

    rows = []
    for u in s["thresholds"]:
        P, Pc, Pc_nf = float((Tf[-1] > u).mean()), float((Tc[-1] > u).mean()), float((Tc_nf[-1] > u).mean())
        adj = M.adjoint(m, e, T0, Cf, dF, u)
        P_rb = adj["P_rb"]
        Pc_rb = M.exact_shift_law(m, e, T0, Cf, dF, u, 1.0)
        eps_sweep = {str(ep): float(np.log(M.exact_shift_law(m, e, T0, Cf, dF, u, ep)) - np.log(P_rb))
                     for ep in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)}
        rows.append(dict(
            u=u, P=P, P_cf=Pc, RR_direct=P / Pc, dlogP_direct=float(np.log(Pc / P)),
            P_cf_soil_fixed=Pc_nf, feedback_dlogP=float(np.log(Pc / P) - np.log(Pc_nf / P)),
            P_rb=P_rb, P_cf_rb=Pc_rb, RR_rb=P_rb / Pc_rb, dlogP_rb=float(np.log(Pc_rb / P_rb)),
            dlogP_adjoint=adj["dlogP"], dlogP_adjoint_se=adj["dlogP_se"], RR_adjoint=float(np.exp(-adj["dlogP"])),
            shortfall_direct=float(1 - adj["dlogP"] / np.log(Pc / P)),
            shortfall_rb=float(1 - adj["dlogP"] / np.log(Pc_rb / P_rb)),
            eps_sweep_dlogP=eps_sweep, D_mean=adj["D_mean"], M_mean=adj["M_mean"], M_sd=adj["M_sd"], s0=adj["s0"]))
    out["thresholds"] = rows
    # total terminal spread and mean shift, for the Gaussian mean-shift prediction
    Mf, s0, w = M.terminal_law(m, e, T0, Cf)
    out["terminal"] = dict(mean=float(Mf.mean()), sd_total=float(np.sqrt(Mf.var() + s0 ** 2)),
                           shift=float(-(w @ dF[:-1]).mean()), s0=float(s0))

    if extras:
        # prescribed observed soil path in both ensembles (exact given that single path)
        Cp = M.soil_forcing(m, e, e["S"].values)[:, None]
        dFp = ref[:, None] - Cp
        out["prescribed_soil"] = []
        for u in s["thresholds"]:
            Pp = M.exact_shift_law(m, e, T0, Cp, dFp, u, 0.0); Ppc = M.exact_shift_law(m, e, T0, Cp, dFp, u, 1.0)
            out["prescribed_soil"].append(dict(u=u, P=Pp, P_cf=Ppc, RR=Pp / Ppc))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site"); ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tail", default="linear")
    a = ap.parse_args()
    r = run_site(a.site, a.N, a.seed, a.tail)
    json.dump(r, open(f"results/attribution_{a.site}_{a.tail}.json", "w"), indent=1)
    f = r["fit"]
    print(f"{a.site} [{a.tail}] t1={r['t1']} lead={r['lead_days']:.2f} d  tau_T={f['tau_T']:.2f} d  tau_S={f['tau_S']:.0f} d  "
          f"R2={f['r2_T']:.3f}  het={f['het_T']:+.3f}  excluded gaps={f['n_excluded_gaps']}")
    print(f"  S0-clim={r['S0_minus_clim']:+.4f} ({r['S0_std_anom']:+.2f} sd)  warming at t1={r['mean_warming_t1']:.2f} C")
    print("  support:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["support"].items()})
    print("  resid acf:", {k: round(v, 3) for k, v in r["resid_acf_T"].items()})
    print("   u    P      P_cf    RR_dir  RR_rb   RR_adj  short_dir short_rb  fb_dlogP   presc_RR")
    for row, pr in zip(r["thresholds"], r.get("prescribed_soil", [{}] * 9)):
        print(f"  {row['u']}  {row['P']:.4f} {row['P_cf']:.4f}  {row['RR_direct']:.3f}  {row['RR_rb']:.3f}  "
              f"{row['RR_adjoint']:.3f}   {100*row['shortfall_direct']:5.1f}    {100*row['shortfall_rb']:5.1f}   "
              f"{row['feedback_dlogP']:+.4f}   {pr.get('RR', float('nan')):.3f}")
