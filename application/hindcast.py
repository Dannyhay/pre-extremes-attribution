"""
Leave-one-year-out hindcasts of the fitted generator (Sec. VII A).

For every baseline year y (event year excluded), the generator is refitted without y
and run from the observed state at 00:00 UTC on six dates of summer y, with Z and H
prescribed as observed, to the event's lead time and hour of day (the same conditional
setting as the attribution).  The predictive law of T at the terminal hour is the
Gaussian mixture over simulated soil paths; the probability integral transform (PIT)
of the observed temperature, the coverage of central predictive intervals and the ratio
of root-mean-square error to predictive spread summarise calibration.  The same is done
for soil moisture from the simulated soil ensemble.

usage: python hindcast.py trappes|voronezh [--tail linear] [--N 2000]
"""
import argparse, json, numpy as np, pandas as pd
from scipy.stats import norm
import model as M
from attribute_event import SITES, BASELINE

STARTS = ["06-21", "07-01", "07-11", "07-21", "08-01", "08-11"]


def run(site, tail="linear", N=2000, diurnal_mod=True, colored=True):
    s = SITES[site]
    df = M.load(s["csv"])
    years = [y for y in BASELINE if y != s["year"]]
    lead_h = {"trappes": 5 * 24 + 15, "voronezh": 4 * 24 + 12}[site]
    rows = []
    for y in years:
        m = M.fit(df, [x for x in years if x != y], tail=tail, diurnal_mod=diurnal_mod, colored=colored)
        for st in STARTS:
            t0 = pd.Timestamp(f"{y}-{st} 00:00"); t1 = t0 + pd.Timedelta(hours=lead_h)
            e = M.window(df, t0, t1)
            if len(e) != lead_h + 1:
                continue
            noise = np.random.default_rng(y * 100 + STARTS.index(st)).standard_normal((lead_h, 2, N))
            T0, S0 = float(e["T"].iloc[0]), float(e["S"].iloc[0])
            Tf, Sf, Cf = M.simulate(m, e, T0, S0, noise)
            Mi, s0, _ = M.terminal_law(m, e, T0, Cf)
            Tobs, Sobs = float(e["T"].iloc[-1]), float(e["S"].iloc[-1])
            rows.append(dict(year=y, start=st, T_obs=Tobs, pit_T=float(norm.cdf((Tobs - Mi) / s0).mean()),
                             mean_T=float(Mi.mean()), sd_T=float(np.sqrt(Mi.var() + s0 ** 2)),
                             S_obs=Sobs, pit_S=float(np.mean(Sf[-1] < Sobs)),
                             mean_S=float(Sf[-1].mean()), sd_S=float(Sf[-1].std()), dS_obs=Sobs - S0))
    r = pd.DataFrame(rows)
    pit = r["pit_T"].values
    summ = dict(site=site, tail=tail, n=len(r),
                cover50_T=float(np.mean((pit > 0.25) & (pit < 0.75))),
                cover90_T=float(np.mean((pit > 0.05) & (pit < 0.95))),
                bias_T=float((r["mean_T"] - r["T_obs"]).mean()),
                rmse_over_spread_T=float(np.sqrt(((r["mean_T"] - r["T_obs"]) ** 2).mean()) / np.sqrt((r["sd_T"] ** 2).mean())),
                pit_hist_T=np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
                cover90_S=float(np.mean((r["pit_S"] > 0.05) & (r["pit_S"] < 0.95))),
                bias_S=float((r["mean_S"] - r["S_obs"]).mean()),
                rmse_over_spread_S=float(np.sqrt(((r["mean_S"] - r["S_obs"]) ** 2).mean()) / np.sqrt((r["sd_S"] ** 2).mean())),
                frac_rain_jumps=float(np.mean(r["dS_obs"] > 0.005)))
    return summ, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site"); ap.add_argument("--tail", default="linear"); ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--no-diurnal-mod", action="store_true"); ap.add_argument("--white", action="store_true")
    a = ap.parse_args()
    summ, r = run(a.site, a.tail, a.N, not a.no_diurnal_mod, not a.white)
    tag = f"{a.site}_{a.tail}" + ("_nodm" if a.no_diurnal_mod else "") + ("_white" if a.white else "")
    json.dump(dict(summary=summ, cases=r.to_dict(orient="records")), open(f"results/hindcast_{tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
