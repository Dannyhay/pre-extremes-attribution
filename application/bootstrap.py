"""
Uncertainty from fitting the generator (Sec. VII A): year-block bootstrap.

The baseline summers (event year excluded) are resampled with replacement as whole
years, both equations are refitted (knots and the residual autocorrelation held at
their full-sample values), and the soil ensemble, factual and counterfactual
probabilities and the first-order adjoint are recomputed on common random numbers.
The spread across replicates is the parameter uncertainty; the Monte Carlo error at
N = 20000 is an order of magnitude smaller.

usage: python bootstrap.py trappes|voronezh [--B 200] [--N 20000] [--tail linear]
"""
import argparse, json, numpy as np
import model as M
from attribute_event import SITES, event_setup


def gram_solver(X, y, ok, years_of_rows, w_year):
    """Weighted least squares from per-year Gram blocks: beta = (sum w G_y)^-1 sum w b_y."""
    Xo, yo, yr = X[ok], y[ok], years_of_rows[ok]
    G, b = {}, {}
    for yv in np.unique(yr):
        sel = yr == yv
        G[yv] = Xo[sel].T @ Xo[sel]; b[yv] = Xo[sel].T @ yo[sel]
    def make(weights):
        def solve(_Xo, _yo, _ok):
            A = sum(weights.get(k, 0) * G[k] for k in G); r = sum(weights.get(k, 0) * b[k] for k in b)
            lam = 1e-9 * np.trace(A) / A.shape[0]          # guards against knots unsupported in a resample
            return np.linalg.solve(A + lam * np.eye(A.shape[0]), r)
        return solve
    return make


def run(site, B=200, N=20000, tail="linear", seed=1):
    s = SITES[site]
    df = M.load(s["csv"])
    e, t1, years, sm_clim, _ = event_setup(df, site)
    cache = {}
    m0 = M.fit(df, years, tail=tail, cache=cache)
    d, dT, dS, dt = M.increments(df, years)
    yrs_rows = d.index.year.values
    mk = {"T": gram_solver(cache["T"]["X"], dT, m0["T"]["ok"], yrs_rows, None),
          "S": gram_solver(cache["S"]["X"], dS, m0["S"]["ok"], yrs_rows, None)}
    n = len(e) - 1
    noise = np.random.default_rng(0).standard_normal((n, 2, N))
    T0, S0 = float(e["T"].iloc[0]), float(e["S"].iloc[0])
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B + 1):
        if b == 0:
            wts = {y: 1 for y in years}
        else:
            draw = rng.choice(years, size=len(years), replace=True)
            wts = {y: int(np.sum(draw == y)) for y in years}
        m = M.fit(df, years, tail=tail, knots=m0["knots"], year_weights=wts, cache=cache,
                  solvers={"T": mk["T"](wts), "S": mk["S"](wts)}, rho=m0["T"]["rho"])
        Tf, Sf, Cf = M.simulate(m, e, T0, S0, noise)
        ref = M.soil_forcing(m, e, sm_clim)
        dF = ref[:, None] - Cf
        row = dict(rep=b, tau_T=1 / m["T"]["theta"], tau_S=1 / m["S"]["theta"], rr={}, rr_adj={}, P={})
        Mi, s0, w = M.terminal_law(m, e, T0, Cf)
        row["warming"] = float(-(w @ dF[:-1]).mean())
        for u in s["thresholds"]:
            a = M.adjoint(m, e, T0, Cf, dF, u)
            P = a["P_rb"]; Pc = M.exact_shift_law(m, e, T0, Cf, dF, u, 1.0)
            row["P"][str(u)] = P; row["rr"][str(u)] = P / Pc; row["rr_adj"][str(u)] = float(np.exp(-a["dlogP"]))
        out.append(row)
        if b % 20 == 0:
            print(site, b, {k: round(v, 3) for k, v in row["rr"].items()}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site"); ap.add_argument("--B", type=int, default=200); ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--tail", default="linear")
    a = ap.parse_args()
    out = run(a.site, a.B, a.N, a.tail)
    json.dump(out, open(f"results/bootstrap_{a.site}_{a.tail}.json", "w"))
    rr = {u: np.array([r["rr"][u] for r in out[1:]]) for u in out[0]["rr"]}
    for u, v in rr.items():
        print(f"u={u}: RR {out[0]['rr'][u]:.3f}  90% [{np.percentile(v, 5):.3f}, {np.percentile(v, 95):.3f}]")
    wv = np.array([r["warming"] for r in out[1:]])
    print(f"warming {out[0]['warming']:.2f} C  90% [{np.percentile(wv,5):.2f}, {np.percentile(wv,95):.2f}]")
