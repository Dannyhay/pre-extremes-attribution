"""Threshold sweep for the two observed heatwaves, for the manuscript figure."""
import json, subprocess, sys, os

RUNS = [
    ("site.csv",      "trappes", "2003-08-01 00:00", "2003-08-14 23:00",
     [31, 32, 33, 34, 35, 36, 37]),
    ("site_2010.csv", "voronezh", "2010-08-01 00:00", "2010-08-11 23:00",
     [32, 33, 34, 35, 36, 37, 38]),
]
out = {}
for csv, key, t0, t1, us in RUNS:
    out[key] = []
    for u in us:
        tgt = f"sweep_{key}_u{u}.json"
        if not os.path.exists(tgt):
            cmd = [sys.executable, "attribute_event.py", csv, "--t0", t0, "--t1", t1,
                   "--u", str(u), "--N", "20000", "--seed", "0", "--out", tgt]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {key} u={u}: {r.stderr.strip().splitlines()[-1]}")
                continue
        d = json.load(open(tgt))
        out[key].append(d)
        print(f"{key} u={u:>2}  P_base={d['P_base']:.4f}  P_cf={d['clim']['P_cf']:.4f}  "
              f"RR_adj={d['clim']['risk_ratio_adjoint']:.3f}  "
              f"RR_dir={d['clim']['risk_ratio_direct']:.3f}  "
              f"obs_exc={d['obs_exceeded']}")
json.dump(out, open("sweep_all.json", "w"), indent=2)
print("\nwrote sweep_all.json")
