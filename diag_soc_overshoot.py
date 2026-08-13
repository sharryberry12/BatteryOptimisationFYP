"""Follow-up: how common/large are SOC bound violations in the stored
profiles, and does a fresh re-solve reproduce them (solver issue) or not
(stale export)?"""
import numpy as np
import pandas as pd

import osqp_daily as base

DT, T = base.DT, base.T
E = base.E_MAX_DEFAULT
tariff = base.build_tariff()

df = pd.read_csv("profiles/fit_profiles.csv", nrows=48 * 365 * 30)

viol = []
for (c, d), g in df.groupby(["customer", "date"]):
    g = g.sort_values("interval")
    if len(g) != T:
        continue
    soc = g["soc_kwh"].to_numpy()
    over = max(soc.max() - E, 0.0) + max(-soc.min(), 0.0)
    if over > 1e-6:
        viol.append((c, d, over,
                     g["load_kw"].to_numpy(), g["pv_kw"].to_numpy(),
                     g["battery_kw"].to_numpy()))

print(f"customer-days with SOC outside [0,{E}] by >1e-6 kWh: "
      f"{len(viol)} of {df.groupby(['customer','date']).ngroups}")
if viol:
    overs = np.array([v[2] for v in viol])
    print("overshoot kWh: median", round(np.median(overs), 4),
          "p95", round(np.percentile(overs, 95), 4),
          "max", round(overs.max(), 4))
    # re-solve the worst offenders fresh
    viol.sort(key=lambda v: -v[2])
    print("\nfresh re-solve of the 5 worst:")
    for c, d, over, load, pv, b_stored in viol[:5]:
        h, b_new, s = base.optimise_H(load, pv, tariff, E, "fit")
        soc_new = 0.5 * E - DT * np.cumsum(b_new)
        over_new = max(soc_new.max() - E, 0.0) + max(-soc_new.min(), 0.0)
        print(f"  cust {c} {d}: stored overshoot {over:.4f} kWh -> "
              f"fresh {over_new:.6f} kWh | "
              f"|b_new - b_stored|max {np.abs(b_new - b_stored).max():.4f}")
