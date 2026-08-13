"""Throwaway diagnostic: is the QP dispatch 'flatness' expected behaviour?

Checks, in order:
  1. Shape statistics over ~30 customers x 1 year of fit_profiles.csv:
     does the GRID profile flatten (piecewise-constant within TOU tiers)
     while the BATTERY absorbs the net-load variability?
  2. Constraint verification: energy neutrality, rate limit, SOC bounds.
  3. Independent re-solve of sample days with CVXPY (different solver
     path) vs the repo's OSQP result.
  4. Heuristic weight inspection: do the h tiers stay distinct or
     saturate to uniform H_BAR (which would over-flatten)?
"""
import numpy as np
import pandas as pd

import osqp_daily as base

DT, T = base.DT, base.T
tariff = base.build_tariff()
tier_idx = {lvl: np.where(tariff == lvl)[0] for lvl in np.unique(tariff)}

# ---- 1+2: statistics over a sample of the exported profiles ----------
N_ROWS = 48 * 365 * 30          # ~30 customer-years
df = pd.read_csv("profiles/fit_profiles.csv", nrows=N_ROWS)
print(f"rows: {len(df)}, customers: {df['customer'].nunique()}")

recs = []
for (c, d), g in df.groupby(["customer", "date"]):
    g = g.sort_values("interval")
    if len(g) != T:
        continue
    b = g["battery_kw"].to_numpy()
    grid = g["grid_kw"].to_numpy()
    net = g["load_kw"].to_numpy() - g["pv_kw"].to_numpy()
    soc = g["soc_kwh"].to_numpy()
    free = np.abs(np.abs(b) - base.P_MAX) > 1e-6   # rate limit not binding
    tier_stds = []
    for idx in tier_idx.values():
        sel = idx[free[idx]]
        if len(sel) > 1:
            tier_stds.append(grid[sel].std())
    recs.append(dict(
        cust=c, b_std=b.std(), grid_std=grid.std(), net_std=net.std(),
        corr_b_net=np.corrcoef(b, net)[0, 1] if b.std() > 1e-9 else np.nan,
        max_tier_std=max(tier_stds) if tier_stds else 0.0,
        b_absmax=np.abs(b).max(),
        sum_b_kwh=b.sum() * DT,
        soc_min=soc.min(), soc_max=soc.max(),
    ))
r = pd.DataFrame(recs)
print("\nPer customer-day statistics (kW):")
print(r[["b_std", "grid_std", "net_std", "corr_b_net",
         "max_tier_std", "b_absmax"]].describe().round(3).to_string())
print("\nvariance reduction: median grid_std / net_std =",
      round((r.grid_std / r.net_std.clip(lower=1e-9)).median(), 3))
print("median corr(battery, net load) =", round(r.corr_b_net.median(), 3))
print("days with battery inactive (<1e-6 kW):",
      round((r.b_absmax < 1e-6).mean(), 4))
print("CONSTRAINTS: max |sum b|*DT =", round(r.sum_b_kwh.abs().max(), 6),
      "| rate max =", round(r.b_absmax.max(), 4),
      "| soc range = [", round(r.soc_min.min(), 4), ",",
      round(r.soc_max.max(), 4), "]")

# ---- 3+4: reference re-solve with CVXPY on sample days ----------------
import cvxpy as cp

rng = np.random.default_rng(0)
sample_keys = [tuple(x) for x in
               df[["customer", "date"]].drop_duplicates()
               .sample(5, random_state=0).to_numpy()]
print("\nCVXPY cross-check on 5 random customer-days:")
for c, d in sample_keys:
    g = df[(df["customer"] == c) & (df["date"] == d)].sort_values("interval")
    load = g["load_kw"].to_numpy()
    pv = g["pv_kw"].to_numpy()
    b_stored = g["battery_kw"].to_numpy()
    net = load - pv

    h, b_osqp, s = base.optimise_H(load, pv, tariff, base.E_MAX_DEFAULT,
                                   "fit")
    bv = cp.Variable(T)
    soc = 0.5 * base.E_MAX_DEFAULT - DT * cp.cumsum(bv)
    cons = [cp.abs(bv) <= base.P_MAX, soc >= 0,
            soc <= base.E_MAX_DEFAULT, cp.sum(bv) == 0]
    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(h, cp.square(net - bv)))), cons)
    prob.solve(solver=cp.CLARABEL)
    gap_solver = np.abs(b_osqp - bv.value).max()
    gap_stored = np.abs(b_osqp - b_stored).max()
    print(f"  cust {c} {d}: |osqp-cvxpy|max = {gap_solver:.2e}  "
          f"|osqp-stored|max = {gap_stored:.2e}  "
          f"h levels = {np.unique(h).astype(int).tolist()}")
