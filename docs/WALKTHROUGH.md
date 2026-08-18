# Walkthrough — the QP dispatch, the Elermore Vale model and the VPP layer, by hand

A guided tour of the three layers with snippets you can paste into a Python
session started at the repo root (`python`, or `ipython`). Every snippet was
run against the code as of 2026-08-18. Prerequisites: `data.csv` at the
root, `outputs/profiles/fit_profiles.csv` (from `python dispatch/osqp_daily.py`), and the
usual `pip install -r requirements.txt`. Figures land in
`outputs/figures/walkthrough/` (gitignored). Read `dispatch/FORMULATION.md` §2–4 next to
Part 1, `MODEL_VERIFICATION.md` next to Part 2, `VPP_EXTENSION.md` §2/§6
next to Part 3.

Sign conventions used everywhere: **b** = battery power (kW), b > 0
discharge, b < 0 charge; **net = load − pv**; grid flow **p = net − b**
(p < 0 = export). Half-hours: T = 48, DT = 0.5 h.

---

## Part 1 — the R15 QP: one customer, one day

### 1.1 Get one day of data (uses the vpp cache, ~1 s after the first run)

```python
import numpy as np, matplotlib.pyplot as plt
from paths import PROFILES, FIGURES, GLM_DIR, GLM_COMMON   # every location in the repo
from dispatch import osqp_daily as B         # the Part A scheduler
from vpp import vpp_common as vc             # imports dispatch.osqp_daily; caches cleaned day arrays

OUT = FIGURES / "walkthrough"; OUT.mkdir(parents=True, exist_ok=True)
day_arrays = vc.load_day_arrays()            # {customer: [(date, load, pv), ...]}
cust = sorted(day_arrays)[0]
date, load, pv = next(d for d in day_arrays[cust] if d[0] == "7-Jan-11")
net = load - pv
print(f"customer {cust}, {date}: load {load.sum()*B.DT:.1f} kWh, PV {pv.sum()*B.DT:.1f} kWh, "
      f"peak export {net.min():.2f} kW at {net.argmin()/2:.1f} h")
```

### 1.2 The weights and the QP (`build_H0_diag`, `solve_battery`)

```python
tariff = B.build_tariff()                     # 0.03 off-peak / 0.06 shoulder / 0.30 peak $/kWh
h0 = B.build_H0_diag(tariff)                  # tariff / min(tariff), clipped to [1, H_BAR]
print("h0 by tier:", sorted(set(h0)))         # -> [1.0, 2.0, 10.0]

b = B.solve_battery(load, pv, h0, B.E_MAX_DEFAULT)   # minimise sum_k h_k (net_k - b_k)^2
p = net - b
soc = 0.5 * B.E_MAX_DEFAULT - B.DT * np.cumsum(b)

# the three constraint families, checked by hand
print("rate   |b| <= 5 kW :", np.abs(b).max() <= B.P_MAX + 1e-6)
print("SOC in [0, 10] kWh :", soc.min() >= -1e-6 and soc.max() <= B.E_MAX_DEFAULT + 1e-6)
print("neutral sum b = 0  :", abs(b.sum()) < 1e-6)
print("objective sum h p^2: {:.2f}  (no battery: {:.2f})".format(
      (h0 * p**2).sum(), (h0 * net**2).sum()))
```

The objective is a *weighted flattening of the grid profile*, not dollars.
Look at what that does:

```python
hrs = np.arange(B.T) * B.DT
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(hrs, net, label="net = load - pv"); ax[0].plot(hrs, p, label="grid p = net - b")
ax[0].plot(hrs, b, label="battery b (+ discharge)"); ax[0].axhline(0, c="k", lw=.5); ax[0].legend()
ax[1].plot(hrs, soc, label="SOC kWh"); ax[1].step(hrs, tariff * 30, where="post", label="tariff x30")
ax[1].legend(); ax[1].set_xlabel("hour")
fig.savefig(OUT / "p1_one_day.png", dpi=110); plt.show()
```

Things to notice: p is flatter than net (that is the objective); charging
happens where h is smallest *relative to the flow*, i.e. midday PV gets
absorbed even in FiT mode where PV earns 40 c regardless — because the QP
sees kW², not cents; SOC ends where it started (Σb = 0).

### 1.3 The bill, and the heuristic that makes the surrogate chase dollars

```python
for mode in ("fit", "net"):
    base_cost = B.bill(load, pv, np.zeros(B.T), tariff, mode)
    cost = B.bill(load, pv, b, tariff, mode)
    print(f"{mode}: bill no-battery ${base_cost:.3f}  with QP(h0) ${cost:.3f}  savings ${base_cost-cost:.3f}")

# Algorithm 1 by hand: double the peak-tier weights and see whether the REAL bill improves
h_try = h0.copy(); h_try[h0 == h0.max()] *= 2
b_try = B.solve_battery(load, pv, h_try, B.E_MAX_DEFAULT)
print("peak weights x2 -> fit savings",
      round(B.bill(load, pv, np.zeros(B.T), tariff, "fit") - B.bill(load, pv, b_try, tariff, "fit"), 3))

# what optimise_H does automatically (greedy doubling by tier, keep if savings improve, <= 20 rounds)
h_best, b_best, s_best = B.optimise_H(load, pv, tariff, B.E_MAX_DEFAULT, "fit")
print("optimise_H: savings $%.3f, final weights by tier %s" % (s_best, sorted(set(h_best))))

# the same customer on a winter day, for contrast
_, load_w, pv_w = next(d for d in day_arrays[cust] if d[0] == "1-Jul-10")
h_w, b_w, s_w = B.optimise_H(load_w, pv_w, tariff, B.E_MAX_DEFAULT, "fit")
print("winter day 1-Jul-10: savings $%.3f, weights by tier %s, PV %.1f kWh" % (s_w, sorted(set(h_w)), pv_w.sum() * B.DT))
```

Expect a *negative* number for the January day (≈ −$1.26 with h₀, ≈ −$0.78
after the heuristic) and a positive one for the July day. That is not a
bug, it is the surrogate showing its seams: minimising Σ h p² charges the
battery midday to flatten the PV export, but in gross-FiT metering PV is
paid regardless and charging is bought at the shoulder rate on the load
meter, so on a sunny day the battery only *costs*. The heuristic can
re-weight tiers (watch the off-peak/shoulder weights climb towards
`H_BAR`) but it can never choose b = 0; annual savings are positive because
winter days dominate. `dispatch/FORMULATION.md` §9 lists this "surrogate objective,
not settlement dollars" gap.

`bill_topology1` (fit): PV credited flat at `FIT_RATE` on the gross meter,
`max(load − b, 0)` billed at TOU. `bill_topology2` (net): one meter on p,
imports at TOU, exports credited. Read both (`osqp_daily.py:389-408`) —
they are the whole economic model.

### 1.4 Where this becomes `outputs/profiles/*.csv`

`run_all` runs `simulate_day` (= `optimise_H` + p) for every customer-day in
a `multiprocessing.Pool` (per-customer jobs, persistent OSQP workspace per
worker), and `save_profiles` writes the long-format CSV every network
script reads: `customer, date, interval, hour, load_kw, pv_kw, battery_kw,
grid_kw, soc_kwh, daily_savings`. Check one row against what you just
computed:

```python
import pandas as pd
df = pd.read_csv(PROFILES / "fit_profiles.csv")
row = df[(df.customer == cust) & (df.date == "7-Jan-11")].sort_values("interval")
print("stored battery == optimise_H result:", np.allclose(row.battery_kw.values, b_best, atol=1e-4))
```

(If this prints False the profiles were generated with different
E_MAX/tariff settings than the defaults — regenerate with `python dispatch/osqp_daily.py`.)

### 1.5 The DOE extension (`osqp_daily_with_DOE.py`)

Same QP over `x = [b | c | s]` — battery, **curtailed PV** (0 ≤ c ≤ pv) and an
**import-shortfall** slack (s ≥ 0) — with `p = load − pv + c − b` and per-interval
bounds `doe_min ≤ p` (export cap, hard: curtailment always makes it feasible)
and `p ≤ doe_max + s` (import cap, soft: you cannot shed load, so the shortfall
is reported). All rows live in the OSQP workspace from setup and only their
bounds change per day — never a new `A` (see `tests/test_doe_constraints.py`
for the bug that taught us that). The h-scaled penalties on c and s exceed
any flattening gain, so relief is used only when the battery cannot comply.

```python
from dispatch import osqp_daily_with_DOE as D
dmin, dmax = D.generate_doe_envelope("conservative", base_export_limit=3.0)   # export cap 2.4 kW
r = D.solve_battery(load, pv, h0, D.E_MAX_DEFAULT, dmin, dmax)               # DispatchResult
print("export cap 2.4: min p no-battery / QP / QP+cap:", round(net.min(), 2), round(p.min(), 2),
      round((load - pv + r.curtail - r.b).min(), 2), "| curtailed kWh", round(r.curtail.sum() * B.DT, 3))
for cap in (0.4, 0.0):                                       # tighten until the battery cannot cope
    r2 = D.solve_battery(load, pv, h0, D.E_MAX_DEFAULT, -cap * np.ones(B.T), np.inf * np.ones(B.T))
    print(f"export cap {cap}: min p", round((load - pv + r2.curtail - r2.b).min(), 3), "| curtailed kWh",
          round(r2.curtail.sum() * B.DT, 3), "| SOC max", round((5 - B.DT * np.cumsum(r2.b)).max(), 2))
r3 = D.solve_battery(load, pv, h0, D.E_MAX_DEFAULT, *D.generate_doe_envelope("none", base_import_limit=1.0))
print("import cap 1.0: max p", round((load - pv - r3.b).max(), 3), "| shortfall kWh",
      round(r3.import_slack.sum() * B.DT, 3), "| feasible", r3.doe_feasible)
```

On this customer-day the QP already keeps export under 2.4 kW and the
battery alone can hold a 0.4 kW cap; curtailment appears only when the cap
goes to zero and the SOC headroom is exhausted (curtailment is a last
resort by construction). A 1 kW import cap is met by the battery unless the
evening load exceeds what it can supply — then `import_slack` says by how
much. `simulate_day()` wraps this with the heuristic and returns a
`DayResult` (`savings`, `p`, `curtail`, `import_slack`, `doe_compliant`).

---

## Part 2 — the Elermore Vale OpenDSS model

### 2.1 Build it and look inside (~1 s)

```python
from network import elermorevale_openDSS as ev
stats = ev.build_elermorevale(str(GLM_DIR), str(GLM_COMMON))   # full model: PV + batteries as Generators
print(stats)
ckt = ev.dss.ActiveCircuit
print("Lines", ckt.Lines.Count, "Loads", ckt.Loads.Count, "Transformers", ckt.Transformers.Count,
      "Generators", ckt.Generators.Count, "RegControls", ckt.RegControls.Count, "nodes", len(ckt.AllNodeNames))
```

The build order is the physical order: 132 kV `Vsource` → `TXZoneSub` →
`OLTC` autotransformer (+`RegControl`) → 11 kV lines (bare GLM lengths are
**feet**) → 23 distribution transformers → LV lines (phases from GLM) →
1,785 residential `Load`s (1-phase, on their GLM phase, kv 0.24, 3 kW
placeholder) → 155 PV + 40 batteries as `Generator`s. Ask OpenDSS about a
few of them:

```python
def q(prop):
    ev.dss.Text.Command = f"? {prop}"; return ev.dss.Text.Result
print("zone TX kvs      :", q("Transformer.TXZoneSub.kvs"))
print("HP00007159 kvs   :", q("Transformer.HP00007159GTX00000001_TX.kvs"), " <- primary 10.725 kV = +2.56 % boost tap")
print("HP00016304 kvs   :", q("Transformer.HP00016304GTX00000001_TX.kvs"))
print("a load           :", q("Load.load_8633731.bus1"), q("Load.load_8633731.kv"), q("Load.load_8633731.kw"))
ckt.Lines.First; print("first line       :", ckt.Lines.Name, ckt.Lines.Bus1, "->", ckt.Lines.Bus2,
      f"{ckt.Lines.Length:.1f} {['','mi','kft','km','m','ft','in','cm'][ckt.Lines.Units]}")
```

Snapshot solve and the sanity numbers the tests pin (`test_golden_snapshot_regression`
uses the load-only model at 1 kW/household):

```python
ev.build_elermorevale(str(GLM_DIR), str(GLM_COMMON), skip_generators=True)
ev.dss.Text.Command = "BatchEdit Load..* kW=1.0"
assert ev.solve_snapshot()
v = np.array(ckt.AllBusVmagPu); v = v[v > 0.01]
p_src, _ = ckt.TotalPower
print(f"source {-p_src:.1f} kW, losses {ckt.Losses[0]/1000:.2f} kW, V min/mean/max {v.min():.3f}/{v.mean():.3f}/{v.max():.3f}")
# expected ~1859.5 kW / 74.5 kW / 0.830 / 0.949 / 1.005
```

### 2.2 One profile-driven day, baseline vs QP (~30 s incl. CSV load)

```python
profiles = ev.load_profiles_from_csv(str(PROFILES / "fit_profiles.csv"))          # {customer: [day dicts]}
ev.build_elermorevale(str(GLM_DIR), str(GLM_COMMON), skip_generators=True)
lcm = ev.map_customers_to_network_loads(sorted(profiles), ev.get_network_load_names())  # 152 -> 1,785 round-robin
mon = ev.select_monitored_loads(lcm, n_monitors=100)                                    # every 18th load
day = ev.day_index_for_date(profiles, "2011-01-07")

base = ev.simulate_scenario(str(GLM_DIR), str(GLM_COMMON), lcm, mon, profiles, day, use_baseline=True)
qp   = ev.simulate_scenario(str(GLM_DIR), str(GLM_COMMON), lcm, mon, profiles, day, use_baseline=False)
for lbl, r in (("baseline", base), ("QP", qp)):
    print(f"{lbl:9s} Vmin {r['v_min_pu']:.3f} Vmax {r['v_max_pu']:.3f} over {r['n_over']:4d} under {r['n_under']:4d} "
          f"peak TX {np.max(np.abs(r['tx_p_kw'])):.0f} kW  losses {r['loss_kw']:.1f} kW")
```

What `simulate_scenario` did: rebuilt the circuit, attached a `LoadShape`
per load from its customer's `grid` (or `load − pv` for the baseline)
series, ran `Set mode=daily stepsize=30m number=48` + `Solve`, read the 100
monitors, and refused to continue if any of them read 0 V
(`assert_monitors_energised`). Now look at *when* things happen:

```python
V = np.array(list(qp["voltages"].values()))                 # (100 monitors, 48)
over_by_hour = (V > ev.V_UPPER_PU).sum(axis=0); under_by_hour = (V < ev.V_LOWER_PU).sum(axis=0)
print("QP over-voltage points by hour :", {h/2: int(n) for h, n in enumerate(over_by_hour) if n})
print("QP under-voltage points by hour:", {h/2: int(n) for h, n in enumerate(under_by_hour) if n})
ev.OUTPUT_DIR = str(OUT); import matplotlib; matplotlib.use("Agg", force=True)
ev.plot_voltage_envelope(base, qp, "2011-01-07")
ev.plot_voltage_heatmap(qp, "2011-01-07", title_prefix="QP: ")
```

Then run the attribution across the year (about a minute) and see the two
mechanisms `NETWORK_AWARE_DISPATCH.md` is built on:

```bash
python network/diagnostics/diag_violation_attribution.py --every 30
```

### 2.3 Where to read next

`MODEL_VERIFICATION.md` "Known defects" (each one is a lesson: feet vs
metres, L-N vs L-L kv, phantom phases, floating phases, dead circuits that
report Converged=True), then `validation/compare_voltages.py` for how the
GridLAB-D cross-check is joined per node-phase.

---

## Part 3 — the VPP layer: N households, one coupling constraint

### 3.1 The ensemble and the feeder envelope

We use the winter day and an **import-side** envelope, because that is where
the interesting coupling is on this dataset (Part 2 showed the QP's
synchronised 22:00 charging; here you see it at ensemble scale) and because
winter savings are positive, so the fairness metrics mean something.

```python
households, date_iso, tariff = vc.assemble_ensemble(day_arrays, n_households=8, date_iso="2010-07-01")
N = len(households)
B_unc = np.vstack([hh.b_uncoupled for hh in households])
agg_unc = vc.aggregate_pi(households, B_unc)                     # sum_i p_i, kW
print("uncoupled aggregate p: max %.2f kW at %.1f h  (min %.2f kW)" % (agg_unc.max(), agg_unc.argmax() / 2, agg_unc.min()))
print("uncoupled savings $/day:", vc.savings_vector(households, B_unc, tariff, "fit").round(2))

d_min, d_max = vc.feeder_envelope("static", N, export_limit_kw=np.inf, import_limit_kw=2.0)  # aggregate import cap 2 kW x N
print("import cap %.1f kW; uncoupled violation %s" % (d_max[0], vc.envelope_violation(agg_unc, d_min, d_max)))
```

Expect the uncoupled aggregate to peak at ~22.7 kW **at 22:00** — eight
copies of the same "charge at 5 kW when off-peak starts" decision — and to
breach a 16 kW cap by ~6.7 kW in four intervals. Each `HouseholdDay`
carries its own R15 weights `h` (from `optimise_H`, frozen so every method
minimises the *same* coupled objective) and its uncoupled dispatch (today's
behaviour). The coupling is `D_min,k ≤ Σᵢ pᵢ,k ≤ D_max,k`.

### 3.2 Method A — the stacked QP (ground truth)

```python
res = vc.solve_centralised(households, d_min, d_max)          # hard coupling rows
print(res.status, "vars", res.n_variables, "cons", res.n_constraints, "solve %.3f s" % res.solve_time)
agg = vc.aggregate_pi(households, res.B)
print("aggregate import now max %.2f kW (cap %.2f); violation %.3f kW" %
      (agg.max(), d_max[0], vc.envelope_violation(agg, d_min, d_max)["max_kw"]))
print("shadow price of the envelope (dual, non-zero where it binds):",
      {k / 2: round(float(y), 3) for k, y in enumerate(res.y_couple) if abs(y) > 1e-6})
sav_c = vc.savings_vector(households, res.B, tariff, "fit")
print("savings uncoupled $%.2f -> coordinated $%.2f per day for the eight; Jain %.2f" %
      (vc.savings_vector(households, B_unc, tariff, "fit").sum(), sav_c.sum(), vc.jain_index(sav_c)))
print("valid dispatches:", vc.validate_ensemble(households, res.B) == 0)
soft = vc.solve_centralised(households, d_min, d_max, soft=True)   # slack + penalty: never infeasible
print("soft mode slack (kWh):", round(float(soft.slack_up.sum() * vc.DT), 4))
```

`solve_centralised` stacks `build_constraints` blocks block-diagonally and
adds T coupling rows `Σᵢ bᵢ ∈ [agg_net − D_max, agg_net − D_min]`; the
duals of those rows are the marginal value of one more kW of envelope in
that half-hour (expect non-zero values only from 20:00 to 23:30 — the
evening peak plus the off-peak charging block; OSQP reports upper-bound
duals with a negative sign). Coordination costs the eight households
about $1.35/day here ($16.10 → $14.75) and keeps Jain at 0.83. Try
`import_limit_kw=1.5`: hard mode becomes infeasible, soft mode tells you
by how much.

### 3.3 Method D — sharing ADMM (the same answer, one household QP at a time)

```python
from vpp.sharing_admm import sharing_admm as admm
B_admm, hist, n_it = admm.run_admm(households, d_min, d_max, rho=50.0, iters=300, tol_kw=0.05)
gap = vc.objective_surrogate(households, B_admm) / vc.objective_surrogate(households, res.B) - 1
print(f"ADMM converged in {n_it} iterations; objective gap vs centralised {100*gap:.3f} %; "
      f"final primal residual {hist['r'][-1]:.4f} kW; aggregate violation "
      f"{vc.envelope_violation(vc.aggregate_pi(households, B_admm), d_min, d_max)['max_kw']:.3f} kW")
```

Each iteration: every household solves *its own* QP with `P + ρI` (set
once) and a `q`-only update carrying the average/dual terms
(`HouseholdSolver.solve(q_extra=...)`), then a scalar clip onto the
envelope, then a dual update. That is why the per-iteration cost is one
uncoupled QP per household and why the sparsity/warm start survive. Expect
~25 iterations and a gap within ±0.01 % (a slightly negative gap means the
0.05 kW residual tolerance let a hair of violation through — tighten
`tol_kw` and watch it go to zero). Try `rho=2` to see it crawl.

### 3.4 Method B — two-stage DOE allocation (deployed practice) and fairness

```python
from vpp.two_stage_doe_allocation import two_stage_doe_allocation as ts
for rule in ts.RULES:
    B_rule, curtail_kwh, n_failed = ts.run_rule(rule, households, d_min, d_max)
    gap = vc.objective_surrogate(households, B_rule) / vc.objective_surrogate(households, res.B) - 1
    sav = vc.savings_vector(households, B_rule, tariff, "fit")
    viol = vc.envelope_violation(vc.aggregate_pi(households, B_rule), d_min, d_max)["max_kw"]
    print(f"{rule:16s} gap {100*gap:6.1f} %  agg violation {viol:.2f} kW  savings ${sav.sum():.2f}  "
          f"Jain {vc.jain_index(sav):.2f}  Gini {vc.gini(sav):.2f}  infeasible households {n_failed}")
```

Stage 1 splits the feeder envelope into per-household envelopes by a rule;
stage 2 each household solves alone with DOE rows (`HouseholdSolver` clips
an envelope tighter than ±P_MAX and records the shortfall as
`doe_relax_kw`, i.e. curtailment). The gap to Method A is the price of not
coordinating — here ~335 %, i.e. giving each household a fixed 2 kW slice of
the evening window is far worse than letting them trade it — and the
savings/fairness columns say who paid ($14.75 → $8.73 per day for the
eight; Jain 0.83 → 0.52). You will also see `WARNING ... household 9: OSQP
status primal infeasible` lines: households whose 2 kW slice cannot even
cover their own evening load get a zero dispatch (`n_failed` counts them) —
which is exactly why the two-stage method needs curtailment or a smarter
allocation. The four rules differ only in how they split the *export*
side, so with an import cap they coincide; run the summer day
(`date_iso="2011-01-07"`, `export_limit_kw=0.05`, `import_limit_kw=np.inf`)
to see `prorata_surplus` beat the others.

### 3.5 The scripts and the end-to-end pipeline

```bash
python vpp/centralised_qp/centralised_qp.py --n-households 20 --scenario static --save
python vpp/sharing_admm/sharing_admm.py --n-households 20 --scenario static --rho 50 --save
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --n-households 20 --scenario tight_tou --save
python vpp/dual_decomposition/dual_decomposition.py --n-households 20 --save
python vpp/price_based_control/price_based_control.py --n-households 20 --save
python vpp/fcas_cooptimisation/fcas_cooptimisation.py --n-households 20 --save
python vpp/run_vpp_network.py admm --n-households 20 --scenario static     # solve -> export 3 CSVs -> Elermore Vale -> outputs/runs/<id>/
```

`run_vpp_network.py` is the join between Parts 1–3: `vpp_export` writes
no-battery / uncoupled / coupled dispatch CSVs in the Part 1 format, then
`elermorevale_openDSS.simulate_scenario` replays each on the feeder and the
report overlays measured feeder-head power on the envelope.

---

## Check your understanding

1. In fit mode PV earns 40 c/kWh whatever the battery does — so why does
   the QP still charge at midday? *(Because the objective is Σ h p², not the
   bill; the heuristic only re-weights tiers. Look at 1.2.)*
2. Why does `load_8633731` sit above 1.10 pu at 3 am with no PV? *(Its
   transformer HP00007159 has a +2.56 % boost tap encoded as
   `primary_voltage 10725`; on the 240 V base the LV no-load voltage is
   already ~1.07 pu. Look at 2.1.)*
3. Why is 82 % of the QP's under-voltage in 22:00–24:00? *(1,785 copies of
   the same QP start charging at 5 kW when the off-peak tariff begins.
   Look at 2.2, at `h0` in 1.2, and at the 22.7 kW ensemble peak in 3.1.)*
4. Why does one ADMM iteration cost exactly one uncoupled QP per
   household? *(`P + ρI` is set at setup; only `q` changes; the coupling is
   a scalar clip. Look at 3.3 and `HouseholdSolver`.)*
5. Why must DOE rows be pre-allocated in the OSQP workspace? *(`update()`
   cannot change the sparsity pattern and silently ignores `A=`. Look at
   1.5 and `tests/test_doe_constraints.py`.)*
