# dispatch/ — Part A: QP battery scheduling for one household-day

Reproduces Ratnam, Weller & Kellett (2015, "[R15]") with OSQP: for every
clean Ausgrid customer and every day, a 48-interval quadratic program picks
the battery power `b` (kW, + discharge) that flattens the weighted grid
profile, and a greedy heuristic re-weights the tariff tiers so the surrogate
chases real bill savings. Two metering topologies: **fit** (gross feed-in
tariff) and **net** (net metering).

| File | What |
|---|---|
| [`osqp_daily.py`](osqp_daily.py) | The base scheduler: data load + [R17] cleaning, tariff, bills, `build_constraints`, `solve_battery` (persistent OSQP workspace), `optimise_H` (Algorithm 1), `run_all` (multiprocessing), `save_profiles`, paper figures. |
| [`osqp_daily_with_DOE.py`](osqp_daily_with_DOE.py) | Copy-extension (not an import) that adds **Dynamic Operating Envelopes**: per-interval bounds on the grid flow, PV **curtailment** and an **import-shortfall** slack (see below). Same CLI shape plus `--scenarios`, `--export-limit`, `--import-limit`. |
| [`FORMULATION.md`](FORMULATION.md) | Notation, the optimisation problem, the persistent-workspace/warm-start invariant, data conventions, testable invariants, modelling gaps. (This was the project's original CLAUDE.md; `vpp/VPP_EXTENSION.md`'s "§9/§11" cross-references point into it.) |
| `diagnostics/` | Throwaway diagnostics on the exported profiles: `diag_qp_flatness.py`, `diag_dispatch_anatomy.py`, `diag_soc_overshoot.py`. |

## The problem in one screen

Decision variable `b ∈ ℝ⁴⁸`; `net = load − pv`; grid flow `p = net − b`.

    minimise  Σₖ hₖ (netₖ − bₖ)²                     -- weighted flattening, NOT dollars
    s.t.      |bₖ| ≤ P_MAX (5 kW)
              0 ≤ soc₀ − DT·Σⱼ≤ₖ bⱼ ≤ E_MAX (10 kWh, soc₀ = 50 %)
              Σ b = 0                                -- end the day where it started

`h₀ = tariff / min(tariff)` (off-peak 1, shoulder 2, peak 10); the heuristic
doubles a tier's weights and keeps the change if the *bill* improves.
Bills: `bill_topology1` (fit) credits PV flat at 40 c on the gross meter and
charges `max(load − b, 0)` at TOU; `bill_topology2` (net) meters `p`.

The OSQP workspace is set up once per battery size and only `P`'s diagonal,
`q` and the bounds change per day (fixed sparsity, warm start) — see
FORMULATION.md §4. **Never pass a differently-shaped `A` to
`solver.update`**: osqp silently ignores it (that is how the DOE constraint
was a no-op before 2026-08-16).

## The DOE extension

`osqp_daily_with_DOE.py` solves over `x = [b | c | s]`:

    b  battery kW                     c  curtailed PV kW, 0 ≤ c ≤ pv     s  import shortfall kW ≥ 0
    p = load − pv + c − b
    minimise  Σ h p² + PENALTY_KW · Σ h (c + s)
    s.t.      battery rows on b;  doe_min ≤ p  (export cap, hard — always feasible via c);
              p ≤ doe_max + s  (import cap, soft — load cannot be shed, the shortfall is reported)

The h-scaled linear penalties exceed any flattening gain, so relief is used
only when the battery cannot meet the envelope; the bill charges curtailment
through `pv − c`. Without an envelope `c = s = 0` and the result equals
`osqp_daily.solve_battery` exactly (`tests/test_doe_constraints.py`).
`generate_doe_envelope(scenario, base_export_limit, base_import_limit)`
shapes the export side (`none / conservative / tight / rolling`) and adds a
flat import cap.

## Run

```bash
python dispatch/osqp_daily.py                        # both topologies, all customer-days (~minutes)
python dispatch/osqp_daily_with_DOE.py               # fit; none/conservative/tight; + comparison table
python dispatch/osqp_daily_with_DOE.py --scenarios conservative tight --no-compare
python dispatch/osqp_daily_with_DOE.py --scenarios conservative --export-limit 3 1.5 0.75 --no-compare
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 2 --no-compare      # import cap only
```

Inputs: `data/data.csv` (`paths.DATA_CSV`). Outputs → `outputs/profiles/`:

- `fit_profiles.csv`, `net_profiles.csv` — long format, one row per
  customer × date × interval: `customer, date (d-Mon-yy), interval 1..48,
  hour, load_kw, pv_kw, battery_kw, grid_kw, soc_kwh, daily_savings`.
- `fit_doe_<scenario>[_cap<kW>][_imp<kW>].csv` — same columns plus
  `doe_compliant, doe_slack_kw, curtail_kw, import_shortfall_kw`
  (`grid_kw = load − pv + curtail − battery`; `pv_kw` stays raw PV).
- `outputs/doe_scenario_comparison.csv` (unless `--no-compare`).

These CSVs are what `network/elermorevale_openDSS.py --profiles` replays.
Paper figures from `osqp_daily.py` are shown interactively (`plt.show()`).

Mean annual savings logged at the end: ~$430 (fit) / ~$190 (net) per
customer at 10 kWh; on sunny days the surrogate can *lose* money in fit mode
(FORMULATION.md §9, docs/WALKTHROUGH.md §1.3).

## Gotchas

- `osqp_daily_with_DOE.py` duplicates the loader, cleaning rules, constants
  and heuristic of `osqp_daily.py` — a fix in one usually needs mirroring in
  the other.
- Both scripts use `multiprocessing.Pool`; anything at module top level runs
  in every worker on Windows (spawn), so keep it side-effect free.
- The heuristic can only re-weight tiers; it cannot choose `b = 0`, so a
  day's savings can be negative.
