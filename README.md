# Battery Optimisation FYP

QP-based residential battery scheduling on the Ausgrid solar-home dataset, validated on an OpenDSS model of the real Elermore Vale 11 kV feeder (Wallsend, NSW).

The project reproduces the algorithm of Ratnam, Weller & Kellett (*Renewable Energy 75*, 2015) and assesses how the resulting battery dispatches behave when injected into a power-flow simulation of an actual Ausgrid feeder.

## References

- **[R15]** Ratnam, Weller & Kellett, *"An optimization-based approach to scheduling residential battery storage with solar PV: Assessing customer benefit,"* Renewable Energy 75, 2015 — the algorithm paper.
- **[R17]** Ratnam, Weller, Kellett & Murray, *"Residential load and rooftop PV generation: an Australian distribution network dataset,"* IJSE 36(8), 2017 — the dataset paper.
- **[OSQP]** Stellato et al., *"OSQP: An operator splitting solver for quadratic programs,"* Math. Prog. Comp. 12, 2020.

The papers themselves are not redistributed here (`*.pdf` is gitignored); see the publishers' pages.

## Repository layout

| Path | Purpose |
|------|---------|
| [osqp_daily.py](osqp_daily.py) | QP battery scheduler — implements [R15] Sections 4-5 with a persistent OSQP workspace. Produces the half-hourly dispatch profiles consumed by every network model. |
| [osqp_daily_with_DOE.py](osqp_daily_with_DOE.py) | Copy-extension of `osqp_daily.py` that adds Dynamic Operating Envelope rows (per-interval export/import bounds on the grid flow) to the QP. Writes `profiles/<mode>_doe_<scenario>.csv` in the same long format, so the network scripts can replay DOE-constrained dispatch. |
| [elermorevale_openDSS.py](elermorevale_openDSS.py) | Port of the Elermore Vale (Wallsend, NSW) GridLAB-D model to OpenDSS. Translates [Elermorevale/](Elermorevale/) and [common/Line Configs.glm](common/Line%20Configs.glm) at runtime — no static `.dss` files. |
| [elermorevale_gui.py](elermorevale_gui.py) | Self-contained HTML dashboard for the Elermore Vale network: static topology view + animated live-flow view with synced Plotly charts. |
| [Elermorevale/](Elermorevale/) | GridLAB-D source for the Elermore Vale 11 kV feeder (zone substation, 23 distribution transformers, 1,810 GLM `load` objects = 1,785 households + 25 BlueGen CHP units, 155 PV systems, 40 Redflow batteries). The CHP units are excluded from the OpenDSS model — see [MODEL_VERIFICATION.md](MODEL_VERIFICATION.md) known defect #6. |
| [common/](common/) | Shared GridLAB-D includes — line configurations, transformer configs, tariff schedules, GLM modules. |
| [vpp/](vpp/) | Part B VPP extension: multi-household coupling methods (centralised QP, two-stage DOE allocation, dual decomposition, sharing ADMM, price-based control, FCAS co-optimisation), one subfolder + README per approach. See [vpp/README.md](vpp/README.md) and the design docs [VPP_EXTENSION.md](VPP_EXTENSION.md) / [paper_context.md](paper_context.md). |
| [run_vpp_network.py](run_vpp_network.py) | End-to-end pipeline: solve any VPP method, export the dispatch, and validate it on the Elermore Vale OpenDSS model in one command. Design in [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md); artifacts land under `runs/`. |
| [peak_duty_analysis.py](peak_duty_analysis.py) | VPP-as-peaker study: finds the aggregate peak demand condition over the full dataset, sweeps firm-capacity thresholds (f × peak), and reports how often / how long / how large a battery fleet must discharge to cover the top slice — the duty-cycle argument for VPP capability. |
| [replay_peak_event.py](replay_peak_event.py) | Physical companion to the duty study: replays the worst exceedance event through the Elermore Vale OpenDSS model with an explicit peaker dispatch (fleet sized as in the duty analysis), measuring the feeder-head shave and voltage impact. Findings written up in [PEAK_DUTY_FINDINGS.md](PEAK_DUTY_FINDINGS.md). |
| [WALKTHROUGH.md](WALKTHROUGH.md) | Hands-on tour of the three layers with paste-able snippets: one customer-day through the QP and heuristic, the Elermore Vale build / one-day simulation / violation timing, and the VPP ensemble under a binding import envelope (centralised vs ADMM vs two-stage), plus check-your-understanding questions. |
| [NETWORK_AWARE_DISPATCH.md](NETWORK_AWARE_DISPATCH.md) | Can the batteries be scheduled to zero voltage violations? Full-year fit/net sweeps with over/under split, the zone-OLTC negative result, DOE export-cap sweeps (after fixing the never-enforced constraint), and the attribution of what remains (one boost-tap transformer; 22:00 synchronised charging). |
| [diag_violation_attribution.py](diag_violation_attribution.py) | Reproduces that attribution: baseline vs QP on every N-th day, violation-points split over/under, by hour, by feeder. |
| [tests/](tests/) | Verification suite (101 tests, ~7 s): the Elermore Vale GLM→OpenDSS translation (unit tests, source-vs-circuit invariants, physics sanity power flows — see [MODEL_VERIFICATION.md](MODEL_VERIFICATION.md) incl. known defects) plus `test_doe_constraints.py`, which pins that the DOE rows in `osqp_daily_with_DOE.py` actually bind. Run with `python -m pytest`. |
| [validation/](validation/) | Level 4 cross-validation against GridLAB-D: `gen_harness.py` strips the GLM sources into a solvable one-instant harness, `compare_voltages.py` joins both engines' solutions per node-phase. Result: voltages agree to ~1 % mean ([MODEL_VERIFICATION.md](MODEL_VERIFICATION.md)). |
| [profiles/](profiles/) | Output of `osqp_daily.py` (and `osqp_daily_with_DOE.py`): half-hourly load/PV/battery/grid time series in long-format CSV. `fit_profiles.csv` = topology 1 (gross FiT); `net_profiles.csv` = topology 2 (net metering); `fit_doe_<scenario>.csv` = DOE-constrained variants. |
| [figures/](figures/) | Generated plots — paper figures from the QP run, plus voltage envelopes / heatmaps / substation power curves from the network simulations (`figures/net/` for net-metering profiles, `*_oltc/` for runs with the zone regulator active, `figures/doe_*/` for DOE-constrained dispatch). |
| `data.csv` | Raw Ausgrid solar-home dataset (one-year window; cleaning is done in memory — there is no cleaned CSV on disk). Local only, gitignored. |

## Install

```bash
pip install -r requirements.txt
```

Tested on Python 3.13. Core deps: `numpy`, `pandas`, `cvxpy`, `osqp`, `scipy`, `matplotlib`, `dss-python`, `networkx`.

## Running the pipeline

The three steps below are independent — each takes the previous step's output as a CSV.

### 1. Battery dispatch (QP optimisation)

```bash
python osqp_daily.py
```

Reads `data.csv`, cleans to the ~144 valid customers (one-year window), runs the [R15] Algorithm 1 weighting heuristic + OSQP for every customer-day under both metering topologies (FiT and net), writes `profiles/fit_profiles.csv` and `profiles/net_profiles.csv`, and produces paper figures 2/5/6/7/8 in [figures/](figures/).

Mean annual savings are logged at the end (~$430 FiT, ~$190 net for typical capacity).

**DOE-constrained variant** — the same QP with per-interval bounds on the grid flow (a Dynamic Operating Envelope broadcast by the DNSP), for the network-aware dispatch experiment in [NETWORK_AWARE_DISPATCH.md](NETWORK_AWARE_DISPATCH.md):

```bash
# Default: fit mode, scenarios none/conservative/tight at 3 kW export headroom, plus comparison table
python osqp_daily_with_DOE.py

# Just the profiles the network scripts need, and an export-cap tightening sweep
python osqp_daily_with_DOE.py --scenarios conservative tight --no-compare
python osqp_daily_with_DOE.py --scenarios conservative --export-limit 3 1.5 0.75 --no-compare
```

Writes `profiles/fit_doe_<scenario>.csv` (`<scenario>_cap<kW>` when several export limits are given) and `doe_scenario_comparison.csv`.

### 2. Network validation — the Elermore Vale feeder

Plotting CLI: `--save`, `--output-dir`, `--full`, `--max-days`, `--summer-day`, `--winter-day`, plus `--oltc` (activate the zone-substation voltage regulator; outputs are redirected to `<output-dir>_oltc`).

```bash
# Snapshot only (no profiles, just builds + solves):
python elermorevale_openDSS.py

# Daily comparison (baseline vs QP) for representative summer/winter days:
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --save

# Full year sweep:
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --full --save

# Net-metering profiles into their own directory; same again with the zone OLTC active
python elermorevale_openDSS.py --profiles profiles/net_profiles.csv --save --output-dir figures/net
python elermorevale_openDSS.py --profiles profiles/net_profiles.csv --save --output-dir figures/net --oltc   # -> figures/net_oltc/
```

Each daily run produces:
- voltage envelope (min/max p.u. across all monitored loads vs hour)
- voltage heatmap per scenario + Δ heatmap (baseline − QP)
- substation transformer P/Q curves
- per-day summary table (V min/max, violation count, total losses), appended to `summaries.txt` with the network configuration (OLTC on/off) in each block's header

The full sweep writes `opendss_sweep_results.csv` (one row per day, with an `oltc` column) and `sweep_summary.png`. Every daily run asserts that all 100 voltage monitors stayed energised for all 48 intervals; a monitored load on a de-energised node-phase raises `DeadMonitorError` instead of silently reporting V min = 0 (see [MODEL_VERIFICATION.md](MODEL_VERIFICATION.md) known defects #2 and #6).

### 3. End-to-end VPP pipeline (Part B)

```bash
# One command: VPP solve -> dispatch export -> OpenDSS validation -> report
python run_vpp_network.py sharing_admm --n-households 20

# Method aliases: centralised, two_stage, dual, admm, price, fcas
python run_vpp_network.py centralised_qp --soft --scenario tight_tou

# Export only (stages 1-3), then re-run the network stage later:
python run_vpp_network.py admm --skip-network
python run_vpp_network.py resume --run-dir runs/sharing_admm_static_...
```

`--data` defaults to `data.csv` at the repo root; pass it explicitly to point at an alternate Ausgrid drop.

Each run writes `runs/<method>_<scenario>_<date>_<timestamp>/` containing the
three dispatch CSVs (no-battery / uncoupled / coupled), `manifest.json`
(full provenance: args, envelope, convergence, git SHA), `network_summary.csv`
and the three-way comparison figures — including measured feeder-head power
against the DOE envelope. See [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md).

### 4. Peak-finding / VPP duty-cycle analysis

```bash
# Full 3-year dataset, default 70% firm-capacity focus threshold:
python peak_duty_analysis.py --data data_3_years.csv --save

# Restrict to the Ratnam-clean customer set, or analyse gross load:
python peak_duty_analysis.py --data data_3_years.csv --clean --save
python peak_duty_analysis.py --data data_3_years.csv --gross --focus 0.8 --save
```

`data_3_years.csv` is the full three-year Ausgrid file (2010-07 → 2013-06, 300 customers) that [PEAK_DUTY_FINDINGS.md](PEAK_DUTY_FINDINGS.md) was produced from; like `data.csv` it is local-only and not redistributed. `peak_duty_analysis.py` and `replay_peak_event.py` default to that filename — pass `--data data.csv` to run on the one-year window instead.

Builds the aggregate half-hourly demand series (cached under `vpp/cache/`),
finds the peak demand condition, sweeps firm-capacity thresholds, and sizes
the battery fleet (households needed, power- vs energy-limited) plus its duty
cycle (hours/year, events/year, cycle-equivalents). Outputs land in
`figures/peak_duty/`: duration curve, peak-day profile, duty-cycle sweep,
event calendar, and `duty_cycle_summary.csv` / `events_focus.csv`.

### 5. Dashboard

```bash
# Topology only (no DSS solve required, instant):
python elermorevale_gui.py --open

# With baseline + QP simulation overlay:
python elermorevale_gui.py --simulate --day 190 --open
```

Writes `elermorevale_dashboard_v2.html` (override with `--output`) — a single-file dashboard with two tabs (static topology / animated live flow), play/pause/scrub controls, and synced charts. `--serve --port 8765` runs it behind an optional Flask backend instead.

## Implementation notes

- **GLM parsing** — `parse_glm()` in [elermorevale_openDSS.py](elermorevale_openDSS.py) walks the flat object syntax used by the Ausgrid GLM files. Numeric properties may carry units (e.g. `"11.59 m^2"`); `gfloat()` strips them before `float()` conversion.
- **Line impedances** — 11 kV configurations use z-matrix format (Ohm/mile); LV configs reference named conductors with per-mile resistance. Both are extracted at runtime by `extract_impedances()`.
- **Customer-to-load mapping** — the ~144 OSQP customers are spread evenly across the 1,785 residential network loads via `map_customers_to_network_loads()` (the 25 BlueGen CHP `load` objects in `Generators2.glm` are not households and are excluded from the model). Unmapped loads are zeroed during simulation so the QP-vs-baseline signal isn't drowned by a static background.
- **Per-unit base** — Australian residential loads in the GLM are declared at `nominal_voltage = 240 V`, so `V_NOM = 240.0` in the simulators. AS 60038 limits (`+10 % / −6 %`) are applied to the resulting per-unit voltages.
- **Substation power sign** — `collect_tx_power()` negates the monitor channels because OpenDSS monitors record power flowing *into* the monitored terminal; after the fix, positive = import from the grid as documented. Substation-power figures generated before this fix are mirror images of current output.
- **Convergence** — snapshot/daily solves run with `controlmode=off` and elevated iteration caps by default; `--oltc` switches the Elermore Vale daily solves to `controlmode=static` with the zone RegControl built. OpenDSS warning #485 (Max Control Iterations Exceeded) is downgraded to a logged warning since the power-flow result is still valid. `Converged=True` alone is not trusted: the snapshot path rejects an all-zero (dead-circuit) solution and the daily path asserts every monitor stayed energised.
- **Zone OLTC** — with the regulator active on the representative days the 11 kV bus stays within 0.9955–1.0004 pu, inside the ±1 % band, so the tap never moves and the results are identical to `controlmode=off`: the LV violations are downstream of the distribution transformers and out of the zone regulator's reach ([NETWORK_AWARE_DISPATCH.md](NETWORK_AWARE_DISPATCH.md)).

## Data files (not redistributed for size)

`data.csv` is the Ausgrid solar-home electricity dataset ([R17] supplementary data), one-year window (July 2010 – June 2011, 300 customers); the cleaning step is applied in memory and retains the ~144 customers that pass the [R17] Section 3 rules. `data_3_years.csv` (2010-2013) is the optional full file used by the peak-duty study.
