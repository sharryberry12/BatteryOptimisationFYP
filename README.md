# Battery Optimisation FYP

QP-based residential battery scheduling on the Ausgrid solar-home dataset, validated on three OpenDSS distribution-network models (synthetic LV feeder, IEEE 13-bus, and the real Elermore Vale 11 kV feeder).

The project reproduces the algorithm of Ratnam, Weller & Kellett (*Renewable Energy 75*, 2015) and assesses how the resulting battery dispatches behave when injected into a power-flow simulation of an actual Ausgrid feeder.

## References

- **[R15]** Ratnam, Weller & Kellett, *"An optimization-based approach to scheduling residential battery storage with solar PV: Assessing customer benefit,"* Renewable Energy 75, 2015 — the algorithm paper.
- **[R17]** Ratnam, Weller, Kellett & Murray, *"Residential load and rooftop PV generation: an Australian distribution network dataset,"* IJSE 36(8), 2017 — the dataset paper.
- **[OSQP]** Stellato et al., *"OSQP: An operator splitting solver for quadratic programs,"* Math. Prog. Comp. 12, 2020.

PDFs of [R15] and [R17] are checked in at the repo root.

## Repository layout

| Path | Purpose |
|------|---------|
| [osqp_daily.py](osqp_daily.py) | QP battery scheduler — implements [R15] Sections 4-5 with a persistent OSQP workspace. Produces the half-hourly dispatch profiles consumed by every network model. |
| [openDSS_LV_feeder_model.py](openDSS_LV_feeder_model.py) | Synthetic 10-node Australian LV feeder built in-Python via dss-python. Quick smoke-test of the dispatch profiles. |
| [ieee_13_bus_openDSS.py](ieee_13_bus_openDSS.py) | IEEE 13 Node Test Feeder with five LV stubs added. Standardised benchmark network. |
| [elermorevale_openDSS.py](elermorevale_openDSS.py) | Port of the Elermore Vale (Wallsend, NSW) GridLAB-D model to OpenDSS. Translates [Elermorevale/](Elermorevale/) and [common/Line Configs.glm](common/Line%20Configs.glm) at runtime — no static `.dss` files. |
| [elermorevale_gui.py](elermorevale_gui.py) | Self-contained HTML dashboard for the Elermore Vale network: static topology view + animated live-flow view with synced Plotly charts. |
| [Elermorevale/](Elermorevale/) | GridLAB-D source for the Elermore Vale 11 kV feeder (zone substation, 23 distribution transformers, ~1,810 loads, 155 PV systems, 40 Redflow batteries). |
| [common/](common/) | Shared GridLAB-D includes — line configurations, transformer configs, tariff schedules, GLM modules. |
| [vpp/](vpp/) | Part B VPP extension: multi-household coupling methods (centralised QP, two-stage DOE allocation, dual decomposition, sharing ADMM, price-based control, FCAS co-optimisation), one subfolder + README per approach. See [vpp/README.md](vpp/README.md) and the design docs [VPP_EXTENSION.md](VPP_EXTENSION.md) / [paper_context.md](paper_context.md). |
| [run_vpp_network.py](run_vpp_network.py) | End-to-end pipeline: solve any VPP method, export the dispatch, and validate it on the Elermore Vale OpenDSS model in one command. Design in [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md); artifacts land under `runs/`. |
| [peak_duty_analysis.py](peak_duty_analysis.py) | VPP-as-peaker study: finds the aggregate peak demand condition over the full dataset, sweeps firm-capacity thresholds (f × peak), and reports how often / how long / how large a battery fleet must discharge to cover the top slice — the duty-cycle argument for VPP capability. |
| [replay_peak_event.py](replay_peak_event.py) | Physical companion to the duty study: replays the worst exceedance event through the Elermore Vale OpenDSS model with an explicit peaker dispatch (fleet sized as in the duty analysis), measuring the feeder-head shave and voltage impact. Findings written up in [PEAK_DUTY_FINDINGS.md](PEAK_DUTY_FINDINGS.md). |
| [tests/](tests/) | Verification suite for the Elermore Vale GLM→OpenDSS translation: unit tests, source-vs-circuit invariants, and physics sanity power flows. Run with `python -m pytest`; see [MODEL_VERIFICATION.md](MODEL_VERIFICATION.md) (incl. known defects + GridLAB-D cross-validation plan). |
| [profiles/](profiles/) | Output of `osqp_daily.py`: half-hourly load/PV/battery/grid time series in long-format CSV. `fit_profiles.csv` = topology 1 (gross FiT); `net_profiles.csv` = topology 2 (net metering). |
| [figures/](figures/) | Generated plots — paper figures from the QP run, plus voltage envelopes / heatmaps / substation power curves from the network simulations. |
| `data.csv`, `cleaned_data.csv` | Raw and cleaned Ausgrid solar-home dataset. |

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

### 2. Network validation — three options

All three scripts share the same CLI surface for plotting (`--save`, `--output-dir`, `--full`, `--max-days`, `--summer-day`, `--winter-day`).

**Synthetic LV feeder (fastest):**
```bash
python openDSS_LV_feeder_model.py --save
```

**IEEE 13-bus:**
```bash
python ieee_13_bus_openDSS.py --save
```

**Real Elermore Vale feeder:**
```bash
# Snapshot only (no profiles, just builds + solves):
python elermorevale_openDSS.py

# Daily comparison (baseline vs QP) for representative summer/winter days:
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --save

# Full year sweep:
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --full --save
```

Each daily run produces:
- voltage envelope (min/max p.u. across all monitored loads vs hour)
- voltage heatmap per scenario + Δ heatmap (baseline − QP)
- substation transformer P/Q curves
- per-day summary table (V min/max, violation count, total losses)

### 3. End-to-end VPP pipeline (Part B)

```bash
# One command: VPP solve -> dispatch export -> OpenDSS validation -> report
python run_vpp_network.py sharing_admm --data data1.csv --n-households 20

# Method aliases: centralised, two_stage, dual, admm, price, fcas
python run_vpp_network.py centralised_qp --soft --scenario tight_tou --data data1.csv

# Export only (stages 1-3), then re-run the network stage later:
python run_vpp_network.py admm --skip-network --data data1.csv
python run_vpp_network.py resume --run-dir runs/sharing_admm_static_...
```

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

Writes `elermorevale_dashboard.html` — a single-file dashboard with two tabs (static topology / animated live flow), play/pause/scrub controls, and synced charts.

## Implementation notes

- **GLM parsing** — `parse_glm()` in [elermorevale_openDSS.py](elermorevale_openDSS.py) walks the flat object syntax used by the Ausgrid GLM files. Numeric properties may carry units (e.g. `"11.59 m^2"`); `gfloat()` strips them before `float()` conversion.
- **Line impedances** — 11 kV configurations use z-matrix format (Ohm/mile); LV configs reference named conductors with per-mile resistance. Both are extracted at runtime by `extract_impedances()`.
- **Customer-to-load mapping** — the ~144 OSQP customers are spread evenly across the ~1,810 network loads via `map_customers_to_network_loads()`. Unmapped loads are zeroed during simulation so the QP-vs-baseline signal isn't drowned by a static background.
- **Per-unit base** — Australian residential loads in the GLM are declared at `nominal_voltage = 240 V`, so `V_NOM = 240.0` in the simulators. AS 60038 limits (`+10 % / −6 %`) are applied to the resulting per-unit voltages.
- **Substation power sign** — `collect_tx_power()` negates the monitor channels because OpenDSS monitors record power flowing *into* the monitored terminal; after the fix, positive = import from the grid as documented. Substation-power figures generated before this fix are mirror images of current output.
- **Convergence** — snapshot/daily solves run with `controlmode=off` and elevated iteration caps. OpenDSS warning #485 (Max Control Iterations Exceeded) is downgraded to a logged warning since the power-flow result is still valid.

## Data files (not redistributed for size)

`data.csv` and `cleaned_data.csv` are the Ausgrid solar-home electricity dataset (2010-2013). Source: [R17] supplementary data. The cleaning step retains the ~144 customers without missing values across a one-year window.
