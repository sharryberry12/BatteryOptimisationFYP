# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Final-year project: QP-based residential battery scheduling on the Ausgrid solar-home dataset, validated on three OpenDSS distribution-network models. Reproduces the algorithm of Ratnam, Weller & Kellett (*Renewable Energy 75*, 2015 — "[R15]") using OSQP, then injects the resulting dispatch profiles into power-flow simulations of a synthetic LV feeder, the IEEE 13-bus feeder, and the real Elermore Vale 11 kV feeder (Wallsend, NSW). The README.md documents the full pipeline and paper references; keep it in sync when changing CLI surfaces or outputs.

## Commands

```bash
pip install -r requirements.txt          # Python 3.13 tested

# Step 1 — QP battery dispatch (writes profiles/fit_profiles.csv, profiles/net_profiles.csv, paper figures)
python osqp_daily.py

# Step 1 variant — with Dynamic Operating Envelopes (writes profiles/fit_doe_<scenario>.csv +
#          doe_scenario_comparison.csv; CLI: --mode, --scenarios, --export-limit, --no-compare)
python osqp_daily_with_DOE.py
python osqp_daily_with_DOE.py --scenarios conservative tight --no-compare

# Step 2 — network validation (all three share the same plotting CLI:
#          --save, --output-dir, --full, --max-days, --summer-day, --winter-day;
#          Elermore Vale also takes --oltc = zone RegControl active, outputs -> <output-dir>_oltc)
python openDSS_LV_feeder_model.py --save                                  # synthetic LV feeder (fastest smoke test)
python ieee_13_bus_openDSS.py --save                                      # IEEE 13-bus benchmark
python elermorevale_openDSS.py                                            # snapshot build+solve only
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --save        # representative days
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --full --save # full-year sweep
python elermorevale_openDSS.py --profiles profiles/net_profiles.csv --save --output-dir figures/net --oltc

# Peak-duty study + physical replay (need the 3-year Ausgrid file data_3_years.csv, local only)
python peak_duty_analysis.py --data data_3_years.csv --save
python replay_peak_event.py --data data_3_years.csv

# Part B — VPP coupling methods (run from repo root; shared CLI: --n-households,
#          --date, --scenario {none,static,tight_tou,dynamic_solar}, --save)
python vpp/centralised_qp/centralised_qp.py --n-households 20 --save   # ground truth
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save # deployed practice
python vpp/sharing_admm/sharing_admm.py --save                         # best decomposition

# Step 3 — dashboard (single-file HTML output)
python elermorevale_gui.py --open                        # topology only, instant
python elermorevale_gui.py --simulate --day 190 --open   # with baseline-vs-QP overlay
python elermorevale_gui.py --serve --port 8765           # optional Flask backend instead of static HTML
```

There is no linter config or build step. The Elermore Vale GLM→OpenDSS translation has a pytest suite (`python -m pytest`, ~7 s, 101 tests: unit tests, source-vs-circuit invariants, physics sanity power flows — see `MODEL_VERIFICATION.md`; census/golden constants in `tests/` are measured ground truth — plus `tests/test_doe_constraints.py`, which pins that `osqp_daily_with_DOE.py`'s envelope rows actually bind), plus a GridLAB-D cross-validation harness under `validation/` (voltages agree to ~1 % mean). Three deliberate modelling decisions documented in `MODEL_VERIFICATION.md` "Known defects": the 40 Redflow batteries are built as dispatchable `Generator` elements, not `Storage` (2+ active Storage elements make DSS C-API 0.14.5 collapse to a dead circuit that still reports Converged=True); bare GLM line lengths are FEET per GridLAB-D semantics (`glm_length_m`); and the 25 BlueGen CHP units declared as `object load` in `Generators2.glm` are excluded (`is_chp_load`) — 14 of them sit on a phase their service point doesn't carry and floated at 0 V, which zeroed V min in every sweep between 2026-08-13 and 2026-08-16. Elermore Vale network results generated before 2026-08-16 should be regenerated (3.28× inflated 11 kV impedances before 2026-08-13; dead-monitor contamination after). `Converged=True` is never trusted alone: `solve_snapshot` rejects an all-zero solution and `simulate_scenario` raises `DeadMonitorError` if any monitor reads 0 V. Everything else is verified by running the scripts and inspecting logged metrics (e.g. mean annual savings ~$430 FiT / ~$190 net) and generated figures.

## Architecture

Standalone top-level scripts — no shared package, no cross-imports among the core pipeline (`run_vpp_network.py`, `replay_peak_event.py` and `validation/compare_voltages.py` do import `elermorevale_openDSS` for the network stage; `diag_*.py` are throwaway diagnostics). Data flows between pipeline stages only via CSV files:

```
data.csv (Ausgrid, kWh per half-hour)
   └─ osqp_daily.py / osqp_daily_with_DOE.py   (clean → per-customer-day QP via OSQP)
        └─ profiles/{fit,net}_profiles.csv     (long-format, half-hourly kW)
             ├─ openDSS_LV_feeder_model.py ─┐
             ├─ ieee_13_bus_openDSS.py      ├─ figures/ (voltage envelopes, heatmaps, substation P/Q)
             ├─ elermorevale_openDSS.py    ─┘
             └─ elermorevale_gui.py → elermorevale_dashboard_v2.html
```

- **`osqp_daily_with_DOE.py` is a copy-extension of `osqp_daily.py`**, not an import. The cleaning rules, constants, heuristic, and QP core are duplicated — a fix to one usually needs mirroring in the other. Its docstring lists exactly what it adds (DOE envelope generation, extra constraint rows, slack metrics).
- **The three network scripts are also siblings, not layers**: each builds its network in-Python via `dss-python`, applies the same customer-to-load mapping idea, and exposes the same plotting CLI, but shares no code.
- **Elermore Vale model is translated at runtime** from GridLAB-D sources (`Elermorevale/*.glm` + `common/Line Configs.glm`) — there are no static `.dss` files. `parse_glm()` / `extract_impedances()` / `gfloat()` in `elermorevale_openDSS.py` do the translation; `elermorevale_gui.py` re-parses the same GLM files independently to build its topology graph.
- **`vpp/` is the Part B multi-household extension** and breaks the no-shared-code convention deliberately: `vpp/vpp_common.py` *imports* `osqp_daily` (data pipeline, tariff, billing, heuristic, constraint blocks) and each coupling method lives in its own subfolder (`centralised_qp`, `two_stage_doe_allocation`, `dual_decomposition`, `sharing_admm`, `price_based_control`, `fcas_cooptimisation`) with a script named after the approach plus a README. Scripts run from the repo root (`python vpp/<approach>/<approach>.py`), share a common CLI, and benchmark against `vpp_common.solve_centralised`. Design docs: `VPP_EXTENSION.md` (methods) and `paper_context.md` (base formulation, invariants, modelling gaps — it was the project's original CLAUDE.md, and `VPP_EXTENSION.md`'s "§9/§11" cross-references point into it). First run caches cleaned day arrays in `vpp/cache/*.pkl`.
- **`osqp_daily_with_DOE.py` → network**: its `profiles/<mode>_doe_<scenario>.csv` output has the same columns `load_profiles_from_csv()` reads, so any network script can replay DOE-constrained dispatch (`--profiles profiles/fit_doe_tight.csv`). Findings in `NETWORK_AWARE_DISPATCH.md`.

## Domain conventions and gotchas

- Both QP scripts load `"data.csv"` at the repo root (gitignored, local-only). If a `data1.csv` appears later it is an alternate dataset drop, not the pipeline default.
- **Units**: the Ausgrid CSV stores kWh-per-half-hour; it is converted to kW once at load time. All downstream math (cleaning thresholds, QP, `flow_kW * tariff * DT` billing) is in kW with `DT = 0.5` h, `T = 48` intervals/day.
- **Interval ordering** uses the integer index 1..48 from CSV column order, never parsed clock times — sorting "0:00" as a time shifts the whole day by 30 min and breaks the "first 10 intervals = before 5 am" cleaning rule (documented in `osqp_daily.py`'s docstring).
- **Dataset cleaning is rule-based in code** (Ratnam et al. 2017 Section 3 thresholds, `GC_*`/`GG_*` constants), not a hardcoded customer list — it derives ~144 valid customers for whatever date range the CSV covers.
- **Per-unit base is 240 V** (`V_NOM = 240.0`, Australian residential), with AS 60038 limits +10 % / −6 % applied to per-unit voltages.
- **OpenDSS solves run with `controlmode=off`** and raised iteration caps unless `--oltc` is given (then `controlmode=static` with the zone RegControl built — it never moves on the representative days because the 11 kV bus stays within ±0.5 %; documented negative result in `NETWORK_AWARE_DISPATCH.md`); warning #485 (Max Control Iterations Exceeded) is intentionally downgraded to a logged warning — the power-flow result is still valid.
- **Unmapped network loads are zeroed** during profile-driven simulation so the QP-vs-baseline signal isn't drowned by static background load.
- **`summaries.txt` is append-mode** — every block carries a `Zone OLTC:` header line, and `--oltc` redirects to `<output-dir>_oltc`, so OLTC-on/off outputs never interleave.
- The QP scripts use `multiprocessing.Pool`; anything executed at import time runs in every worker on Windows (spawn), so keep module top-level side-effect free.
- `.gitignore` excludes all outputs and data (`*.csv`, `*.html`, `*.pdf`, `figures/`, `profiles/`, `__pycache__/`) — `data.csv` and generated artifacts exist locally only and are never committed.
- Flask is an optional dependency, used only by `elermorevale_gui.py --serve`; the import is deferred with a friendly error if missing.
