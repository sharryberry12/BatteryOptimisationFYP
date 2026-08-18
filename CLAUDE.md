# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Final-year project: QP-based residential battery scheduling on the Ausgrid solar-home dataset, validated on an OpenDSS model of the real Elermore Vale 11 kV feeder (Wallsend, NSW), and extended to multi-household VPP coordination under feeder envelopes. Reproduces Ratnam, Weller & Kellett (*Renewable Energy 75*, 2015 — "[R15]") using OSQP, injects the dispatch profiles into power-flow simulations of that feeder, and benchmarks six coupling methods for a fleet. The README.md is a map of the repo; each top-level folder has its own README with the detail; keep them in sync when changing CLI surfaces or outputs.

## Layout (see README.md and `paths.py`)

```
paths.py     single source of truth for every location (DATA_CSV, GLM_DIR, GLM_COMMON, PROFILES, FIGURES, RUNS, CACHE, ...)
data/        inputs, gitignored: data.csv (one-year Ausgrid window), data_3_years.csv (optional)
dispatch/    Part A: osqp_daily.py, osqp_daily_with_DOE.py, FORMULATION.md, diagnostics/
network/     elermorevale_openDSS.py, elermorevale_gui.py, MODEL_VERIFICATION.md, glm/{Elermorevale,common}/, validation/, diagnostics/
vpp/         Part B: vpp_common.py, vpp_registry.py, vpp_export.py, run_vpp_network.py, six method folders, VPP_EXTENSION.md, PIPELINE_DESIGN.md
studies/     peak_duty_analysis.py, replay_peak_event.py, NETWORK_AWARE_DISPATCH.md, PEAK_DUTY_FINDINGS.md
docs/        WALKTHROUGH.md (hands-on tour, executable snippets)
tests/       pytest suite
outputs/     everything generated, gitignored except outputs/runs/*/manifest.json + extras.npz: profiles/, figures/, runs/, cache/
```

`dispatch`, `network`, `vpp` (and each `vpp/<method>`), `studies` are packages: `from dispatch import osqp_daily`, `from network import elermorevale_openDSS`, `from vpp import vpp_common`. Every entry script inserts the repo root into `sys.path` from its own location, so `python <folder>/<script>.py` works from any cwd, and every default path (`--data`, `--profiles`, `--output-dir`, `--glm-dir`, `--common-dir`, `--runs-root`, caches) comes from `paths.py`. Do not hard-code `profiles/`, `figures/`, `runs/`, `Elermorevale/` or `data.csv` anywhere — import from `paths`.

## Commands

```bash
pip install -r requirements.txt          # Python 3.13 tested
python -m pytest                         # 121 tests, ~12 s; no data.csv needed

# Part A — QP dispatch -> outputs/profiles/{fit,net}_profiles.csv (+ interactive paper figures)
python dispatch/osqp_daily.py
# DOE variant (export envelope shapes none/conservative/tight/rolling, --export-limit sweep, --import-limit flat cap;
#   PV curtailment + import-shortfall slack; writes outputs/profiles/<mode>_doe_<label>.csv)
python dispatch/osqp_daily_with_DOE.py --scenarios conservative tight --no-compare
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 2 --no-compare

# Network — Elermore Vale (CLI: --profiles --save --output-dir --full --max-days --summer-day --winter-day --per-day-plots --oltc)
python network/elermorevale_openDSS.py                                                     # snapshot build+solve
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save  # representative days
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save --full   # full-year sweep
python network/elermorevale_openDSS.py --profiles outputs/profiles/net_profiles.csv --save --output-dir outputs/figures/net --oltc  # -> outputs/figures/net_oltc/
python network/elermorevale_gui.py --open                        # dashboard (topology only, instant)
python network/elermorevale_gui.py --simulate --day 190 --open   # with baseline-vs-QP overlay
python network/diagnostics/diag_violation_attribution.py --every 15
python network/validation/gen_harness.py; gridlabd network/validation/harness.glm; python network/validation/compare_voltages.py

# Part B — VPP (shared CLI: --n-households --date --scenario {none,static,tight_tou,dynamic_solar} --export-limit --import-limit --save)
python vpp/centralised_qp/centralised_qp.py --n-households 20 --save   # ground truth
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save # deployed practice
python vpp/sharing_admm/sharing_admm.py --save                         # best decomposition
python vpp/run_vpp_network.py admm --n-households 20 --scenario static # solve -> export -> Elermore Vale -> outputs/runs/<id>/

# Studies (need data/data_3_years.csv, local only)
python studies/peak_duty_analysis.py --save
python studies/replay_peak_event.py
```

There is no linter config or build step. Verification: `tests/` (translation unit tests, source-vs-circuit invariants, physics goldens, GridLAB-D harness checks — `network/MODEL_VERIFICATION.md`; census/golden constants are measured ground truth), `tests/test_doe_constraints.py` (the DOE rows bind; curtailment/import-shortfall semantics; no-envelope equals `osqp_daily`), `tests/test_vpp_methods.py` (cross-method consistency on a synthetic ensemble). Everything else is verified by running the scripts and inspecting logged metrics and figures.

## Architecture

```
data/data.csv (Ausgrid, kWh per half-hour)
   └─ dispatch/osqp_daily.py | osqp_daily_with_DOE.py    (clean -> per-customer-day QP via OSQP)
        └─ outputs/profiles/*.csv                        (long-format, half-hourly kW)
             ├─ network/elermorevale_openDSS.py -> outputs/figures/<run>/  (envelopes, heatmaps, summaries.txt, sweep CSV)
             └─ network/elermorevale_gui.py -> outputs/elermorevale_dashboard_v2.html
vpp/  ── imports dispatch.osqp_daily; N households + feeder envelope; six methods; run_vpp_network.py -> network stage -> outputs/runs/<id>/
```

- **`osqp_daily_with_DOE.py` is a copy-extension of `osqp_daily.py`**, not an import: loader, cleaning rules, constants, heuristic and QP core are duplicated — a fix to one usually needs mirroring in the other. It solves over `x = [b | c | s]` (battery, curtailed PV, import shortfall); the export cap is hard (always feasible via curtailment), the import cap is soft (shortfall reported); h-scaled penalties keep relief a last resort; with no envelope it equals `osqp_daily` exactly.
- **Persistent OSQP workspace invariant** (`dispatch/FORMULATION.md` §4): the constraint matrix is built once; per day only `P` values, `q` and bounds change. `osqp.update(A=...)` is silently ignored by osqp 1.x — that is how the DOE constraint was a no-op before 2026-08-16. Pre-allocate rows and toggle bounds instead.
- **Elermore Vale is translated at runtime** from `network/glm/` (no static `.dss`): `parse_glm()` / `extract_impedances()` / `glm_length_m()` in `network/elermorevale_openDSS.py`; the GUI re-parses the GLM independently for its topology graph.
- **`vpp/` builds on Part A**: `vpp_common.py` imports `dispatch.osqp_daily` (data pipeline, tariff, billing, heuristic, constraint block); each household's local problem is the Part A QP with frozen weights `h`; methods differ only in how they enforce `D_min ≤ Σᵢ pᵢ ≤ D_max`. `vpp_registry.py` imports methods as `vpp.<m>.<m>`. Design docs: `vpp/VPP_EXTENSION.md`, `vpp/PIPELINE_DESIGN.md`; base formulation `dispatch/FORMULATION.md` (the project's original CLAUDE.md — VPP_EXTENSION's "§9/§11" refs point into it). Cleaned day arrays are cached in `outputs/cache/*.pkl`.
- **Results write-ups** live next to their code: `studies/NETWORK_AWARE_DISPATCH.md` (network results), `network/MODEL_VERIFICATION.md` (verification), method READMEs (VPP).

## Domain conventions and gotchas

- **Units**: the Ausgrid CSV stores kWh-per-half-hour; converted to kW once at load time. All downstream math is kW with `DT = 0.5` h, `T = 48`. **Interval ordering** uses the CSV column index 1..48, never parsed clock times.
- **Cleaning is rule-based** ([R17] §3 thresholds, `GC_*`/`GG_*` constants) — derives ~152 valid customers; no cleaned CSV on disk.
- **Sign conventions**: `b` > 0 discharge; `net = load − pv`; grid `p = net − b` (Part A) or `net + c − b` (DOE); `p < 0` export. Feeder envelope `D_min ≤ 0` caps export, `D_max ≥ 0` caps import. OSQP coupling duals `y` are negative where an import cap binds; dual/price methods use `μ ≈ −y`.
- **Per-unit base 240 V** (`V_NOM`), AS 60038 limits +10 % / −6 %. Violation-points = monitored load × half-hour outside the window, split over/under.
- **Network model decisions** (`network/MODEL_VERIFICATION.md` "Known defects"): bare GLM lengths are FEET; 40 Redflow batteries are `Generator`s not `Storage` (engine collapses); 25 BlueGen CHP `load`s in `Generators2.glm` are excluded (`is_chp_load`; 14 float at 0 V otherwise); `Converged=True` is never trusted alone (`solve_snapshot` rejects dead circuits; `simulate_scenario` raises `DeadMonitorError`). Network results before 2026-08-16 must be regenerated.
- **OLTC**: solves run `controlmode=off` unless `--oltc`; the zone regulator never moves on real profiles (11 kV bus within ±0.5 %) — a documented negative result. `summaries.txt` is append-mode with an OLTC header per block; `--oltc` redirects outputs to `<output-dir>_oltc`.
- **Unmapped network loads are zeroed** in profile mode; the ~152 customers are cycled over the 1,785 loads.
- Both QP scripts use `multiprocessing.Pool` (Windows spawn) — keep module top level side-effect free.
- `.gitignore` excludes `data/`, `outputs/*` (except run manifests/npz), `*.csv`, `*.pdf`, `*.html`, `*.pkl`. Method figures go to `outputs/figures/vpp/<method>/`.
- Flask is optional (`network/elermorevale_gui.py --serve`); the import is deferred with a friendly error.
