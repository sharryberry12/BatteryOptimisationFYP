# network/ — the Elermore Vale 11 kV feeder in OpenDSS

A runtime translation of Ausgrid's GridLAB-D model of the Elermore Vale
feeder (Wallsend, NSW) into OpenDSS via `dss-python`, plus everything needed
to replay battery dispatch profiles on it and to trust the numbers.

| Path | What |
|---|---|
| [`elermorevale_openDSS.py`](elermorevale_openDSS.py) | The model. `build_elermorevale()` parses `glm/` and issues OpenDSS commands (132 kV source → zone TX → OLTC → 11 kV lines → 23 distribution TXs → LV lines → 1,785 loads, plus PV/batteries as `Generator`s in full mode). Profile mode: `simulate_scenario()` attaches a `LoadShape` per load from a dispatch CSV, runs 48 half-hour solves, monitors 100 loads + the zone TX; `run_full_sweep()` does the year. CLI below. |
| [`elermorevale_gui.py`](elermorevale_gui.py) | Single-file HTML dashboard: static topology view + animated live-flow view; `--simulate` overlays baseline vs QP for one day. |
| [`glm/Elermorevale/`](glm/Elermorevale/), [`glm/common/`](glm/common/) | The GridLAB-D sources (feeder, subs, transformer configs, generators scenario file) and shared includes (`Line Configs.glm` …). Read-only inputs, tracked. |
| [`validation/`](validation/) | Level 4 cross-validation: `gen_harness.py` strips the GLM into a one-instant harness GridLAB-D can solve, `compare_voltages.py` joins both engines' node-phase voltages (~1 % mean agreement). |
| [`diagnostics/diag_violation_attribution.py`](diagnostics/diag_violation_attribution.py) | Where the violations come from: baseline vs QP on every N-th day, split over/under, by hour, by feeder. |
| [`MODEL_VERIFICATION.md`](MODEL_VERIFICATION.md) | **Read this before quoting any network number.** The four-level verification pyramid, measured ground truth, known approximations, and the known-defect log (feet-vs-metres, phantom phases, floating BlueGen loads, dead-circuit "Converged=True", …). |

## Run

```bash
python network/elermorevale_openDSS.py                                                     # snapshot build + solve
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save  # representative days
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save --full   # full year
python network/elermorevale_openDSS.py --profiles outputs/profiles/net_profiles.csv --save --output-dir outputs/figures/net
python network/elermorevale_openDSS.py --profiles outputs/profiles/net_profiles.csv --save --output-dir outputs/figures/net --oltc  # -> outputs/figures/net_oltc/
python network/elermorevale_gui.py --open                       # topology only, instant
python network/elermorevale_gui.py --simulate --day 190 --open  # baseline vs QP overlay
python network/diagnostics/diag_violation_attribution.py --every 15
python network/validation/gen_harness.py && gridlabd network/validation/harness.glm && python network/validation/compare_voltages.py
```

CLI: `--profiles`, `--save`, `--output-dir` (default `outputs/figures`),
`--full`, `--max-days`, `--summer-day`, `--winter-day`, `--per-day-plots`,
`--oltc` (build the zone RegControl and run with `controlmode=static`; outputs
redirect to `<output-dir>_oltc`), `--glm-dir` / `--common-dir` (default
`glm/…` via `paths.py`).

Outputs per run: `voltage_envelope_<date>.png`, `substation_power_<date>.png`,
`heatmaps_{baseline,qp,delta}/`, `summaries.txt` (append-mode; every block
carries the date and the OLTC state; violations split into over > +10 % and
under < −6 %), and with `--full` `opendss_sweep_results.csv` (one row per
day: `oltc, base_/qp_ v_min, v_max, violations, over, under, peak_tx_kw,
loss_kw`) + `sweep_summary.png`.

## Conventions and gotchas

- **Per-unit base 240 V** (`V_NOM`), AS 60038 window +10 % / −6 %.
- Bare GLM line lengths are **feet** (GridLAB-D default); explicit units win
  (`glm_length_m`).
- Loads sit on their GLM phase; the 25 BlueGen CHP units declared as
  `object load` in `glm/Elermorevale/generators/Generators2.glm` are
  excluded (`is_chp_load`) — 14 of them are on a phase their service point
  doesn't carry and would float at 0 V.
- The 40 Redflow batteries are `Generator` elements, not `Storage` (2+
  active Storage elements collapse the DSS C-API 0.14.5 solve to a dead
  circuit that still reports Converged=True).
- `Converged=True` is never trusted alone: `solve_snapshot` rejects an
  all-zero solution and every profile-driven day asserts all monitors stayed
  energised (`DeadMonitorError`).
- Solves run `controlmode=off` unless `--oltc`; warning #485 is downgraded.
  With `--oltc` the zone regulator never moves on real profiles (the 11 kV
  bus stays within ±0.5 %) — see `studies/NETWORK_AWARE_DISPATCH.md`.
- Every profile-driven run maps the ~152 clean customers round-robin over the
  1,785 loads (`map_customers_to_network_loads`) and monitors every 18th load.

Tests: `python -m pytest` (from the repo root) runs the translation unit
tests, source-vs-circuit invariants, physics goldens and the harness
checks — see MODEL_VERIFICATION.md.
