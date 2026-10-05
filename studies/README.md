# studies/ — experiments and write-ups built on dispatch/, network/ and vpp/

| Path | What |
|---|---|
| [`NETWORK_AWARE_DISPATCH.md`](NETWORK_AWARE_DISPATCH.md) | **The main network result.** Can the batteries be scheduled to zero voltage violations on Elermore Vale? Full-year fit/net sweeps with the over/under split, the zone-OLTC negative result, DOE export-cap sweeps, the attribution of the residual violations (one boost-tap transformer; 22:00 synchronised charging), and the import-cap + curtailment experiment. Reproduce with the commands inside it. |
| [`peak_duty_analysis.py`](peak_duty_analysis.py) | VPP-as-peaker study: aggregate demand over the full dataset, firm-capacity threshold sweep, how often / how long / how large a fleet must discharge to cover the top slice. Needs `data/data_3_years.csv`. Outputs → `outputs/figures/peak_duty/`, cache → `outputs/cache/`. |
| [`replay_peak_event.py`](replay_peak_event.py) | Physical companion: replays the worst exceedance event through the Elermore Vale model with an explicit peaker dispatch and measures the feeder-head shave and voltage impact. Outputs → `outputs/runs/<id>/`. |
| [`static_vs_doe_replay.py`](static_vs_doe_replay.py) | Feeder power profile Σᵢ p_ik and zone-transformer flow on one day (default 2011-02-05, the heatwave peak) under three regimes: no battery, static limits (flat 1.5 kW export cap, no import cap — the tariff herds charging into a 22:00 spike) and a **dynamic operating envelope derived from the Jesmond 132/11 kV substation's measured headroom** (`data/Jesmond-132_11kV-FY2011.csv`). Both battery regimes are Method A soft solves; all three go through the Elermore Vale model and the measured substation profile is overlaid on the modelled transformer flow. Needs `data/data.csv` + the Jesmond CSV. Outputs → `outputs/runs/static_vs_doe_<date>_<ts>/` (dispatch CSVs, `summary.csv`, manifest, `figures/{doe_derivation,feeder_profile,zone_transformer_power}.png`; with the network stage also `voltages_<case>.npy`, monitored loads × 48 p.u. in `voltage_monitors.txt` order). `--two-stage-rules maxmin,equal` adds Method B cases (`two_stage_<rule>`: the same envelope pre-allocated into soft per-household slices through `two_stage_doe_allocation.run_rule`, as `doe_day_sweep.py` does) to every artifact. |
| [`battery_location_study.py`](battery_location_study.py) | Does it matter *where* the flexibility sits? Takes a dispatch already exported by a run (default: the latest `static_vs_doe_*` run's DOE case) and injects the same Σᵢ bᵢ two ways: distributed behind the meters (net loadshapes, the pipeline default) vs one aggregate `Generator` at the 11 kV feeder-head bus (`BusZoneSub11kV`). Solves the day step by step to get per-interval circuit losses; reports zone-transformer flow, daily losses, customer voltage envelope and violation counts for no-battery / distributed / aggregate. Outputs → `outputs/runs/battery_location_<date>_<ts>/` (`summary.csv`, manifest, `figures/{transformer_and_losses,voltage_envelope,location_value}.png`). |
| [`doe_day_sweep.py`](doe_day_sweep.py) | The single-day static-vs-DOE recipe on **every day the Jesmond record covers** (105 days, 14 Jan – 30 Apr 2011), full ensemble each day: no battery / static / centralised DOE / two-stage DOE (soft slices, all four rules), zone limit 95 % of each day's measured peak by default (`--zone-limit-of period` or `--zone-limit-mw` for one limit across the sweep), optional network stage for the base cases plus `--network-rules`. Appends one row per (day, case) to `sweep_results.csv` after each day (`--resume` continues, `--summarise-only` rebuilds the summary and figures). Outputs → `outputs/runs/doe_sweep_<first>_<last>_<ts>/` (`sweep_results.csv`, `sweep_summary.csv`, manifest, `figures/sweep_{daily_peak,zone_exceedance,cumulative_savings}.png`). ~27 s/day with the network stage. |
| [`paper_figures.py`](paper_figures.py) | Final-paper figures from existing run folders only (no solver or power-flow calls): per-household savings CDF with Jain's index (Fig. 6), zone-limit sensitivity from `--zone-limit-frac` runs (Fig. 10), customer voltage envelope from `voltages_<case>.npy` (Fig. 8) and the optional year-sample 22:00 attribution from `diag_violation_attribution.py --csv`. One colour and line style per regime in every figure; a two-stage series is drawn when a run made with `--two-stage-rules` is given. Outputs → `outputs/figures/paper/<name>.{png,pdf}` (300 dpi, one IEEE column), a CSV twin per figure and `checks.md`, which recomputes every number the paper quotes (expected / computed / match). |
| [`PEAK_DUTY_FINDINGS.md`](PEAK_DUTY_FINDINGS.md) | Write-up of the peak-duty study (2010-07 → 2013-06 data). |

```bash
python studies/peak_duty_analysis.py --save                       # default --data data/data_3_years.csv
python studies/peak_duty_analysis.py --clean --save
python studies/replay_peak_event.py
python studies/static_vs_doe_replay.py                            # 2011-02-05, N=152, zone limit 95 % of measured peak
python studies/static_vs_doe_replay.py --zone-limit-frac 0.9 --skip-network
python studies/static_vs_doe_replay.py --two-stage-rules maxmin,equal   # + Method B series, voltages_<case>.npy
python studies/battery_location_study.py                          # latest static_vs_doe run, --dispatch doe
python studies/battery_location_study.py --run-dir outputs/runs/centralised_qp_static_<...> --dispatch coupled
python studies/doe_day_sweep.py                                   # all 105 Jesmond days, network on (~50 min)
python studies/doe_day_sweep.py --skip-network --dates 2011-02-01:2011-02-10
python studies/paper_figures.py --run outputs/runs/static_vs_doe_<ts> --network-run outputs/runs/static_vs_doe_<ts2> \
    --sweep-runs "outputs/runs/static_vs_doe_2011-02-05_*" --attribution-csv outputs/figures/paper/attribution_by_hour.csv
```

Related, elsewhere: the VPP → network pipeline (`vpp/run_vpp_network.py`)
and its run manifests under `outputs/runs/`; the violation attribution
diagnostic (`network/diagnostics/diag_violation_attribution.py`).
The draft final-paper section that cites these studies, with its figure list
and number provenance: [`docs/NUMERICAL_SIMULATION_DRAFT.md`](../docs/NUMERICAL_SIMULATION_DRAFT.md).
