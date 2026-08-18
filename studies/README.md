# studies/ — experiments and write-ups built on dispatch/, network/ and vpp/

| Path | What |
|---|---|
| [`NETWORK_AWARE_DISPATCH.md`](NETWORK_AWARE_DISPATCH.md) | **The main network result.** Can the batteries be scheduled to zero voltage violations on Elermore Vale? Full-year fit/net sweeps with the over/under split, the zone-OLTC negative result, DOE export-cap sweeps, the attribution of the residual violations (one boost-tap transformer; 22:00 synchronised charging), and the import-cap + curtailment experiment. Reproduce with the commands inside it. |
| [`peak_duty_analysis.py`](peak_duty_analysis.py) | VPP-as-peaker study: aggregate demand over the full dataset, firm-capacity threshold sweep, how often / how long / how large a fleet must discharge to cover the top slice. Needs `data/data_3_years.csv`. Outputs → `outputs/figures/peak_duty/`, cache → `outputs/cache/`. |
| [`replay_peak_event.py`](replay_peak_event.py) | Physical companion: replays the worst exceedance event through the Elermore Vale model with an explicit peaker dispatch and measures the feeder-head shave and voltage impact. Outputs → `outputs/runs/<id>/`. |
| [`PEAK_DUTY_FINDINGS.md`](PEAK_DUTY_FINDINGS.md) | Write-up of the peak-duty study (2010-07 → 2013-06 data). |

```bash
python studies/peak_duty_analysis.py --save                       # default --data data/data_3_years.csv
python studies/peak_duty_analysis.py --clean --save
python studies/replay_peak_event.py
```

Related, elsewhere: the VPP → network pipeline (`vpp/run_vpp_network.py`)
and its run manifests under `outputs/runs/`; the violation attribution
diagnostic (`network/diagnostics/diag_violation_attribution.py`).
