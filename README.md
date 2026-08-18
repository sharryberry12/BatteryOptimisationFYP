# Battery Optimisation FYP

QP-based residential battery scheduling on the Ausgrid solar-home dataset,
validated on an OpenDSS model of the real **Elermore Vale 11 kV feeder**
(Wallsend, NSW), and extended to multi-household VPP coordination under
feeder envelopes.

The project reproduces the algorithm of Ratnam, Weller & Kellett
(*Renewable Energy 75*, 2015) with OSQP, injects the resulting dispatch into a
power-flow simulation of an actual Ausgrid feeder to see what the batteries do
to the network, and then asks how a fleet of them should be coordinated.

## Where things are

```
paths.py          the single source of truth for every location below
data/             INPUTS (not in git): data.csv (Ausgrid one-year window), data_3_years.csv
dispatch/         PART A -- QP battery scheduling for one household-day
network/          the Elermore Vale OpenDSS model, its GridLAB-D sources, validation, dashboard
vpp/              PART B -- multi-household coupling methods + the VPP -> network pipeline
studies/          experiments and write-ups built on the above
docs/             cross-cutting: WALKTHROUGH.md (hands-on tour of all three layers)
tests/            pytest suite (translation invariants, physics goldens, DOE + VPP consistency)
outputs/          EVERYTHING GENERATED (not in git): profiles/, figures/, runs/, cache/
```

| Folder | Read first | Entry points |
|---|---|---|
| [dispatch/](dispatch/) | [dispatch/README.md](dispatch/README.md), [dispatch/FORMULATION.md](dispatch/FORMULATION.md) | `osqp_daily.py` (R15 QP + heuristic), `osqp_daily_with_DOE.py` (+ export/import envelopes, PV curtailment, import shortfall), `diagnostics/` |
| [network/](network/) | [network/README.md](network/README.md), [network/MODEL_VERIFICATION.md](network/MODEL_VERIFICATION.md) | `elermorevale_openDSS.py` (build + profile-driven simulation), `elermorevale_gui.py` (dashboard), `glm/` (GridLAB-D sources), `validation/` (GridLAB-D cross-check), `diagnostics/` |
| [vpp/](vpp/) | [vpp/README.md](vpp/README.md), [vpp/VPP_EXTENSION.md](vpp/VPP_EXTENSION.md), [vpp/PIPELINE_DESIGN.md](vpp/PIPELINE_DESIGN.md) | six method folders (`centralised_qp`, `two_stage_doe_allocation`, `dual_decomposition`, `sharing_admm`, `price_based_control`, `fcas_cooptimisation`), `run_vpp_network.py` (solve → export → Elermore Vale → report) |
| [studies/](studies/) | [studies/README.md](studies/README.md), [studies/NETWORK_AWARE_DISPATCH.md](studies/NETWORK_AWARE_DISPATCH.md), [studies/PEAK_DUTY_FINDINGS.md](studies/PEAK_DUTY_FINDINGS.md) | `peak_duty_analysis.py`, `replay_peak_event.py` |
| [docs/](docs/) | [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | — |
| [data/](data/) | [data/README.md](data/README.md) | — |

## Quick start

```bash
pip install -r requirements.txt            # Python 3.13 tested; dss-python for the network model
python -m pytest                           # 121 tests, ~12 s (no data.csv needed)

# 1. Part A -- QP dispatch for every customer-day -> outputs/profiles/{fit,net}_profiles.csv
python dispatch/osqp_daily.py
python dispatch/osqp_daily_with_DOE.py --scenarios conservative tight --no-compare      # + DOE envelopes
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 2 --no-compare   # import cap only

# 2. Network -- replay a profile set on Elermore Vale (representative days / full year)
python network/elermorevale_openDSS.py                                                   # snapshot only
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save --full
python network/elermorevale_gui.py --open                                                # dashboard

# 3. Part B -- coordinate N households under a feeder envelope, and push it through the network
python vpp/centralised_qp/centralised_qp.py --n-households 20 --save
python vpp/sharing_admm/sharing_admm.py --n-households 20 --save
python vpp/run_vpp_network.py admm --n-households 20 --scenario static
```

Every script runs from the repo root (or anywhere: they locate the repo
themselves) and every default input/output path comes from
[paths.py](paths.py); pass `--output-dir` / `--profiles` / `--data` to
override. `outputs/` is created on demand.

## Results and write-ups

- **What the batteries do to the feeder, and whether they can be scheduled
  to zero voltage violations:** [studies/NETWORK_AWARE_DISPATCH.md](studies/NETWORK_AWARE_DISPATCH.md)
  (full-year sweeps, zone-OLTC negative result, DOE export/import envelopes,
  attribution of the residual violations).
- **How the network model was verified:** [network/MODEL_VERIFICATION.md](network/MODEL_VERIFICATION.md)
  (four-level pyramid, GridLAB-D cross-validation, known defects incl. the
  fixed ones).
- **VPP methods, their maths and their measured behaviour:** [vpp/README.md](vpp/README.md)
  and each method's README; cross-method consistency is pinned by
  [tests/test_vpp_methods.py](tests/test_vpp_methods.py).
- **VPP as a peaker:** [studies/PEAK_DUTY_FINDINGS.md](studies/PEAK_DUTY_FINDINGS.md).
- **Learn it by hand:** [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## References

- **[R15]** Ratnam, Weller & Kellett, *"An optimization-based approach to scheduling residential battery storage with solar PV: Assessing customer benefit,"* Renewable Energy 75, 2015 — the algorithm paper.
- **[R17]** Ratnam, Weller, Kellett & Murray, *"Residential load and rooftop PV generation: an Australian distribution network dataset,"* IJSE 36(8), 2017 — the dataset paper.
- **[OSQP]** Stellato et al., *"OSQP: An operator splitting solver for quadratic programs,"* Math. Prog. Comp. 12, 2020.

The papers and the Ausgrid data are not redistributed here (`*.pdf`,
`data/`, `outputs/` are gitignored); see [data/README.md](data/README.md).
