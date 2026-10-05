# Method B — Two-Stage DOE Allocation

**The deployed-practice architecture.** Stage 1: the DNSP splits the feeder
envelope into per-household envelopes under an allocation rule. Stage 2: every
household solves its existing QP independently with its allocated envelope —
zero coordination at solve time. This is what SA Power Networks' Flexible
Exports trial actually does, and the closest-to-zero-new-code option
(VPP_EXTENSION.md §4): only the allocator is new.

## Allocation rules implemented

| Rule | Export budget `-D_min,k` | Import budget `D_max,k` |
|---|---|---|
| `equal` | `1/N` each — trivially fair in envelope, wasteful in outcome | `1/N` each |
| `prorata_pv` | proportional to day-peak PV (proxy for system size; the CSV's Generator Capacity column is not carried through the pipeline) | `1/N` each (PV size says nothing about import need) |
| `prorata_surplus` | per-interval proportional to forecast surplus `(pv - load)+`; equal split when nobody has surplus | per-interval proportional to forecast **need** `(load - pv)+`; equal when nobody imports |
| `maxmin` | max-min fair progressive filling (`water_fill`) against each household's physical export cap `max(P_MAX - net, 0)` | max-min fair with **floors** (`floor_fill`, added 2026-09-29): every household gets the same allowance except those whose unavoidable import `max(net - P_MAX, 0)` exceeds it, which get their floor |

By construction `sum_i D_min,i >= D_min` and `sum_i D_max,i <= D_max`, so
feeder compliance holds without any runtime coordination (verified in the
output anyway). Unbounded intervals stay unbounded on both sides.

Per-household envelopes can demand more export headroom than a 5 kW battery can
physically deliver; those bounds are relaxed to the battery limit. The script
then reports the **realised** out-of-envelope energy of each dispatch against
its original allocation, split into its two distinct physical quantities:
**curt kWh** (export excess — PV a deployed inverter would spill, so it is
credited before the residual "viol kW" is measured) and **short kWh** (import
excess the battery cannot cover — nothing physical removes it, so it stays
in "viol kW").

**Energy-infeasible slices: hard vs `--soft` (read before quoting the
numbers).** The relaxation above only handles per-interval *power*; a slice
can still be infeasible on *energy* (an import cap that would need more
evening discharge than the 10 kWh battery holds, or an export cap whose
forced charging exceeds the SOC headroom). In the default **hard** solve
those households come back `OSQP status primal infeasible`, get a **zero
dispatch** (no battery at all: `pi = net`) and are counted in `n_failed`;
the aggregate then violates the envelope by whatever those households
import/export unaided, and the objective gap mostly measures that fallback
(a no-battery household's objective is ~4x its optimised one). With
`--soft`, every household instead gets `vpp_common.HouseholdSolver(soft=True)`:
the same slack formulation as Method A's soft mode, `x = [b | s_up | s_lo]`
with `pi - s_up <= D_max,i` (import shortfall) and `pi + s_lo >= D_min,i`
(export excess), `s >= 0`, at a linear penalty of `SOFT_PENALTY_DEFAULT`
per kW. The battery does everything it physically can and the slack
carries only the remainder, so nobody drops out and the reported
`short kWh` is the genuine undeliverable part of the allocation.

Measured on the 5 Feb 2011 zone-substation DOE of
`studies/static_vs_doe_replay.py` (N = 152, `--envelope-from` the study's
run; Method A soft: objective 1,092,017, shortfall 0, savings $205.68/day;
tables in `outputs/figures/vpp/two_stage_doe_allocation/jesmond_2011-02-05_{hard,soft}/rule_comparison.csv`):

| Rule | hard: fail / short kWh / gap | soft: short kWh / residual viol kW / gap / savings $/day |
|---|---|---|
| `equal` | 74 / 1,982 / 185 % | 1,419 / 99 / 1.5 % / 232.16 |
| `prorata_pv` | 73 / 1,982 / 181 % | 1,419 / 100 / 1.0 % / 234.45 |
| `prorata_surplus` | 51 / 541 / 117 % | **203 / 47 / 4.3 % / 210.16** |
| `maxmin` | 74 / 1,959 / 185 % | 1,396 / 101 / 1.2 % / 234.09 |

Two lessons. (1) The import-side max-min rule buys almost nothing over an
equal split here: floors bind for only 12 of 152 households (net above
5.66 kW at the substation peak), and what makes the slices infeasible is
*energy* over the six-hour window, which a per-interval floor cannot see.
(2) Need-proportional slices (`prorata_surplus`) deliver all but 203 kWh of
the envelope at the same household cost as the centralised solve ($210 vs
$206), while equal/max-min slices keep household savings near the static
case ($232–234) by leaving 1.4 MWh undelivered. In soft mode read the
objective gap together with `short kWh`: a small gap with a large
shortfall means the households solved almost as if unconstrained.

## The research signal

For each rule the script reports **efficiency gap vs the centralised optimum**
(Method A, solved as a soft benchmark) *and* **fairness of realised savings**
(Jain index + Gini). Allocating equal envelopes does not produce equal benefit —
this efficiency-versus-fairness tension is exactly the AEMO fairness question,
and the grouped bar chart is the policy-legible artefact.

## Run

```bash
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py \
    --rules equal,maxmin --scenario tight_tou --save
# soft slices on the envelope of an existing run (N and date must match its manifest)
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --n-households 152 --date 2011-02-05 \
    --envelope-from outputs/runs/static_vs_doe_2011-02-05_<ts> --soft --save \
    --output-dir outputs/figures/vpp/two_stage_doe_allocation/jesmond_2011-02-05_soft
```

Outputs: per-rule table (objective, gap %, savings, Jain, Gini, residual
violation, curtailment, import shortfall, failed solves; written as
`rule_comparison.csv` next to the figures with `--save`), `outputs/figures/vpp/two_stage_doe_allocation/two_stage_aggregate.png` and
`outputs/figures/vpp/two_stage_doe_allocation/two_stage_tradeoff.png`.

## Assessment (from VPP_EXTENSION.md §4)

| | |
|---|---|
| Optimality | Suboptimal by design — allocation precedes revelation of need; the gap is the headline number |
| Privacy | Good — DNSP needs only aggregate/forecast information |
| Robustness | Excellent — households are fully independent |
| Realism | Highest of all methods; matches Australian DNSP direction |
