# Method B — Two-Stage DOE Allocation

**The deployed-practice architecture.** Stage 1: the DNSP splits the feeder
envelope into per-household envelopes under an allocation rule. Stage 2: every
household solves its existing QP independently with its allocated envelope —
zero coordination at solve time. This is what SA Power Networks' Flexible
Exports trial actually does, and the closest-to-zero-new-code option
(VPP_EXTENSION.md §4): only the allocator is new.

## Allocation rules implemented

| Rule | Stage-1 split of the export budget `-D_min,k` |
|---|---|
| `equal` | `1/N` each — trivially fair in envelope, wasteful in outcome |
| `prorata_pv` | proportional to day-peak PV (proxy for system size; the CSV's Generator Capacity column is not carried through the pipeline) |
| `prorata_surplus` | per-interval proportional to forecast surplus `(pv - load)+`; equal split when nobody has surplus |
| `maxmin` | max-min fair progressive filling (`water_fill`) against each household's physical export cap `max(P_MAX - net, 0)` |

By construction `sum_i D_min,i >= D_min`, so feeder compliance holds without
any runtime coordination (verified in the output anyway).

Per-household envelopes can demand more export headroom than a 5 kW battery can
physically deliver; those bounds are relaxed to the battery limit and reported
as **required curtailment (kWh)** — physically, PV spill.

**Energy-infeasible slices (read before quoting the numbers).** The relaxation
above only handles per-interval *power*; a slice can still be infeasible on
*energy* (e.g. an import cap that would need more evening discharge than the
10 kWh battery holds, or an export cap whose forced charging exceeds the SOC
headroom). Those households come back `OSQP status primal infeasible`, get a
**zero dispatch** (no battery at all: `pi = net`) and are counted in
`n_failed`; the aggregate then violates the envelope by whatever those
households export/import unaided. Measured 2026-08-18 on 8 households: 2 of 8
fail under a 2 kW/household import cap (winter) and under a 0.05 kW/household
export cap (summer, aggregate violation 2.6 kW), while every rule shows a
300–900 % objective gap versus Method A. That is a real property of
allocate-then-solve with hard envelopes, but the zero fallback is
*pessimistic*: a deployed inverter would curtail (export) or the household
would simply exceed the cap with the battery still helping (import). A
best-effort fallback (soft per-household DOE with a slack penalty, or a PV
curtailment variable) is the natural next step — see studies/NETWORK_AWARE_DISPATCH.md §5.
Note also that all four rules split the *import* side equally, so under a
pure import cap they coincide.

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
```

Outputs: per-rule table (objective, gap %, savings, Jain, Gini, violation,
curtailment, failed solves), `outputs/figures/vpp/two_stage_doe_allocation/two_stage_aggregate.png` and
`outputs/figures/vpp/two_stage_doe_allocation/two_stage_tradeoff.png`.

## Assessment (from VPP_EXTENSION.md §4)

| | |
|---|---|
| Optimality | Suboptimal by design — allocation precedes revelation of need; the gap is the headline number |
| Privacy | Good — DNSP needs only aggregate/forecast information |
| Robustness | Excellent — households are fully independent |
| Realism | Highest of all methods; matches Australian DNSP direction |
