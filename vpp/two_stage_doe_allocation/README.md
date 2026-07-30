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
curtailment, failed solves), `figures/two_stage_aggregate.png` and
`figures/two_stage_tradeoff.png`.

## Assessment (from VPP_EXTENSION.md §4)

| | |
|---|---|
| Optimality | Suboptimal by design — allocation precedes revelation of need; the gap is the headline number |
| Privacy | Good — DNSP needs only aggregate/forecast information |
| Robustness | Excellent — households are fully independent |
| Realism | Highest of all methods; matches Australian DNSP direction |
