# VPP coordination: comparison plan, the DNSP-run hybrid, and the proposed figures

Notes for the final FYP paper, 29 Sep 2026. Figure and equation numbers match the current `FYP_final_paper.tex` build.

---

## 1. What the paper compares

All regimes use the same 152 households, the same test day (5 Feb 2011), the same tariff and the same frozen weights hᵢ. They differ only in how the import side of each household is limited.

| Regime | Who decides the dispatch | Import limit | Status |
|---|---|---|---|
| No battery | Nobody | None | Done |
| Static limits | Each household alone | None (flat 1.5 kW export cap only) | Done |
| Two-stage allocation (Method B) | DNSP splits D̅_F into slices; each household solves alone | D̅ᵢ = D̅_F / N (Section II-C) | Running |
| Centralised VPP (Method A) | One stacked solve over all households (Eq. 14) | Feeder total ≤ D̅_F (Eq. 15) | Done (the current "DOE" row) |
| Hybrid (optional) | DNSP runs Method A on forecasts, issues each πᵢ* as that household's envelope; each household solves alone | D̅ᵢ = max(πᵢ*, 0) | Not run |

**Relabel the current "DOE" row as "Centralised VPP".** Static limits, two-stage and centralised all use a DOE in some form, so "DOE" alone no longer identifies a regime.

**Metrics for every row** (the Table I columns plus three new ones): feeder peak and its time, maximum half-hour ramp, standard deviation, substation excess, fleet savings, and then:
- **objective gap to centralised**, the price of not coordinating;
- **Jain's index** on per-household savings, the fairness cost;
- **number of households with import shortfall**, i.e. households whose slice can't cover their own load.

### Caveat for the two-stage run

The walkthrough (§3.3, Q4) notes that all four allocation rules give **the same result when the feeder limit is on import**. The rules only differ in how they split the export budget; imports are split equally by all of them.

5 February is an import-limited day, so **water-filling (Section II-C) will give exactly the same answer as the equal split** here. The DOE minimum is 101 kW, i.e. 0.66 kW per household at 16:30. I'd expect many households to hit their shortfall variables in that interval.

That's fine for the three-way comparison. If water-filling itself is meant to be a result, pick one of these:
- run an export-limited day as well (the walkthrough suggests 7 Jan 2011 with a tiny export limit and no import limit); or
- apply water-filling on the import side too, capping each household at its forecast net load plus B̅, so households that can't use their share give it up.

---

## 2. The hybrid: the DNSP runs the centralised solve and issues the results as envelopes

### How it works (day-ahead)

1. The DNSP forecasts ℓᵢ and gᵢ for every household and computes the feeder limit D̅_F from the zone-substation headroom (Eq. 15).
2. The DNSP solves the stacked problem (Eq. 14) and gets each household's optimal grid profile πᵢ*.
3. It issues each connection point the envelope D̅ᵢ,ₖ = max(πᵢ,ₖ*, 0).
4. Each household, or its aggregator, solves its own QP under that envelope. This is two-stage stage 2, and the code already exists.

The solve happens centrally, but the dispatch still happens at each household.

### Why it recovers the centralised optimum (under ideal conditions)

Assumptions: perfect forecasts, the DNSP models each household with the same hᵢ the household uses, and the feeder limit is hard.

- **The envelopes are safe.** If every πᵢ ≤ πᵢ*, then Σᵢ πᵢ ≤ Σᵢ πᵢ* ≤ D̅_F. Any dispatch that respects the per-household envelopes also respects the feeder limit.
- **The households choose πᵢ\* anyway.** πᵢ* satisfies household i's own envelope. So when household i solves alone, its local optimum x̂ᵢ satisfies fᵢ(x̂ᵢ) ≤ fᵢ(xᵢ*).
- **Combining the two.** The combined solution x̂ is feasible for the centralised problem, so Σ fᵢ(x̂ᵢ) ≥ Σ fᵢ(xᵢ*), because x* is the centralised optimum. Together with the previous point, every inequality is an equality.
- **So the households land exactly on πᵢ\*.** fᵢ is strictly convex in πᵢ (H is diagonal and positive), and βᵢ is fixed by the power balance once πᵢ is. So x̂ᵢ = xᵢ*.

In words: the 300–900 % two-stage gap comes from a poor choice of split, not from making the decision in two stages. A split chosen by the centralised solve closes the gap completely.

**This is my own derivation.** It is standard convexity reasoning, not taken from the repo docs or a paper. Confirm it numerically before stating it as a result (see "How to test it" below).

### Comparison

| | Two-stage (equal split) | Centralised | Hybrid |
|---|---|---|---|
| Objective vs optimum | 300–900 % gap | Optimal | Optimal under perfect forecasts |
| Feeder safety | Guaranteed by construction | Only if every battery follows the dispatch | Guaranteed by construction |
| Who controls batteries | Households or aggregators | One VPP operator | Households or aggregators |
| Works with several aggregators on one feeder | Yes | No | Yes |
| Data the DNSP needs | Totals and forecasts | Every household's load, PV and objective | Every household's forecast, plus a model of its objective |
| Fairness | Set by the split rule | A side effect of minimising the total | Same as centralised, unless the DNSP adds fairness constraints |
| Enforceable at the connection point | Yes | No | Yes |

### Where it breaks down (what makes it a research question)

1. **The DNSP's model of each household can be wrong.** Households sit on different retailers' tariffs. If the assumed hᵢ is wrong, the envelopes stay safe but are no longer optimal. Model error costs efficiency, never network safety.
2. **Forecasts can be wrong.** A household whose evening load beats its forecast hits its own envelope and falls back on local curtailment or import shortfall; the feeder doesn't breach. Again, errors cost efficiency, not safety, which is the property a DNSP cares about most.
3. **Import envelopes must be non-negative.** Section II defines D̅ᵢ ≥ 0, but πᵢ* can be negative (exporting) in some intervals. Using max(πᵢ*, 0) keeps the result exact wherever the feeder limit binds, because in the evening every household is importing. It's only loose in intervals where the limit isn't binding anyway.
4. **Fairness carries over from the centralised solve**, including 136 of 152 households worse off. But because the DNSP runs the solve, it can add fairness constraints (a floor on per-household savings, a cap on how far any household's envelope is cut) before issuing envelopes. That's the most direct answer to AEMO's fairness question [11] of any regime here.
5. **Data access.** The DNSP needs per-household forecasts. DNSPs see smart-meter data in the NEM, as far as I know, but check the current rules before stating this in the paper.
6. **It's not a new idea in the literature.** DNSP-side, optimisation-based DOE calculation already exists, for example [2] (Liu and Braslavsky). The hybrid's specific angle is choosing envelopes by the VPP's objective, not only by network limits. Check how close [2] and related work come before claiming the hybrid as new.

### How to test it (cheap, reuses existing code)

1. Take `dispatch_doe.csv` from the centralised run (`outputs/runs/static_vs_doe_2011-02-05_20260929-111819/`).
2. For each household, set D̅ᵢ,ₖ = max(grid_kwᵢ,ₖ, 0).
3. Pass those envelopes to the two-stage stage-2 solver (`run_rule` with a custom allocation).
4. Report the objective gap to centralised. **Under perfect forecasts it should be about 0 %.** If it isn't, the likely causes are the non-negative clipping (point 3) or soft slacks leaving the feeder limit non-binding.
5. **Robustness run.** Perturb load forecasts by ±10 % before stage 2 and report the gap, the number of households with shortfall, and any feeder-limit violation. This shows points 1–2: the gap grows, but feeder safety holds.

This adds a fifth row to Table I and is arguably the paper's most useful policy result.

---

## 3. The proposed (not yet generated) figures and why each matters

The current build has 10 figure slots; three are proposed. Once two-stage (and possibly the hybrid) are added, each proposed figure gains one or two series.

### Fig. 6: Per-household daily savings distribution (`fig_savings_cdf`)

**What it shows.** An empirical CDF (or paired histogram) of the 152 daily savings under each regime, with Jain's index in the legend.

**Why it matters.**
- **It's the only figure about fairness.** Every other figure shows the network or the fleet as a whole. Table I reports fleet totals, which hide the finding that 136 of 152 households are worse off under the centralised VPP. The paper's fairness argument rests on a figure that doesn't exist yet.
- **It's the core of the three-way comparison.** Centralised maximises the total, two-stage splits by a rule, and the hybrid can add fairness constraints. The CDF is where readers see who pays under each. Expect two-stage to shift the whole curve left (lower savings) and possibly spread it; the walkthrough's 8-household run went from Jain 0.83 to 0.52.
- **It ties the paper to AEMO [11].** AEMO asks who pays for network relief; this figure answers it with data.

**If you drop it.** Readers get the fairness claim only as two numbers (Jain 0.71 → 0.66), and the conclusion's call for fairness-aware allocation has little visible evidence behind it.

**How to build it.** The `daily_savings` column in each regime's dispatch CSV; plot one step line per regime. No new simulation needed.

**Priority: high.** It's cheap, and the comparison story depends on it.

### Fig. 8: Customer voltage envelope (`fig_voltage_env`)

**What it shows.** The minimum and maximum voltage across the 100 monitored loads over the day, one band per regime, with the AS IEC 60038 limits (0.94 and 1.10 p.u.).

**Why it matters.**
- **It shows the effect that the violation count hides.** Under static limits the daily violation count *falls* (340 → 297), which reads as an improvement, yet the worst voltage drops to 0.648 p.u. at 22:00. Table II says this in numbers; the figure shows the 22:00 dip directly.
- **It links the feeder profile (Fig. 5) to physical harm.** Fig. 5 shows the 22:00 peak in kW; Fig. 8 shows what that peak does to customers' supply voltage.
- **It adds a network test to the comparison.** Two-stage is likely to cut the evening peak unevenly. Some households' slices bind while others don't, so voltage at particular LV nodes could behave differently from the centralised case even at similar feeder totals.
- **It gives the residual violations context.** The boost-tap transformer keeps some loads high all day. The band shows that upper edge, which helps explain why the model reports violations even with no battery.

**If you drop it.** Table II still carries the numbers, but the "count falls yet worst voltage worsens" point is easy for a reader to miss in a table.

**How to build it.** A small addition to `studies/static_vs_doe_replay.py`, reusing `plot_voltage_envelope` from the location study (walkthrough §2.2). It needs the per-monitor voltage arrays, which the replay script already collects.

**Priority: medium.** It strengthens Section III-E, but Table II covers the essentials if space is tight.

### Fig. 10: Zone-limit sensitivity (`fig_sensitivity`)

**What it shows.** Against the zone-limit fraction f ∈ {0.99, 0.97, 0.95, 0.92, 0.90}: the feeder peak, fleet savings and import shortfall, as twin axes or three small panels.

**Why it matters.**
- **It turns one data point into a trade-off curve.** Every current result is at f = 0.95, which was chosen so the substation excess (693 kWh) fits inside the fleet's energy (1,520 kWh). A reviewer will ask whether the conclusions depend on that choice. This figure answers it.
- **It shows where the fleet runs out.** Around f = 0.90 (about 2 MWh of excess) the soft solve reports shortfall. The figure shows how much substation relief 152 batteries can buy and at what cost to households, which a DNSP actually needs when sizing a VPP program.
- **It shows how the gap grows under pressure.** Plot centralised and two-stage on the same axes. At loose limits (f = 0.99) both regimes barely bind and should nearly coincide; as the limit tightens the equal split should fail first, so the price of not coordinating should grow with how tight the limit is. That's a stronger result than a single gap figure on one day.
- **It supports the "must be dynamic" argument.** The shortfall curve shows directly where even the best coordination can't meet the limit.

**If you drop it.** Section III-G has to say "results are for one zone-limit level" as a limitation rather than addressing it, and there's a red TODO in the text waiting on this sweep.

**How to build it.** `python studies/static_vs_doe_replay.py --zone-limit-frac f` for the five values, about 3 minutes in total; each run writes the numbers to `summary.csv`. For the two-stage curve, run the two-stage script at the same fractions.

**Priority: high.** Cheap, it answers the most obvious reviewer question, and it clears a TODO in the text.

### Optional: year-long 22:00 attribution (not in the build)

**What it shows.** The share of added under-voltage points by time of day over the full year; the draft reports 96 % in 22:00–24:00 (`studies/NETWORK_AWARE_DISPATCH.md` §4.3).

**Why it matters.** It shows the 22:00 herding isn't a quirk of the test day. The paper currently cites the number only.

**Check first.** The walkthrough says 82 % of the scheduler's *under-voltage* falls in 22:00–24:00, while the draft says 96 % of *added* under-voltage. These are probably two different measures, both correct. Make sure the paper's sentence matches whichever one the figure plots.

**Priority: low.** Include it only if space allows or a reviewer questions whether the effect generalises.

### Suggested order to generate them

1. **Fig. 6 (savings CDF)**: no new runs, and the comparison story depends on it.
2. **Fig. 10 (sensitivity)**: about 3 minutes of runs; clears a TODO and answers the obvious reviewer question.
3. **Fig. 8 (voltage envelope)**: small script change.
4. **Year-long attribution**: only if space allows.

---

## 4. Open items

- [ ] Two-stage run on 5 Feb, all 152 households → new Table I row, and a two-stage series in Figs. 5, 6, 8 and 10.
- [ ] Decide: hybrid as a fifth regime, or a discussion paragraph only.
- [ ] If hybrid: run the perfect-forecast test (expect about 0 % gap) and the ±10 % forecast-error test.
- [ ] Decide whether water-filling needs an export-limited day or an import-side version to be tested.
- [ ] Relabel "DOE" as "Centralised VPP" throughout Tables I–II and the text.
- [ ] Confirm whether the soft centralised solve uses both slacks or only `slack_up` (affects Eq. 14 and the stacked matrices).
- [ ] Settle 145 vs 152 customers.
- [ ] Check [2] and related OPF-based DOE work before claiming the hybrid as new.
