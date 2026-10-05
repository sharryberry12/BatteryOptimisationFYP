# Revision brief for `FYP_final_paper.tex`

Written 30 Sep 2026 against the 611-line tex of 29 Sep 17:34. Line numbers
below refer to that file. This brief is meant to be handed, together with
the tex, to an assistant that does **not** have the repository: every number
it needs is in here, with its source in the last section. Nothing in this
brief is a guess; where something is unverified it says so.

**How to use it.** Work through sections 3 to 8 in order. Keep the
IEEEtran class, the notation of Section II (ℓ, g, β, π, D̅, D̲, s = 48) and
the equation labels. Do not invent numbers: if a figure the text needs is
not in this brief, leave a `\todo{}`. Remove the drafting macros only at the
very end (section 9).

---

## 1. Verdict

The tex is a faithful port of the single-day study as it stood on 29
September morning, and its Section III is in good shape: the data, the
replication, the feeder envelope, the peak-day comparison, the network
validation and the location study all read correctly and their numbers
match the run folders. Four things have moved since it was written and are
not yet in it:

1. **The two-stage (deployed-practice) method was rerun properly.** The
   tex still says its gap is "300–900 %" and that "the results below use
   the stacked solve" (line 451). Both statements are superseded: the
   allocator gained an import-side max-min rule and a need-proportional
   rule, per-household slices became soft, and the comparison on 5 Feb 2011
   now exists (section 5.1 below). Section II-C (lines 326–373), which
   says the import side is split evenly, is now wrong.
2. **A 105-day season sweep exists** (14 Jan – 30 Apr 2011, every day the
   substation record covers), with the same five regimes through the
   feeder model. It turns every single-day claim into a seasonal one and
   deserves its own subsection (section 5.2).
3. **The zone-limit sensitivity sweep has been run** and its figure
   generated; the TODO at line 590 can be closed (section 5.3).
4. **Three of the four "proposed" figures now exist** at print quality
   (section 6).

Independently of new results, the tex has structural problems a reviewer
would catch: the introduction promises transmission–distribution
co-simulation and line-current and transformer-loading checks that the
paper never delivers; Algorithm 1 is NumPy code, not pseudocode; five
`\todo` notes are unresolved; the bibliography file is missing; and the
author line is malformed. Section 3 lists every fix by line.

---

## 2. Regime names — adopt one set everywhere

The word "DOE" currently names one regime (Table I, Table II, the text) even
though every battery regime uses a DOE in some form. Rename once and use the
same labels in every table, figure caption and sentence:

| Current label | Use instead | What it is |
|---|---|---|
| No battery | **No battery** | β = 0 |
| Static limits | **Static limits** | each household alone under a fixed 1.5 kW export cap and no import cap (the tariff coordinates the fleet) |
| DOE | **Centralised VPP** | the stacked fleet QP of Eq. (14) under the feeder envelope of Eq. (15), soft form |
| — | **Two-stage DOE (rule)** | the DNSP splits the same envelope into per-household slices by a rule (equal, need-proportional, max-min) and each household solves alone with soft slices |

---

## 3. Corrections by line (must fix)

| Line(s) | Now | Change to | Why |
|---|---|---|---|
| 20 | `Sharan . Satheeshkumar, Jonathan K.F. Ah Sang, Chameli . Dias` | remove the stray periods (or supply the initials) | malformed author line |
| 25–26 | old abstract kept as a comment | delete | drafting residue |
| 50–62 | introduction promises transmission–distribution co-simulation as part of the scope and objective (three paragraphs plus "Finally, the framework is extended…") | cut to one sentence in the last paragraph of the introduction as future work, or drop entirely; also delete the TODO at line 75 and the bullet it refers to | no co-simulation code or result exists in the project |
| 48, 58 | "voltage profiles, line currents and transformer loading can be assessed" | "zone-transformer loading, circuit losses and customer voltages" | the validation reports transformer flow, losses and voltages only; no line-current (AS/NZS 3008) or transformer-loading (AS/NZS 60076.7) check exists |
| 69, 73 | hard-coded "Section~III-C" | `Section~\ref{sec_feeder_envelope}` after adding `\label{sec_feeder_envelope}` to the subsection at line 401 | cross-references must be labels |
| 326–373 (II-C) | import side "evenly split", export side by water-filling, Algorithm 1 in NumPy | rewrite per section 5.4 below: both sides, three rules, soft slices, math pseudocode | the implemented method changed; the text contradicts Section III-C |
| 381 `\todo` | 145 vs 152 | see section 4.1 | resolved |
| 391 `\todo` | replication computed on 145 | see section 4.1 | resolved |
| 449 `\todo` | do both slacks exist? | see section 4.2; delete the TODO | resolved |
| 451 | "On small ensembles its objective gap against the stacked solve is 300–900 %, and equal slices of an import cap become energy-infeasible … The results below therefore use the stacked solve" | replace with the paragraph in section 5.1 | superseded |
| 493 | "over the full year of the dataset, 96 % of the under-voltage points that the tariff-driven scheduler adds … fall in the 22:00–24:00 charging block" | "on 25 days sampled across the year, 96 % of the tariff-driven scheduler's under-voltage points fall in the 22:00–24:00 charging block (1,037 points against 301 without batteries, of which 14 % lie in that block)" | the attribution is a 25-day sample (every 15th day), not the full year; the numbers are from `attribution_by_hour.csv` |
| 505 | "Jain's fairness index … falls from 0.71 to 0.66" | add "with negative savings clipped to zero" (raw-vector values are 0.67 → 0.60) | the quoted values use the clipped definition; say which |
| 539 | "year-long sweeps put the tariff-driven scheduler at 24 % below the no-battery violation count and a flat 2 kW per household import cap at 41 % below, at a cost of 11.6 % of annual savings" | keep, but add that those sweeps are the per-household scheduler without fleet coupling | reader should not read them as VPP results |
| 586 `\todo` | co-simulation subsection? | delete; handled by the introduction edit | no results exist |
| 588–590 | "Sensitivity and Limitations" with a TODO | replace the first paragraph with section 5.3 and rename the subsection "Limitations" after inserting the season subsection (5.2) before it | sweep done |
| 604 | conclusion: "preserving 79 % of fleet savings" | add the season figure: 92 % over 105 days | new result |
| 606 | "the water-filling allocation of Section II-C is one such rule to compare against" | replace with the two-stage finding: slices that follow need come within 203 kWh of the envelope at the same household cost as the fleet solve; equal and max-min slices leave 1.4 MWh undelivered and cost 28–31 % of savings over the season | it has been compared |
| 608 | `\bibliography{references}` | provide `references.bib` (section 8) | file missing; the tex does not compile |

---

## 4. Resolved TODOs

### 4.1 145 versus 152 customers (lines 381, 391)

The current pipeline's cleaning pass (`dispatch/osqp_daily.py`, the [1]/[6]
thresholds) yields **152** customers with complete load and PV years, and
every Part B result in the paper uses those 152. The initial paper's 145
came from an earlier cleaning pass whose thresholds are not recorded. Two
acceptable resolutions:

- **Preferred:** state 152 everywhere and re-run the replication on the 152
  (one command, `python dispatch/osqp_daily.py`, then take the mean annual
  savings for fit and net from the log). Until that is done, keep the
  $364.95 / $87.02 figures but label them "on the 145-customer ensemble of
  the initial study".
- **Acceptable:** keep both counts, stated once each, with one sentence
  saying the replication predates the final cleaning pass.

Do not leave the two numbers unexplained.

### 4.2 Which slacks the soft solve carries (line 449)

`vpp_common.solve_centralised(soft=True)` carries **both** slacks: `slack_up`
on the import side (σ̄ in the tex) and `slack_lo` on the export side (σ̲).
Eq. (14) and the stacked matrices are correct as written. The penalty ρ is
10³ per kW of slack; state it, with the justification that the marginal
value of one kW of battery action in the flattening term is 2 h_k |π_k|,
which for h_k ≤ H̄ and |π_k| of a few kW stays in the low hundreds, so slack
is never used to flatten, only when the envelope cannot be met.

The per-household solver used by the two-stage method (added 29 Sep) has
the same two slacks at household level: x_i = [β_i | s̄_i | s̲_i] with
π_i − s̄_i ≤ D̅_i and π_i + s̲_i ≥ D̲_i, the same ρ. The "[β | c | s]"
formulation the TODO mentions belongs to the Part A single-household DOE
script (`osqp_daily_with_DOE.py`), which Part B does not use; drop that
sentence.

### 4.3 Co-simulation (lines 75, 586)

Not done. Move to future work in one sentence or remove.

---

## 5. New results to add

### 5.1 Two-stage DOE on the peak day (replaces line 451; add to Table I)

Text-ready paragraph for Section III-C, after Eq. (16):

> The deployed-practice alternative pre-allocates the feeder envelope into
> per-household slices and lets each household solve alone (the
> architecture of the SA Power Networks trial [3]). The import budget
> D̅_{F,k} is split by one of three rules: equally; in proportion to each
> household's forecast need (ℓ_{i,k} − g_{i,k})⁺; or max-min fairly with
> floors (Algorithm 2), where every household receives the same allowance
> except those whose unavoidable import (ℓ_{i,k} − g_{i,k} − B̅)⁺ exceeds
> it. Each household then solves its local QP with the same slack
> formulation as the stacked solve, so a slice it cannot meet is met
> best-effort and the remainder is reported as shortfall. Solved instead
> with hard slices, 74 of the 152 households are infeasible on power or
> energy and would fall back to no dispatch, which is why the earlier
> "300–900 %" objective gaps of the hard version measured the fallback
> rather than the allocation.

Add these rows to Table I (`tab_feeder_profile`), same columns:

| Regime | Peak (kW) | Peak time | Max ½-h ramp (kW) | Std. dev. (kW) | Substation excess (kWh) | Fleet savings ($/day) |
|---|---|---|---|---|---|---|
| Two-stage DOE, equal | 537 (+23 %) | 22:00 | 184 | 115 | 66 | 232.2 |
| Two-stage DOE, max-min | 539 (+23 %) | 22:00 | 187 | 116 | 65 | 234.1 |
| Two-stage DOE, need-proportional | 486 (+11 %) | 23:30 | 126 | 102 | 11 | 210.2 |

And a new short table (call it Table III-4 or fold into Table I as extra
columns) on delivery:

| Architecture | Undelivered import (kWh) | Residual feeder excess (kW) | Fleet savings ($/day) |
|---|---|---|---|
| Centralised VPP | 0 | 0 | 205.7 |
| Two-stage, equal slices | 1,419 | 99 | 232.2 |
| Two-stage, max-min with floors | 1,396 | 101 | 234.1 |
| Two-stage, need-proportional | 203 | 47 | 210.2 |

Text-ready paragraph for the end of Section III-D:

> Table III-4 asks whether the deployed-practice architecture can deliver
> the same envelope without a fleet-level solve. It cannot with size-blind
> slices: equal and max-min allocations give every household 0.66 kW at
> the substation peak against a median net demand of 1.86 kW, and the
> floors of the max-min rule bind for only 12 households, because what
> defeats a slice is energy over the six-hour window, not power in one
> interval. Those households keep almost all of their savings by leaving
> 1.4 MWh of the required reduction undelivered and re-creating a 22:00
> peak of 539 kW. Need-proportional slices come within 203 kWh of the
> envelope at the same household cost as the stacked solve. The price of
> allocating before need is revealed is therefore small if the allocation
> follows need, and large if it follows fairness in the envelope rather
> than in the outcome, which is the distinction [11] draws.

Network side of the same runs (add rows to Table II, `tab_network`):

| Regime | Peak P (MW) | Peak time | Min. V (p.u.) | Violations (of 4,800) |
|---|---|---|---|---|
| Two-stage DOE, max-min | 6.96 | 22:00 | 0.674 | 290 |
| Two-stage DOE, need-proportional | 6.28 | 23:30 | 0.699 | 287 |

Savings distribution (for the Fig. 6 caption and the fairness paragraph):
two-stage max-min gives a median household $2.00/day, Jain 0.69 (clipped;
0.64 raw), and 129 of 152 households worse off than under static limits.

### 5.2 The whole season (new subsection before "Limitations")

Suggested heading: "The Whole Season". Text-ready:

> The single day above is the hottest in the record; Table III-5 repeats
> the experiment on every day the substation record covers, 14 January to
> 30 April 2011 (105 days), with the full ensemble each day and the zone
> limit at 95 % of each day's own measured peak, so that the DNSP asks the
> fleet to shave the top of every day (37 to 693 kWh, 17.0 MWh in total).
> The same five regimes are run through the feeder model on every day.

| Regime | Mean daily peak (kW) | Days above the no-battery peak | Mean max ½-h ramp (kW) | Substation excess left (MWh) | Fleet savings ($) | Under-voltage points | Losses (MWh) |
|---|---|---|---|---|---|---|---|
| No battery | 180 | — | 48 | 17.0 | 0 | 1,582 | 82.4 |
| Static limits | 266 (+50 %) | 104 of 105 | 226 | 6.3 | 13,650 | 3,297 | 90.3 |
| Centralised VPP | 180 (0 %) | 0 | 142 | 0.0 | 12,586 (−7.8 %) | 1,530 | 75.5 |
| Two-stage DOE, max-min | 182 | 52 | 112 | 3.1 | 9,357 (−31 %) | 1,243 | 61.4 |
| Two-stage DOE, need-proportional | 171 | 18 | 105 | 1.3 | 9,816 (−28 %) | 1,229 | 65.1 |

> Four things hold across the season that the single day only suggested.
> First, the tariff herd is systematic: static limits create a new daily
> peak on 104 of 105 days, at a median 1.43 times the no-battery peak and
> up to 2.29 times, and the feeder pays for it with twice the under-voltage
> points and 10 % more losses, even though the tariff-driven discharge
> removes 63 % of the substation excess on its own. Second, the
> centralised VPP is met on every day with no shortfall and never creates
> a new peak, and over the season it costs the households 7.8 % of their
> savings rather than the 21 % of the hottest day, because on most days
> the binding term is the feeder cap at 22:00 rather than the substation.
> Third, pre-allocated slices cost three to four times as much (28 to 31 %
> of savings) and still fail to deliver: with max-min slices the fleet
> exceeds the no-battery peak on half the days and leaves 3.1 MWh at the
> substation, while need-proportional slices leave 1.3 MWh and exceed the
> peak on 18 days. The max-min floors change almost nothing relative to an
> equal split on any day. Fourth, the two-stage regimes post the best
> network figures of all, fewer under-voltage points and lower losses than
> even the centralised VPP, but for the wrong reason: the slices
> over-constrain every household, the batteries cycle less, and the LV
> network carries less current. That is a cost borne entirely by the
> households, which is why the savings column must be read with the
> network columns and not instead of them.

Figures for this subsection: `sweep_daily_peak.png` (daily peak by regime,
the herd visible as the orange line above grey on almost every day) and
`sweep_cumulative_savings.png` (the price of each regime in one line each).

### 5.3 Zone-limit sensitivity (replaces the first paragraph of lines 588–590)

Sweep of the zone limit on 5 Feb 2011, centralised VPP and two-stage
max-min, everything else as in Table I:

| f | Zone limit (MW) | Substation excess to absorb (kWh) | Centralised VPP: peak / shortfall / savings | Static: substation excess left | Two-stage max-min: peak / feeder breach / savings |
|---|---|---|---|---|---|
| 0.99 | 4.22 | 30 | 438 kW / 0 kWh / $220.8 | 0 kWh | 516 kW / 149 kWh / $213.7 |
| 0.97 | 4.13 | 273 | 438 kW / 0 kWh / $220.9 | 0 kWh | 523 kW / 164 kWh / $223.8 |
| 0.95 | 4.05 | 693 | 438 kW / 0 kWh / $205.7 | 64 kWh | 539 kW / 260 kWh / $234.1 |
| 0.92 | 3.92 | 1,589 | 562 kW / 660 kWh / ($272.3) | 528 kWh | 568 kW / 806 kWh / $248.6 |
| 0.90 | 3.83 | 2,271 | 569 kW / 1,350 kWh / ($271.7) | 1,225 kWh | 581 kW / 1,504 kWh / $255.3 |

Text-ready:

> Fig. 10 sweeps the zone limit from 99 % to 90 % of the day's measured
> peak. Down to 95 % the fleet meets the envelope with no shortfall and
> holds the feeder at the no-battery peak; the cost to the households is
> 15 % of savings when only the feeder cap binds (99 % and 97 %) and 21 %
> when the substation term binds as well (95 %). At 92 % and 90 % the
> excess to absorb (1.6 and 2.3 MWh) exceeds the 1.5 MWh the fleet holds:
> the solve reports 660 and 1,350 kWh of shortfall, and because each
> battery must still return to its initial state of charge by midnight,
> the recharge it cannot fit under the feeder cap spills into a new 22:00
> peak of 562 to 569 kW. The two-stage max-min slices breach the feeder
> envelope at every level, 149 kWh even at 99 % where the substation is
> barely stressed, because the feeder cap itself is what size-blind slices
> cannot share.

Do not present the $272 "savings" at 92 % and 90 % as a gain: in that
regime the envelope is not met, so the fleet is discharging into the peak
tariff without paying the recharge constraint. Report those two points as
"fleet capacity exceeded".

### 5.4 Rewrite of Section II-C "Dynamic Operating Envelope Calculation"

Replace lines 326–373 with a subsection that (a) states that the feeder
envelope of Eq. (15) is split into per-household slices D̲_{i,k}, D̅_{i,k}
with Σ_i D̲_{i,k} ≥ D̲_{F,k} and Σ_i D̅_{i,k} ≤ D̅_{F,k}; (b) gives the rules
for both sides:

| Rule | Export budget −D̲_{F,k} | Import budget D̅_{F,k} |
|---|---|---|
| equal | 1/N each | 1/N each |
| need-proportional | ∝ (g_{i,k} − ℓ_{i,k})⁺ (forecast surplus) | ∝ (ℓ_{i,k} − g_{i,k})⁺ (forecast need) |
| max-min | water-filling against each household's physical export cap (B̅ − ℓ_{i,k} + g_{i,k})⁺, Algorithm 1 | water-filling with floors: same allowance λ for all, except households whose unavoidable import (ℓ_{i,k} − g_{i,k} − B̅)⁺ exceeds λ, which receive their floor; λ set so the slices sum to the budget (Algorithm 2) |

(c) rewrites Algorithm 1 as mathematical pseudocode, not NumPy. A faithful
version:

```
Algorithm 1  Water-filling export allocation (one interval)
Input:  budget b ≥ 0, caps c ∈ R^N_{≥0}
Output: allocation a ∈ R^N
a ← 0;  r ← b;  A ← { i : c_i > 0 }
while r > 0 and A ≠ ∅ do
    δ ← r / |A|
    for i ∈ A do  a_i ← a_i + min(δ, c_i − a_i)
    r ← b − Σ_i a_i
    A ← { i ∈ A : a_i < c_i }
end while
return a
```

```
Algorithm 2  Floor-filling import allocation (one interval)
Input:  budget b ≥ 0, floors f ∈ R^N_{≥0} with Σ_i f_i ≤ b
Output: allocation a ∈ R^N with Σ_i a_i = b, a_i ≥ f_i
P ← ∅                                     (households pinned at their floor)
repeat
    λ ← ( b − Σ_{i∈P} f_i ) / ( N − |P| )
    Q ← { i ∉ P : f_i > λ }
    P ← P ∪ Q
until Q = ∅
a_i ← f_i for i ∈ P,  a_i ← λ otherwise
return a
```

(if Σ_i f_i > b every household receives its floor and the deficit is
reported as shortfall by the household solves); and (d) states that each
household then solves Eq. (11) with its slice and the two slacks of 4.2.

---

## 6. Figures: what exists, where, and what to do with each placeholder

All paths are relative to the repository root. "Print" means 300 dpi PNG
plus PDF, sized for one IEEE column; "draft" means 150 dpi PNG.

| Placeholder / label | Status | File | Notes for the caption |
|---|---|---|---|
| `fig_replication` | reuse from initial paper | initial paper Fig. 2 | unchanged |
| `fig_doe_derivation` | exists, draft | `outputs/runs/static_vs_doe_2011-02-05_20260929-111819/figures/doe_derivation.png` | caption at line 464 is correct |
| `fig_feeder_profile` | exists, draft; **regenerate with the two-stage series** | `outputs/runs/static_vs_doe_2011-02-05_20260929-175153/figures/feeder_profile.png` (that run includes `two_stage_maxmin` and `two_stage_equal`) | caption at line 499 correct; add the two-stage sentence: "two-stage slices re-create a 539 kW peak at 22:00" |
| `fig_savings_cdf` | **exists, print** | `outputs/figures/paper/savings_cdf.png` / `.pdf` (three series: static, centralised, two-stage max-min) | caption: "…136 of 152 households are worse off under the centralised VPP; Jain's index (negatives clipped) falls from 0.71 to 0.66; two-stage max-min keeps a median of $2.00 but leaves 129 households below static" |
| `fig_zone_tx` | exists, draft | `outputs/runs/static_vs_doe_2011-02-05_20260929-111819/figures/zone_transformer_power.png` | caption at line 545 correct; the subtitle inside the figure already carries the "shapes not levels" caveat |
| `fig_voltage_env` | **exists, print** | `outputs/figures/paper/voltage_envelope.png` / `.pdf` (four regimes) | caption at line 553; add "the static-limits minimum of 0.648 p.u. falls at 22:00; two-stage max-min reaches 0.674 p.u. at the same hour" |
| `fig_location` | exists, draft | `outputs/runs/battery_location_2011-02-05_20260929-125039/figures/location_value.png` (or `transformer_and_losses.png`) | caption at line 582 correct |
| `fig_sensitivity` | **exists, print** | `outputs/figures/paper/zone_limit_sensitivity.png` / `.pdf` | caption: "…centralised VPP and two-stage max-min; the fleet's 1,520 kWh runs out between 95 % and 92 %" |
| new: season daily peak | exists, draft | `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/figures/sweep_daily_peak.png` | "Daily peak of the feeder power profile by regime over 105 days; shading marks days on which the substation exceeded its limit" |
| new: season savings | exists, draft | same folder, `sweep_cumulative_savings.png` | "Cumulative fleet savings by regime over the season" |
| optional: 22:00 attribution | exists, print | `outputs/figures/paper/attribution_2200.png` / `.pdf` | use only if a reviewer questions whether the herd generalises; the sampled-days caveat must be in the caption |
| `system_architecture.png`, `residential_model.png` | referenced at lines 80 and 93, **not in the repository** | supply from the LaTeX project | — |

The draft-quality figures are produced by `studies/static_vs_doe_replay.py`,
`studies/battery_location_study.py` and `studies/doe_day_sweep.py` at
150 dpi; for submission re-save at 300 dpi (one `dpi=` argument in each
script) or accept them as they are at column width.

Recommended figure order for the paper: derivation (X-1), feeder profile
(X-2), savings CDF (X-3), zone transformer (X-4), voltage envelope (X-5),
location value (X-6), season daily peak (X-7), sensitivity (X-8). Drop the
attribution figure and the cumulative-savings figure if the page limit
bites; their numbers are in the text.

---

## 7. Claims to soften or remove

- **Co-simulation** (introduction, lines 50–62; TODO lines 75, 586): not
  done. Future work at most.
- **Line currents and transformer loading against AS/NZS 3008 and 60076.7**
  (lines 48, 58; also the initial paper's abstract): not checked in the
  current pipeline. Say voltages against AS IEC 60038, zone-transformer
  flow and losses.
- **"Over the full year"** for the 96 % figure (line 493): it is a 25-day
  sample.
- **"300–900 %"** (line 451): a property of the hard-slice fallback, not of
  the allocation; replaced by section 5.1.
- **The "DOE must be dynamic" paragraph** (line 515) is sound; keep it.
- **The hybrid scheme** described in the project's planning notes (the
  DNSP runs the fleet solve and issues each household's optimal profile
  as its envelope) has **not been run**. Its convexity argument is
  plausible but untested. Mention it only as future work; do not state
  that it recovers the centralised optimum.
- **Jain's index** values are the clipped-negative definition; say so
  (line 505) or switch to raw values (0.67 → 0.60).
- The feeder-scale magnitudes (5.64 MW no-battery peak) exceed the whole
  measured substation (4.26 MW); the text already says shapes not levels
  (line 522). Keep that sentence wherever a feeder-scale MW appears.

---

## 8. References

`\bibliography{references}` is called but no `references.bib` exists in the
repository. Twelve keys are cited. Entries (from the initial paper's list
unless marked new):

| Key | Entry |
|---|---|
| `ratnam2015optimization` | E. L. Ratnam, S. R. Weller, C. M. Kellett, "An optimization-based approach to scheduling residential battery storage with solar PV: assessing customer benefit," *Renew. Energy*, vol. 75, pp. 123–134, 2015 |
| `liu2023robust` | B. Liu and J. H. Braslavsky, "Robust dynamic operating envelopes for DER integration in unbalanced distribution networks," *IEEE Trans. Power Syst.*, 2024 — the initial paper gives 2024; check the year against the key |
| `sapn2022flexible` | SA Power Networks, "Flexible exports for solar PV: trial report," 2022, arena.gov.au |
| `ratnam2017residential` | E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray, "Residential load and rooftop PV generation: an Australian distribution network dataset," *Int. J. Sustain. Energy*, vol. 36, no. 8, pp. 787–806, 2017 |
| `geth2021representative` | F. Geth and T. Brinsmead, "The Representative Low Voltage Networks data and models package – introduction manual," CSIRO, 2021 |
| `b8` | Standards Australia, *Standard voltages*, AS IEC 60038:2022 |
| `b11` | AEMO, "The fairness in dynamic operating envelope objectives report," 2023 |
| `b12` | AEMO, "VPP demonstrations: knowledge sharing report," 2021 |
| `b13` | SA Power Networks, "Advanced VPP grid integration: final knowledge sharing report," ARENA project G00854, 2021 |
| `stellato2020osqp` (new) | B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd, "OSQP: an operator splitting solver for quadratic programs," *Math. Program. Comput.*, vol. 12, pp. 637–672, 2020 |
| `jain1984quantitative` (new) | R. Jain, D. Chiu, W. Hawe, "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," DEC Research Report TR-301, 1984 |
| `ausgrid_zone` (new) | Ausgrid zone-substation load data (15-minute MW, Jesmond 132/11 kV, 2010–11). **The exact title, URL and access date must be supplied by the authors; the repository records only the file.** |

Also: the initial paper's [9] and [10] (AS/NZS 60076.7 and 3008.1.1) are
no longer cited once the line-current claim is removed; do not add them
back unless the checks are added.

---

## 9. Mechanics and style

- Delete the `\todo` and `\figph` macro definitions (lines 10–13) and every
  use of them once the placeholders are replaced.
- Tables I and III-5 are wide; keep `table*` for Table I and use it for the
  season table too.
- Use one unit per quantity: kW at ensemble scale, MW at feeder scale and
  for the substation; say "ensemble scale" or "feeder scale" the first time
  each appears in a subsection.
- Section II-A "Overview" bullets: the sub-transmission bullet should say
  the measurement is used to derive the import limit of the feeder
  envelope, and the aggregation bullet should name the two architectures
  (centralised VPP, two-stage DOE).
- The introduction is long (lines 33–62); after removing the co-simulation
  paragraphs, one paragraph on what the paper actually does (replicate,
  add DOE rows, lift to a feeder envelope from measured headroom, compare
  static / centralised / two-stage, validate on the feeder, season sweep)
  is enough.
- "Static limits" is defined in the bullet at line 470; the two-stage
  definition belongs in the same list once section 5.1 is added.
- Keep every number to the precision in this brief (integers for kW and
  kWh, one decimal for $ per day, three for p.u.).

---

## 10. Where every number comes from

| Numbers | Source in the repository |
|---|---|
| Peak-day Table I rows for no battery, static, centralised; Table II; savings statistics; Jain (clipped and raw); 136 / 16 households | `outputs/runs/static_vs_doe_2011-02-05_20260929-111819/summary.csv`, `manifest.json`, `dispatch_*.csv`; recomputed in `outputs/figures/paper/checks.md` |
| Peak-day two-stage rows (equal, max-min), their network rows, voltage minimum 0.674 | `outputs/runs/static_vs_doe_2011-02-05_20260929-175153/` and `…_175309/summary.csv` (run with `--two-stage-rules maxmin,equal`, soft slices) |
| Peak-day two-stage need-proportional row and its network row | 5 Feb rows of `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/sweep_results.csv` |
| Hard-slice dropouts 74 / 152 and the undelivered / residual / savings table | `outputs/figures/vpp/two_stage_doe_allocation/jesmond_2011-02-05_{hard,soft}/rule_comparison.csv` |
| Season table and the four seasonal findings (104 / 105 days, 1.43× median, 2.29× max, 63 %, 52 / 18 days) | `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/sweep_summary.csv` (scope `all`) and `sweep_results.csv` |
| Sensitivity table | `outputs/figures/paper/zone_limit_sweep.csv` (from the five `static_vs_doe_2011-02-05_20260929-1752xx/1753xx/1754xx` runs) |
| Attribution 95.8 % / 96.5 % / 1,037 / 301 / 14 % on 25 sampled days | `outputs/figures/paper/attribution_by_hour.csv` and `checks.md` |
| Location study (losses 4,655 / 4,220 / 4,654 kWh etc.) | `outputs/runs/battery_location_2011-02-05_20260929-125039/summary.csv` |
| GridLAB-D agreement 0.001 % / 0.010 % / 0.020 % | `network/MODEL_VERIFICATION.md`, Level 4 |
| Year-long per-household sweeps (−24 %, −41 %, −11.6 %) | `studies/NETWORK_AWARE_DISPATCH.md` §2, §4.4 |
| Peak-duty numbers (772 kW, 16:00–23:30 event) | `studies/PEAK_DUTY_FINDINGS.md` |
| Customer count 152, cleaning | `dispatch/osqp_daily.py` (the [1]/[6] rules); the 145 of the initial paper is not reproducible from the current code |
| Soft penalty ρ = 10³, both slacks | `vpp/vpp_common.py` (`SOFT_PENALTY_DEFAULT`, `solve_centralised`, `HouseholdSolver(soft=True)`) |
| Allocation rules and Algorithms 1–2 | `vpp/two_stage_doe_allocation/two_stage_doe_allocation.py` (`water_fill`, `floor_fill`, `allocate`) |
