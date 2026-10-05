# Draft — Section III "Numerical Simulation" for the final paper

Drafted 2026-09-29 from the repo's experiments. Notation follows the initial
paper: load ℓ, PV g, battery β (positive on discharge), grid power π
(positive on import), import/export limits D̅ ≥ 0 and D̲ ≤ 0, s = 48 intervals
of Δ = 30 min, battery C = 10 kWh, χ₀ = 0.5C, B̅ = −B̲ = 5 kW. Reference
numbers ([1]–[13]) are the initial paper's. Every figure in the text is a
placeholder "Fig. X-n"; the recommended figure list and captions follow the
draft, and Appendix A maps every number to the run folder it came from.

Three statements in the initial paper no longer match the code and should be
revised where the final paper reuses that text:

1. Batteries and PV are **not** injected as `Storage` / `PVSystem` elements.
   Two or more `Storage` elements collapse the DSS engine on this network
   (`network/MODEL_VERIFICATION.md`, defect 1), so every household's net
   grid profile π = ℓ − g − β is attached to its load as a 48-point
   loadshape. Section III-E below describes it that way.
2. The clean ensemble is **152** customers in the current pipeline, not 145.
   The authors should reconcile which cleaning thresholds produced each count
   before quoting either.
3. "No voltage excursions on any bus" was true of the early, lightly loaded
   model. The verified model at full replication does produce violations on
   the peak day (Table III-2); the text below reports them rather than
   claiming zero.

---

## III. NUMERICAL SIMULATION

This section reports the simulation study in four steps. We first restate
the data and the single-household replication of [1]. We then lift the
connection-point DOE of Section II to a feeder-level envelope that couples
the households of a VPP, and derive that envelope from a measured
zone-substation loading profile. The core experiment compares the fleet
under today's static connection limits with the fleet under the DOE on the
peak-demand day of the dataset, and validates both on the Elermore Vale
OpenDSS model. Finally we show that the network value of the same dispatch
depends on where the batteries sit.

### A. Data, ensemble and test day

All experiments use the Ausgrid "Solar Home Electricity" dataset [6]
(300 customers, 1 July 2010 to 30 June 2011, 30-minute gross PV and load).
Applying the cleaning rules of [1] leaves an ensemble of N = 152 customers
with complete load and PV records. Battery parameters follow [1, Sec. 6.1]:
C = 10 kWh, χ₀ = 0.5C and B̅ = −B̲ = 5 kW. The tariff is the three-band
time-of-use schedule of [1] (off-peak $0.03/kWh 22:00–07:00, shoulder
$0.06/kWh, peak $0.30/kWh 14:00–20:00) with a flat $0.40/kWh gross feed-in
credit (metering topology 1 of [1]). Forecasts of ℓ and g are taken as
perfect, as in [1].

The test day is Saturday 5 February 2011, the February 2011 NSW heatwave.
Over the three-year extension of the dataset it is the day of maximum
aggregate net demand (772 kW across 300 customers, 2.57 kW per household at
18:30), and the worst firm-capacity event at a 70 % threshold runs from
16:00 to 23:30 on that day (7.5 h, 1,040 kWh above threshold). Over our
152-customer ensemble the no-battery aggregate Σᵢ πᵢ peaks at 438 kW
(2.88 kW per household) at 20:00, with an overnight minimum of 96 kW.

The same day is covered by Ausgrid's measured loading of the Jesmond
132/11 kV zone substation, which supplies the Elermore Vale feeder. The
15-minute MW record (averaged to 30 minutes) peaks at 4.26 MW at 16:30 and
stays above 95 % of that peak from 15:00 to 21:00. This measurement is what
lets us build a DOE from real network headroom rather than from an assumed
shape.

### B. Single-household replication of [1]

With the DOE rows of A₁ and b₁ suppressed, the OSQP scheduler reproduces the
annual-savings distributions of [1, Fig. 7]: mean annual savings of
$364.95/yr under the gross feed-in tariff and $87.02/yr under net metering,
against $348/yr and $90/yr reported in [1, Table 2], with the same long left
tail of net-metering customers who lose money. A full year of 48-interval
dispatches for one customer solves in 9–11 s on one core with warm-starting,
versus roughly 250 s reported for quadprog in [1]. *(Retained from the
initial paper; update the 145/152 count as noted above.)*

### C. From connection-point DOEs to a feeder envelope

A DNSP does not experience households one at a time: the quantity it must
keep inside the network's limits is the feeder-head power Σᵢ πᵢ. We
therefore lift the per-customer envelope of Section II to a feeder-level
envelope

  D̲_F ≤ Σᵢ₌₁ᴺ πᵢ ≤ D̅_F,  (III-1)

with D̲_F ≤ 0 capping aggregate export and D̅_F ≥ 0 capping aggregate
import, element-wise over the s intervals. Each household keeps the local
problem of Section II (rate limits, SOC band, daily neutrality) with its
weights hᵢ frozen at the values the greedy heuristic of [1] chose for it;
the fleet problem is the N local QPs stacked block-diagonally with the s
coupling rows of (III-1). Because only those s rows tie the blocks together,
the stacked problem keeps the sparsity OSQP exploits, and N = 152 households
solve in about 2 s. We solve it in a *soft* form, adding a slack on each side
of (III-1) with a linear penalty, so that an envelope the 5 kW / 10 kWh fleet
cannot physically meet is reported as import shortfall rather than as an
infeasible solve. We also implemented the deployed-practice alternative, in
which the DNSP pre-allocates (III-1) into per-household slices and each
household solves alone (the architecture of the SA Power Networks trial
[3]). The import budget D̅_{F,k} is split by one of four rules: equally;
in proportion to each household's forecast need (ℓ − g)⁺; or max-min
fairly with floors, where every household receives the same allowance
except those whose unavoidable import (ℓ − g − B̅)⁺ exceeds it. Each
household then solves the local QP with the same slack formulation as the
stacked solve, so a slice it cannot meet is met best-effort and the
remainder is reported as shortfall. Section III-D compares the two
architectures on the test day.

The import side of the envelope is derived from the measured zone
substation. Let L_k be the measured substation loading, P⁰_k = Σᵢ (ℓ_{i,k} −
g_{i,k}) the ensemble's no-battery import, and treat the ensemble as one
slice of the substation load with everything else inflexible. If the DNSP
wants to hold the substation at or below a level C_zone, the headroom
available to the flexible slice in interval k is C_zone − (L_k − P⁰_k), and
the DOE import limit is

  D̅_{F,k} = min( C_zone − (L_k − P⁰_k), C_F ),  (III-2)

where C_F is the feeder's own planning cap, set here to the no-battery peak
so that the fleet may never create a new feeder peak. Below the baseline
where the substation is over its limit (the fleet must shed), above it up to
C_F where the substation has spare capacity (the fleet may charge). We take
C_zone = 95 % of the day's measured peak (4.05 MW); the substation then
exceeds C_zone for twelve intervals holding 693 kWh of excess, against a
fleet energy of 1,520 kWh. The resulting envelope falls to 101 kW
(0.66 kW per household) at 16:30 and relaxes to C_F = 438 kW outside the
15:00–21:00 window (Fig. X-1).

Two regimes are compared against the no-battery baseline:

- **Static limits** — today's practice. Each household optimises its own
  bill under a fixed export limit of 1.5 kW (the static limit of [3]) and
  no import limit; nothing coordinates the import side.
- **DOE** — the same fleet coupled by (III-1) with D̅_F from (III-2). The
  export side is the same flat cap in both regimes, so the regimes differ
  only through the time-varying import limit.

### D. Static limits versus the DOE on the peak day

Table III-1 and Fig. X-2 give the feeder power profile Σᵢ πᵢ for the three
regimes at ensemble scale.

**Table III-1. Feeder power profile Σᵢ πᵢ, 5 Feb 2011, N = 152.**

| | Peak (kW) | Peak time | Max ½-h ramp (kW) | Std. dev. (kW) | Substation excess (kWh) | Fleet savings ($/day) |
|---|---|---|---|---|---|---|
| No battery | 438 | 20:00 | 49 | 121 | 693 | 0 |
| Static limits | 594 (+36 %) | 22:00 | 261 | 129 | 64 | 260.5 |
| DOE | 438 (0 %) | 21:00 | 149 | 101 | 0 | 205.7 |

Under static limits the fleet does relieve the substation: the peak tariff
band (14:00–20:00) already drives 958 kWh of discharge through the stress
window, cutting the substation excess from 693 kWh to 64 kWh. The residual
excess is concentrated at 16:00–17:30 (53 kWh), the substation's own peak,
where tariff-driven discharge alone is not enough, with a further 11 kWh in
20:00–21:00 after the peak band ends. What the static limits cannot prevent
is what happens two hours later. When the off-peak band opens at 22:00 every battery
begins recharging at once, 509 kWh are drawn between 22:00 and 24:00, and the
feeder-head profile steps to a new peak of 594 kW at 22:00, 36 % above the
no-battery peak and 2.4 h after it, with a half-hour ramp of 261 kW. The
household optimum is collectively the worst hour of the feeder's day. This
herding is not a property of the test day: over the full year of the
dataset, 96 % of the under-voltage points that the tariff-driven scheduler
adds to the Elermore Vale model fall in the 22:00–24:00 charging block
(Section III-E).

The DOE removes the new peak. The feeder cap C_F holds the 22:00–24:00 block
at the no-battery peak of 438 kW (204 kWh of recharge in that block instead
of 509 kWh, the rest deferred overnight), the headroom term holds the
substation excess at zero, and the profile is flatter by every measure: the
standard deviation falls from 129 kW to 101 kW and the largest ramp from
261 kW to 149 kW. Inside the window the fleet sits on the envelope only
where the tariff alone is not enough: at 15:30–17:30, when the substation is
at its peak, and at 20:00–21:00 after the peak band ends; elsewhere the
tariff-driven discharge already lies below the limit. From 21:00 the fleet
rides the feeder cap while it recharges.

The DOE costs the households 21 % of their savings on this day ($260.5 to
$205.7 across the fleet, mean $1.71 to $1.35 per household). The cost is not
evenly spread: 136 of the 152 households are worse off, by up to $2.30/day,
the median household falls from $2.60 to $1.59, and Jain's fairness index of
the savings vector falls from 0.71 to 0.66. Who pays for feeder-level
relief is exactly the allocation question raised in [11]; the stacked solve
answers it by minimising the fleet's tariff-weighted objective, not by any
notion of equity, which motivates the allocation rules of the two-stage
method as future work.

The comparison also shows why the DOE must be dynamic. A static import
limit that delivered the same substation relief would have to equal the
minimum of (III-2), 101 kW for the whole ensemble, for the whole day. That
is 5 kW above the overnight no-battery load of 96 kW, so the fleet could
not recharge at all. The DOE tightens to that value only for the one
interval that needs it and relaxes to the feeder cap for the other 36.

Finally, Table III-4 asks whether the deployed-practice architecture can
deliver the same envelope without a fleet-level solve. It cannot with
size-blind slices: equal and max-min allocations give every household
0.66 kW at the substation peak against a median net demand of 1.86 kW,
and the floors of the max-min rule bind for only 12 households, because
what defeats a slice is energy over the six-hour window, not power in
one interval. Those households keep almost all of their savings by
leaving 1.4 MWh of the required reduction undelivered. Need-proportional
slices come within 203 kWh of the envelope at the same household cost as
the stacked solve. The price of allocating before need is revealed is
therefore small if the allocation follows need, and large if it follows
fairness in the envelope rather than in the outcome, which is the
distinction [11] draws.

**Table III-4. Delivering the DOE without a fleet solve, 5 Feb 2011.**

| Architecture | Undelivered import (kWh) | Residual feeder excess (kW) | Fleet savings ($/day) |
|---|---|---|---|
| Stacked fleet QP | 0 | 0 | 205.7 |
| Two-stage, equal slices | 1,419 | 99 | 232.2 |
| Two-stage, max-min with floors | 1,396 | 101 | 234.1 |
| Two-stage, need-proportional | 203 | 47 | 210.2 |

### E. Network validation on the Elermore Vale feeder

Each regime's per-household profiles π_{i,k} are injected into the OpenDSS
model of the Elermore Vale 11 kV feeder (one 132/11 kV zone substation, 23
distribution transformers, 1,785 LV loads), translated at runtime from the
GridLAB-D sources. The translation is verified at four levels against the
GridLAB-D reference, with an agreement of 0.001 % at 11 kV and 0.010 % mean
(0.020 % worst) at LV. Since 152 profiles are replicated over 1,785 loads
(×11.7), the feeder envelope scales by the same factor and the model's
magnitudes are an upper bound: the modelled no-battery peak of 5.64 MW at
the zone transformer exceeds the 4.26 MW measured for the whole Jesmond
substation. Levels are therefore not comparable with the measurement, but
shapes are (Fig. X-3). Batteries and PV enter as each load's net loadshape
π = ℓ − g − β; a 48-step daily power flow records the zone-transformer
power and the voltage at 100 monitored loads (4,800 load-intervals), checked
against the +10 % / −6 % band of AS IEC 60038 [8].

**Table III-2. Zone-transformer flow and customer voltages, 5 Feb 2011,
feeder scale (×11.7).**

| | Peak transformer P (MW) | Peak time | Min. voltage (p.u.) | Violation points (of 4,800) |
|---|---|---|---|---|
| No battery | 5.64 | 20:00 | 0.749 | 340 |
| Static limits | 7.66 | 22:00 | 0.648 | 297 |
| DOE | 5.63 | 22:30 | 0.714 | 288 |

The transformer sees what the aggregate predicted. Static limits push the
feeder head to 7.66 MW at 22:00 and the worst customer voltage to 0.648 p.u.
during the charging block, even though the daily violation count falls,
because the evening discharge relieves more intervals than the herd
damages. The DOE keeps the feeder head at the no-battery level, lifts the
worst voltage back to 0.714 p.u. and gives the lowest violation count of the
three. The residual violations are the model's exaggeration of a real
feeder condition (a boost-tap distribution transformer and the replicated
loading); the year-long sweeps put the tariff-driven scheduler at 24 %
below the no-battery violation count and a flat 2 kW/household import cap at
41 % below, at 11.6 % of annual savings, which brackets the single-day DOE
result above.

### F. Where the flexibility sits

A VPP is often abstracted as a single dispatchable plant at the substation.
To test what that abstraction hides, the DOE dispatch of Section III-D was
injected a second way: every load carries its no-battery profile and one
constant-power generator at the 11 kV feeder-head bus injects Σᵢ β_{i,k}
at feeder scale, so both representations move exactly the same energy
through the zone transformer. Losses were recorded at every interval by
solving the day step by step.

**Table III-3. Same dispatch, two locations, 5 Feb 2011.**

| | Daily circuit losses (kWh) | Change | Under-voltage points | Transformer import (MWh) |
|---|---|---|---|---|
| No battery | 4,655 | — | 338 | 67.25 |
| Distributed, behind the meter | 4,220 | −9.4 % | 287 | 66.88 |
| Aggregate generator at 11 kV | 4,654 | −0.03 % | 339 | 67.25 |

The transformer flows differ by at most 309 kW, which is the loss
difference. Everything else the fleet does for the network happens below
the 11 kV bus: distributed batteries cut daily losses by 436 kWh and
under-voltage points by 15 %, while the identical dispatch at the feeder
head changes neither (Fig. X-4). The loss saving is concentrated in the
15:00–21:00 discharge window and partly given back after 21:00, when
recharging raises LV currents. A feeder-head abstraction therefore
reproduces the substation and market view of a VPP but none of its
distribution-network value, which is the value a DOE is designed to unlock.

### G. The whole season

The single day above is the hottest in the record; Table III-5 repeats the
experiment on every day the substation record covers, 14 January to 30
April 2011 (105 days), with the full ensemble each day and the zone limit
at 95 % of each day's own measured peak, so that the DNSP asks the fleet
to shave the top of every day (37 to 693 kWh, 17.0 MWh in total). The
deployed-practice architecture is included with soft slices under the
max-min and need-proportional rules; the same five regimes are run through
the feeder model on every day.

**Table III-5. 105 days, 14 Jan to 30 Apr 2011, N = 152, zone limit 95 %
of each day's peak. Peaks and ramps are daily means of the feeder profile
Σᵢ πᵢ; the last three columns are from the feeder model.**

| Regime | Daily peak (kW) | Days above the no-battery peak | Max ½-h ramp (kW) | Substation excess left (MWh) | Fleet savings ($) | Under-voltage points | Losses (MWh) |
|---|---|---|---|---|---|---|---|
| No battery | 180 | — | 48 | 17.0 | 0 | 1,582 | 82.4 |
| Static limits | 266 (+50 %) | 104 | 226 | 6.3 | 13,650 | 3,297 | 90.3 |
| DOE, centralised | 180 (0 %) | 0 | 142 | 0.0 | 12,586 (−7.8 %) | 1,530 | 75.5 |
| DOE, two-stage max-min | 182 | 52 | 112 | 3.1 | 9,357 (−31 %) | 1,243 | 61.4 |
| DOE, two-stage need-proportional | 171 | 18 | 105 | 1.3 | 9,816 (−28 %) | 1,229 | 65.1 |

Four things hold across the season that the single day only suggested.
First, the tariff herd is systematic: static limits create a new daily
peak on 104 of 105 days, at a median 1.43 times the no-battery peak and up
to 2.29 times, and the feeder pays for it with twice the under-voltage
points and 10 % more losses, even though the tariff-driven discharge
removes 63 % of the substation excess on its own. Second, the centralised
DOE is met on every day with no shortfall and never creates a new peak,
and over the season it costs the households 7.8 % of their savings rather
than the 21 % of the hottest day, because on most days the binding term
is the feeder cap at 22:00 rather than the substation. Third,
pre-allocated slices cost three to four times as much (28 to 31 % of
savings) and still fail to deliver: with equal or max-min slices the
fleet exceeds the no-battery peak on half the days and leaves 3.1 MWh at
the substation, while need-proportional slices leave 1.3 MWh and exceed
the peak on 18 days. The max-min floors change almost nothing relative to
an equal split on any day, confirming that the slice problem is energy
over the window, not power in an interval. Fourth, the two-stage regimes
post the best network figures of all, fewer under-voltage points and
lower losses than even the centralised DOE, but for the wrong reason: the
slices over-constrain every household, the batteries cycle less, and the
LV network carries less current. That is a cost borne entirely by the
households, which is why the savings column must be read with the
network columns and not instead of them.

### H. Limitations

Forecasts are perfect and the DOE is computed from the measured substation
profile of the same day; a day-ahead DOE would rest on a forecast of L_k.
The ensemble is treated as a slice of the substation load with all other
load inflexible, and the two data sources' time bases (standard versus
daylight-saving time) could not be verified, so a half-hour to one-hour
offset between them is possible. Results are for one day at one zone-limit
level; the 95 % level was chosen so that the excess energy (693 kWh) lies
within the fleet's capacity, and at 90 % (about 2 MWh of excess) the fleet
cannot meet the envelope and the soft solve reports shortfall. Finally the
replicated feeder overstates loading, so violation counts and losses should
be read relative to each other, not as absolute predictions for Elermore
Vale.

---

## Recommended figures

Ordered by how much of the story each one carries. Existing files are
listed by run folder; the proposed additions are cheap to produce and would
close the remaining gaps.

| # | Figure | Source | Why it is in the paper |
|---|---|---|---|
| X-1 | **Where the DOE comes from**: measured Jesmond MW with the zone limit and shaded excess (top); ensemble baseline import with the stepped DOE limit and feeder cap (bottom) | `outputs/runs/static_vs_doe_2011-02-05_20260929-111819/figures/doe_derivation.png` | Grounds the envelope in a real DNSP measurement, which is the novelty over an assumed TOU shape; makes (III-2) visual |
| X-2 | **Feeder power profile Σᵢ πᵢ**, three regimes, DOE limit and stress window, peaks annotated | same folder, `feeder_profile.png` | The hero figure: the 22:00 herd under static limits and its absence under the DOE, in one frame |
| X-3 | **Zone-transformer flow**, OpenDSS model for the three regimes with the measured substation curve overlaid | same folder, `zone_transformer_power.png` | Closes the loop from the QP to the physical feeder and back to the real substation; carries the "compare shapes, not levels" caveat in its subtitle |
| X-4 | **Location value**: paired bars of daily losses and under-voltage points for no battery / distributed / aggregate | `outputs/runs/battery_location_2011-02-05_20260929-125039/figures/location_value.png` (the two-panel `transformer_and_losses.png` if space allows the time series) | Makes the point that a substation-level VPP abstraction has no network value; three bars, no reading effort |
| X-5 | **Daily feeder-profile peak over the season**, five regimes, 105 days | `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/figures/sweep_daily_peak.png` | Shows the herd is systematic (static above no battery on 104 of 105 days) and that the centralised DOE tracks the no-battery peak exactly; the two-stage lines breaking above it are the allocation failures |
| X-6 | **Cumulative fleet savings over the season** | `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/figures/sweep_cumulative_savings.png` | The price of each regime in one picture: centralised DOE 8 % below static, two-stage 28 to 31 % below. (`sweep_zone_exceedance.png` in the same folder is the substation-side companion if space allows.) |

Proposed additions, in priority order:

1. **Zone-limit sensitivity** (new): run `studies/static_vs_doe_replay.py
   --zone-limit-frac f` for f ∈ {0.99, 0.97, 0.95, 0.92, 0.90} and plot,
   against f, the feeder peak, the fleet savings and the import shortfall.
   This turns one operating point into a cost curve for the DNSP: how much
   substation relief the fleet can buy, what it costs the households, and
   where the 1,520 kWh fleet runs out. About three minutes of compute; the
   script already writes every number the plot needs to `summary.csv`.
2. **Per-household savings under static versus DOE** (new): an empirical
   CDF or paired histogram of the 152 daily savings, with the Jain index in
   the legend. The dispatch CSVs already carry `daily_savings`. This is the
   figure that speaks to the AEMO fairness question [11] and justifies the
   allocation-rule future work.
3. **Customer voltage envelope for the three regimes on the peak day**
   (small addition to `static_vs_doe_replay.py`): min/max band across the
   monitored loads with the statutory limits, as the location study already
   draws. Table III-2 says it in numbers; a panel would show the 22:00
   voltage collapse under static limits directly.
4. **Year-long 22:00 attribution** (existing, from
   `studies/NETWORK_AWARE_DISPATCH.md` §4.3): the figure showing that 96 %
   of the scheduler's under-voltage points fall in 22:00–24:00. Include it
   if the reviewers might read the peak-day herd as a one-off; otherwise
   cite the number.

Suggested captions:

- Fig. X-1. Derivation of the DOE import limit on 5 February 2011. Top:
  measured loading of the Jesmond 132/11 kV zone substation with the zone
  limit C_zone = 4.05 MW (95 % of the day's peak); the shaded excess is what
  the fleet must absorb. Bottom: the ensemble's no-battery import and the
  resulting limit D̅_F from (III-2), which relaxes to the feeder cap outside
  the stress window.
- Fig. X-2. Feeder power profile Σᵢ πᵢ for 152 households on 5 February
  2011 under no battery, static connection limits and the DOE. Static
  limits create a 594 kW charging peak at 22:00, 36 % above the no-battery
  peak; the DOE holds the profile at the feeder cap and removes the
  substation excess.
- Fig. X-3. Active power through the 132/11 kV zone transformer of the
  Elermore Vale model for the three regimes, with the measured Jesmond
  loading (all feeders) overlaid. The model replicates 152 profiles over
  1,785 loads, so shapes rather than levels are comparable.
- Fig. X-4. Network value of the same DOE dispatch injected behind the
  meters versus as one generator at the 11 kV feeder-head bus: daily
  circuit losses (left) and under-voltage load-intervals (right).

---

## Appendix A — where every number comes from

| Number | File |
|---|---|
| Ensemble N = 152 (152 unique), test-day peaks, ramps, standard deviations, substation excess, savings, transformer peaks, voltages, violation points (Tables III-1, III-2) | `outputs/runs/static_vs_doe_2011-02-05_20260929-111819/summary.csv` and `manifest.json` (identical to the earlier `..._20260927-133641` run) |
| Jesmond peak 4.26 MW at 16:30, zone limit 4.05 MW, 12 stress intervals, 693 kWh excess, DOE minimum 101 kW | same `manifest.json`, keys `jesmond` and `envelope` |
| Fleet energy split (958 / 827 kWh discharge in window, 509 / 204 kWh recharge 22:00–24:00), per-household savings statistics, Jain 0.71 / 0.66, 136 of 152 worse off | computed from `dispatch_static.csv` and `dispatch_doe.csv` in the same folder (`battery_kw`, `daily_savings` columns) |
| Losses 4,655 / 4,220 / 4,654 kWh, under-voltage 338 / 287 / 339, transformer import 67.25 / 66.88 / 67.25 MWh, max flow gap 309 kW (Table III-3) | `outputs/runs/battery_location_2011-02-05_20260929-125039/summary.csv` and `manifest.json` |
| 3-year peak 772 kW at 18:30 on 5 Feb 2011, worst 70 % event 16:00–23:30, 7.5 h, 1,040 kWh | `studies/PEAK_DUTY_FINDINGS.md` §3–§5 |
| Full-year violation points: baseline 112,070; QP −24 %; 96 % of added under-voltage in 22:00–24:00; 2 kW import cap −41 % at −11.6 % savings | `studies/NETWORK_AWARE_DISPATCH.md` §2, §4.3, §4.4 |
| GridLAB-D agreement 0.001 % (11 kV), 0.010 % mean / 0.020 % max (LV) | `network/MODEL_VERIFICATION.md`, Level 4 |
| Two-stage on the test-day envelope (Table III-4): soft slices 1,419 / 1,396 / 203 kWh undelivered, 99 / 101 / 47 kW residual, savings $232.2 / $234.1 / $210.2; hard slices 74 / 74 / 51 of 152 infeasible, gap 185 / 185 / 117 % | `outputs/figures/vpp/two_stage_doe_allocation/jesmond_2011-02-05_soft/rule_comparison.csv` and `..._hard/rule_comparison.csv` (run with `--envelope-from` the static_vs_doe run; `vpp/two_stage_doe_allocation/README.md` explains the columns) |
| Replication figures $364.95 / $87.02 per year, 9–11 s per customer-year | initial paper, Section III |
| Season sweep (Table III-5): 105 days, per-regime peaks, ramps, excess, undelivered energy, savings, under-voltage points, losses; 104 of 105 herd days, 0 DOE shortfall days, two-stage above-peak days 52 / 18 | `outputs/runs/doe_sweep_2011-01-14_2011-04-30_20260929-173422/sweep_summary.csv` (scope `all`) and `sweep_results.csv` (one row per day and case); produced by `studies/doe_day_sweep.py` with defaults |
