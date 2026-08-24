# Network-aware dispatch on Elermore Vale — can the batteries be made to enforce zero voltage violations?

> **Model epoch (2026-08-24):** the OpenDSS translation was reworked —
> GridLAB-D-faithful Carson line impedances, the exact z-matrix sequence
> reduction, the zone-transformer impedance base and an ideal source
> (network/MODEL_VERIFICATION.md, Level 4 agreement now ≤0.02 %) — and
> **every sweep, representative day and attribution in this note was
> regenerated on the reworked model the same day** (the pre-rework
> outputs are archived as `outputs/figures/*.stale-20260824`). The old
> model overstated line impedance (backbone ~2×), so its numbers — e.g.
> a 132,170-point baseline, QP V min 0.813, the export caps clipping up
> to 10 % of residual over-voltage — are pessimistic and must not be
> mixed with anything below.

*Working note, 2026-08-16; §4 extended 2026-08-19 with the import-side
envelope and PV-curtailment experiment; all network numbers regenerated
2026-08-24 on the impedance-reworked model (see the epoch note above).
Answers the question raised after the 2026-08-13 sweeps: "how can I
enforce 0 voltage violations? or do some violations occur in real life?"
Every number below comes from the model as of 2026-08-24
(network/line_impedance.py; Level 4 agreement with GridLAB-D ≤0.02 %)
and the DOE scheduler as of 2026-08-19 (envelope actually enforced, PV
curtailment and import shortfall as decision variables — see §4.1).
Numbers from earlier runs are not comparable.*

## 0. TL;DR

- **Violations do occur in real life** and the model exaggerates them (§6).
  AS 60038 gives the DNSP a +10 % / −6 % window at the point of supply; Ausgrid
  manages it with off-load distribution-transformer tap boosts, dynamic
  export limits (DOEs) and the mandatory AS/NZS 4777.2:2020 inverter volt-var /
  volt-watt response — none of which the model has. Zero violations is a
  *design target for the batteries* worth quantifying, not a physical
  expectation for this feeder.
- **The R15 QP moves the problem, it does not remove it.** Over the year the
  QP dispatch cuts violation-points by 24–27 % (fit / net) — almost entirely
  over-voltage at midday — but the synchronised charging it induces raises
  under-voltage (4,801 → 15,128 points): mean daily V min falls 0.923 →
  0.847 pu, mean daily peak substation power rises 2.4 → 3.7 MW, annual
  losses ×4.4 (§2).
- **The zone-substation OLTC cannot help.** With the regulator active the
  11 kV bus sits at 0.9979–1.0002 pu all day, inside its ±1 % band; the tap
  never moves and every result is identical to controls-off. The violations
  are LV-side, downstream of the distribution transformers (§3).
- **Export-cap DOEs on the QP act only at the margin here** (§4.2–4.3): with
  PV curtailment holding the cap on every day, the tightest envelopes
  (0.6 kW flat, `tight`) remove ~6.5 % of the QP's remaining over-voltage —
  the inverter clipping 2.6–2.9 % of the year's PV at a handful of large
  systems, at a cost that falls almost entirely on them (median household
  ±$0.03/yr, worst −$2,600–2,800; fleet mean −10 %) — and none of its
  under-voltage, because the plain QP already charges into the PV peak
  (feeder head at noon: +59 kW import vs −900 kW export in the baseline).
  Note that the DOE constraint in `osqp_daily_with_DOE.py` had never
  actually been enforced before 2026-08-16 (§4.1) — the DOE numbers here
  are the first real ones.
- **What the plain QP leaves is two specific things, not a diffuse
  problem.** 95 % of its residual over-voltage points sit on the LV feeder
  of one transformer (`HP00007159`, +2.56 % boost tap in the GLM) that is
  above 1.10 pu whenever it is lightly loaded — a DNSP tap-setting matter no
  dispatch can fix; and 96 % of its under-voltage points fall in
  22:00–24:00, when every battery starts charging at 5 kW as the off-peak
  tariff begins (feeder-head peak 4.0 → 5.9 MW).
- **An import-side envelope is the lever that works** (§4.4). A flat
  2 kW/household import cap on the same QP takes the year to **66,422
  violation-points, 41 % below baseline** (plain QP: 24 % below): under-voltage
  15,128 → 6,078 (baseline 4,801), mean daily V min 0.847 → 0.910
  (baseline 0.923), mean daily peak substation power 3.75 → 2.81 MW
  (baseline 2.42), annual losses ×2.4 instead of ×4.4 — and it also trims
  over-voltage (69,799 → 60,344) because more of the charging moves under
  the PV peak. Cost: 12 % of the savings ($365 → $323/yr) and 38 MWh/yr of
  import the battery cannot bring under the cap on the coldest / hottest
  13 % of customer-days (a flat cap is the wrong shape for those days; the
  shortfall variable measures exactly that).
- **What is still left** after the import cap is the HP00007159 feeder on
  the over side (still 94 % of the residual over-points) and, on the under
  side, the fact that every household still steps to *its* cap at 22:00
  (89 % of the residual under-points) plus a thin all-night band while the
  batteries recharge at the cap (9 %); the evening-peak (17:00–19:30)
  under-voltage the baseline has is gone entirely. §5 lists what to try
  next — a time-shaped / feeder-derived import envelope, staggered charge
  start, and the HP00007159 tap sensitivity — and what each needs from the
  code.

## 1. The question, and what "violation" means here

A *violation-point* is one monitored load × one half-hour whose per-unit
voltage (base 240 V) is above 1.10 or below 0.94. 100 monitored loads
(every 18th of the 1,785 residential loads) × 48 intervals = 4,800 points
per day; the annual sweep sums 365 days. It counts breadth × duration, not
severity, so it is paired below with V min / V max and peak substation power.

Two dispatch scenarios are compared each day: **baseline** (grid = load − PV,
no battery) and **QP** (grid = load − PV − b with b from `osqp_daily.py`,
10 kWh / 5 kW battery, R15 heuristic weights, daily SOC neutrality).

## 2. Ground truth after the model fixes (full year, 2010-07 → 2011-06)

`python network/elermorevale_openDSS.py --profiles outputs/profiles/{fit,net}_profiles.csv --full --save`

| Metric (per day unless noted) | Baseline | QP (fit) | QP (net) |
|---|---|---|---|
| Violation-points, annual | 112,070 | 84,927 (−24.2 %) | 81,538 (−27.2 %) |
| … of which over / under | 107,269 / 4,801 | 69,799 / 15,128 | 66,668 / 14,870 |
| Violation-points, daily mean / max | 307 / 906 | 233 / 403 | 223 / 324 |
| Days QP < baseline | — | 215 / 365 | 216 / 365 |
| Mean daily V min (pu) | 0.923 | 0.847 | 0.849 |
| Worst V min in the year (pu) | 0.742 | 0.648 | 0.650 |
| Mean daily V max (pu) | 1.145 | 1.127 | 1.123 |
| Mean daily peak substation P (kW) | 2,424 | 3,746 | 3,708 |
| Worst peak substation P (kW) | 5,639 | 7,661 | 7,578 |
| Mean daily losses (kW, day-sum of interval losses) | 61.9 | 272.1 | 268.9 |

Relative to the pre-rework model the baseline lost 15 % of its
violation-points (132,170 → 112,070), almost all on the under side
(11,472 → 4,801): halving the backbone impedance mostly relieves
voltage *drop*, so the over-voltage share is now 96 % of the baseline
problem. The QP's relative effect is unchanged (−24 to −27 %), and it
still trades over-voltage for under-voltage: 4,801 → 15,128 points.

The over/under split shows the mechanism on the winter representative
day (2010-07-01, net profiles): baseline 118 over / 61 under → QP 66
over / **110 under**. Every battery starts charging at 5 kW the moment
the off-peak tariff begins (22:00), and 1,785 copies of the same
decision put a new 5.9 MW peak on a feeder whose baseline peak was 4.0 MW.
This is the classic tariff-herding effect; it is not an artefact of the
translation, it is what a price-only objective does at scale.

## 3. Lever 1 — the zone-substation OLTC (negative result)

`--oltc` builds the GLM regulator as an OpenDSS RegControl (vreg 120 V on a
52.92 PT ratio = 1.0 pu at the 11 kV winding, ±1 % band, 16 raise / 10 lower
taps of 1.25 %) and runs the daily solves with `controlmode=static`. The
mechanism is verified: in the 3 kW-placeholder snapshot the tap moves to
1.025 (`tests/test_zone_regcontrol_*`).

With real profiles it never moves. On both representative days the 11 kV
bus stays between **0.9979 and 1.0002 pu** for all 48 intervals — the 132 kV
source is ideal and the zone transformer drop even at the QP's 5.9 MW peak
is ~0.2 % (the 2026-08-24 rework put the zone TX on GridLAB-D's impedance
base, a third of what the model used before) — so the
regulator sits at neutral and every metric in `outputs/figures/net_oltc/summaries.txt`
is byte-identical to `outputs/figures/net/summaries.txt`. The violations live across
the distribution transformers and LV feeders (2–3 % drop each way), out of
the zone regulator's reach. Real DNSPs fix that with off-load tap boosts on
the distribution transformers (typically +2.5 / +5 %) — a static offset the
model leaves at nominal (§6).

## 4. Lever 2 — DOE-constrained dispatch (per-household envelopes on the QP)

### 4.1 The constraint had never been enforced — and what the envelope means now

`osqp_daily_with_DOE.py` extends the QP with a per-household envelope
`doe_min ≤ p_k ≤ doe_max` on the grid flow. Its persistent OSQP workspace
was set up with the 2T+1 base rows and each day called
`solver.update(A=<A with T extra rows>, l=…, u=…)`. `osqp` 1.x
`update(**kwargs)` silently ignores `A`, so the solver kept solving the
unconstrained problem: every scenario produced the same dispatch and the
same annual savings ($364.94) and "98.5 % compliant" only measured how often
the unconstrained dispatch happened to fit. **Any DOE result produced before
2026-08-16 is invalid.** The 2026-08-16 fix kept the envelope rows in the
workspace from setup (inactive at ±∞) and updated only their bounds per day,
falling back to the unconstrained dispatch on infeasible days.

On 2026-08-19 the scheduler was extended so that the envelope is always
honoured and the fallback is gone. It now solves over `x = [b | c | s]`
(3T = 144 variables):

- `c_k ∈ [0, g_k]` — **curtailed PV** (the AS/NZS 4777.2 volt-watt / DOE
  response an inverter performs); the grid flow becomes `p = l − g + c − b`.
- `s_k ≥ 0` — **import shortfall**: load cannot be shed, so the import cap
  is soft, `p_k ≤ doe_max_k + s_k`.
- The export cap `p_k ≥ doe_min_k` is **hard** — curtailment always makes it
  feasible.
- Objective `Σ h_k p_k² + 100·Σ h_k (c_k + s_k) + 10⁻³ Σ h_k (c_k² + s_k²)`:
  the h-scaled linear penalty (100 $/kW-equivalent) dominates any bill
  saving, so relief is used only when the battery physically cannot meet the
  envelope; the small quadratic term keeps the problem strictly convex (the
  zero-export corner otherwise hits OSQP's iteration cap). The bill is
  computed on `g − c`. With no envelope the dispatch equals `osqp_daily.py`
  to solver tolerance.

Per customer-day the profiles carry `doe_compliant` (max `s` ≤ 10⁻⁴ kW),
`curtail_kw` and `import_shortfall_kw`. `tests/test_doe_constraints.py`
(13 tests) pins that the rows bind, that no-envelope reproduces
`osqp_daily.py`, that a zero export cap is met by curtailment, that a
1 kW import cap is met by discharge and a 0 kW cap reports the shortfall,
and that the reported quantities reconcile with `p`.

### 4.2 Export-cap scenarios and what they cost

Envelopes are per household, on the grid flow p (kW, export negative);
`--export-limit` scales the `conservative` scenario. The four profile sets
were regenerated 2026-08-19 under the curtailment formulation (§4.1), so the
cap holds on every customer-day by construction; what is left to report is
how much PV the inverter has to clip to hold it, who pays, and what the
network sees (§4.3). The 152 customers generate 394.5 MWh of PV over the
year.

| Profile set | Export cap | Curtailed PV | Customer-days curtailing | Peak clip | Mean / median annual savings (fit) |
|---|---|---|---|---|---|
| `fit_profiles` (no DOE) | — | — | — | — | $364.94 / $372.38 |
| `fit_doe_conservative` | 2.4 kW flat | 3.63 MWh/yr (0.92 %) | 351 (0.6 %) | 5.2 kW | $354.37 / $364.86 |
| `fit_doe_conservative_cap1.5` | 1.2 kW flat | 7.35 MWh/yr (1.86 %) | 569 (1.0 %) | 6.9 kW | $341.26 / $358.13 |
| `fit_doe_conservative_cap0.75` | 0.6 kW flat | 11.29 MWh/yr (2.86 %) | 1,084 (2.0 %) | 7.5 kW | $326.99 / $350.93 |
| `fit_doe_tight` | 1.5 / 0.9 / 0.45 kW (off-peak / shoulder / 14–20 h) | 10.11 MWh/yr (2.56 %) | 943 (1.7 %) | 7.5 kW | $327.88 / $350.86 |

Curtailment stays small because the QP has perfect foresight: it
pre-discharges in the morning so the full 10 kWh can absorb the midday
surplus, and the inverter clips only what is left over on the days the
surplus exceeds the battery. **The cost is not spread — it lands on the
large-PV households.** At 2.4 kW the median customer's annual savings change
by $0.01; 99 % of the fleet's $10.57/customer mean loss is two customers
(157: 39 kWh/day of PV against 21 kWh of load, 2,637 kWh clipped, −$1,092;
75: 23 kWh/day, 996 kWh, −$450), who between them are essentially all of the
3.63 MWh curtailed. Under `tight` the mean loss is $37.06/customer
(−10.2 %) but the median customer still changes by $0.03; 23 customers
lose more than $20, eight more than $100, and the worst five carry 81 % of
the fleet's loss (157: 6,052 kWh clipped, −$2,579; 75: 2,794 kWh,
−$1,230). Under topology 1 every PV kWh is metered and paid the $0.40/kWh
gross feed-in tariff, so a clipped kWh is $0.40 of revenue lost (2,637 kWh
× 0.40 = $1,055 of customer 157's $1,092); the remainder is charging
re-timed out of the off-peak block to sit under the PV peak — ~$10–30/yr
for the handful of mid-sized systems that never curtail at 2.4 kW, up to
~$150–190/yr for the same households under the tighter caps (customer 273
under `tight`: −$258, of which $73 is clipped FiT).

For comparison, the 2026-08-16 fallback formulation reported these sets as
$364.26 / $362.29 / $361.36 / $359.33 with 0.6–2 % of customer-days
"infeasible" and exporting above the cap; those are the days that now
curtail, and the difference is what the fallback was hiding.
`doe_compliant` is 0 on 11 of the 221,920 customer-days across the four
sets (0 / 2 / 8 / 1), all where OSQP stopped at its 40,000-iteration cap
with an inexact solution; the post-hoc breach (`doe_slack_kw`) is
0.002 kW at the median and 0.088 kW at worst (one day of the 0.6 kW set) —
solver tolerance, not an envelope the battery could not meet.

### 4.3 What the network sees

`python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_doe_<scenario>.csv --full --save --output-dir outputs/figures/doe_<scenario>`

Full year, 100 monitors × 48 intervals × 365 days; baseline (no battery)
is the same in every row: **112,070** violation-points = 107,269 over +
4,801 under. `qp_*` columns are the dispatched scenario.

| Dispatch | Violation-points | over (> +10 %) | under (< −6 %) | mean daily V min / worst | mean daily V max / worst | mean daily peak TX (kW) |
|---|---|---|---|---|---|---|
| Baseline (no battery) | 112,070 | 107,269 | 4,801 | 0.923 / 0.742 | 1.145 / 1.180 | 2,424 |
| QP, fit profiles (no DOE) | 84,927 (−24 %) | 69,799 | 15,128 | 0.847 / 0.648 | 1.127 / 1.147 | 3,746 |
| QP, net profiles (no DOE) | 81,538 (−27 %) | 66,668 | 14,870 | 0.849 / 0.650 | 1.123 / 1.144 | 3,708 |
| QP + DOE 2.4 kW export cap | 83,756 (−25 %) | 68,611 | 15,145 | 0.847 / 0.648 | 1.123 / 1.142 | 3,747 |
| QP + DOE 1.2 kW export cap | 81,868 (−27 %) | 66,725 | 15,143 | 0.847 / 0.648 | 1.120 / 1.135 | 3,747 |
| QP + DOE 0.6 kW export cap | 80,357 (−28 %) | 65,235 | 15,122 | 0.847 / 0.649 | 1.118 / 1.127 | 3,743 |
| QP + DOE `tight` (1.5/0.9/0.45 kW) | 80,366 (−28 %) | 65,245 | 15,121 | 0.847 / 0.649 | 1.118 / 1.127 | 3,743 |

Three things the table says:

1. **The export caps act only on the over-voltage side, and only at the
   margin.** With curtailment holding the cap on every day, the
   2.4 / 1.2 / 0.6 kW caps remove 1.7 / 4.4 / 6.5 % of the QP's
   over-voltage points and `tight` 6.5 %, and each pulls the worst-day
   V max down (1.147 → 1.142 / 1.135 / 1.127 / 1.127 pu) — that is the
   inverter clipping the handful of large-PV households (§4.2) whose export
   the battery cannot absorb. (On the pre-2026-08-24 model the same caps
   measured 3.3–10.4 %: the old overstated impedance made voltage twice as
   sensitive to the clipped export.) Under-voltage does not move (15,128 →
   15,121–15,145 across the four sets) and neither do V min, peak
   substation power or losses.
   The caps have little to work on because the plain fit QP already charges
   into the PV peak: on the summer representative day the aggregate PV peak
   is 203 kW across the 152 profiles and the batteries are absorbing 103 kW
   at that moment, so the feeder-head flow at noon is +59 kW *import*
   (baseline: −900 kW export). Per household the QP's worst-interval export
   averages 0.25 kW; only 17 of 152 customers ever export more than 0.6 kW
   on that day.
2. **What the QP leaves behind on the over-voltage side is one transformer.**
   Attributing points on 25 days spread through the year: 95 % of the QP's
   residual over-voltage points sit on the 7 monitors behind
   `HP00007159GTX00000001_TX` (300 kVA, GLM `primary_voltage 10725` — a
   +2.56 % off-load boost tap; 433 V × 11000/10725 = 444 V, 256 V L-N =
   1.068 pu on the 240 V base before any load effect). Those loads sit above
   1.10 pu through every light-load interval (07:00–21:30) whatever the
   dispatch does. In the
   baseline the same feeder is 36 % of the over-points; PV export elsewhere
   is the rest, and that is the part the QP removes. **This is a tap-setting
   question for the DNSP, not a battery-scheduling one** — and it is a
   modelling one for the thesis: on the 240 V base the AS 60038 window is
   generous (1.10 = 264 V; the standard's nominal is 230 V), the 11 kV bus is
   held at exactly 1.0 pu, and the boost tap the source encodes was set for a
   feeder without 155 PV systems.
3. **What the QP adds on the under-voltage side is the 22:00 charging block.**
   96 % of the QP's under-voltage points fall in 22:00–24:00 (baseline:
   14 %); on the winter day V min drops to 0.754 pu at
   22:00 as every battery starts charging at 5 kW the moment the
   off-peak tariff begins, lifting the feeder-head peak from 4.04 MW
   (18:00, baseline) to 5.89 MW (22:00). Export caps cannot see this; it needs
   an *import*-side constraint or de-synchronisation (§5, items 1 and 3).

So the honest answer to "how do I enforce zero violations": on this model
you cannot get there from the export side. The plain QP already does most
of the export side's job, and the tightest cap buys only another ~5 % of
the residual (84.9 k → 80.4 k points/yr) by clipping a few large-PV
households. What remains is dominated by over-voltage on the boosted
HP00007159 feeder (DNSP-side) plus the QP's synchronised 22:00 charging
(dispatch-side) — the second is exactly what an import envelope targets
(§4.4).

### 4.4 Import-cap scenarios — the lever that works

```bash
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 2 --no-compare   # -> outputs/profiles/fit_doe_none_imp2.csv
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 3 --no-compare   # -> outputs/profiles/fit_doe_none_imp3.csv
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_doe_none_imp2.csv --full --save --output-dir outputs/figures/doe_none_imp2
python network/diagnostics/diag_violation_attribution.py --profiles outputs/profiles/fit_doe_none_imp2.csv --every 15
```

A flat per-household **import** cap `doe_max = 2 kW` (and 3 kW), no export
cap, on the same fit QP. The under-voltage the plain QP creates is import at
22:00, so the cap forces each battery to spread its charging over the whole
off-peak window (and, when that is not enough, under the PV peak) instead of
front-loading it. Load cannot be shed, so the cap is soft: the shortfall
variable `s` records what the battery physically could not bring under the
cap.

**Dispatch side** (152 customers × 365 days):

| Profile set | Import cap | Customer-days meeting the cap | Unmet import | Mean annual savings (fit) |
|---|---|---|---|---|
| `fit_profiles` (no DOE) | — | — | — | $364.94 |
| `fit_doe_none_imp3` | 3 kW flat | 94.4 % | 13.3 MWh/yr (87 kWh/customer) | $350.07 (−4.1 %) |
| `fit_doe_none_imp2` | 2 kW flat | 87.0 % | 38.2 MWh/yr (251 kWh/customer) | $322.59 (−11.6 %) |

The shortfall is concentrated exactly where the cap is the wrong shape: the
non-compliant customer-days average 37.5 kWh of load (2 kW cap; 43.0 kWh at
3 kW) against 19.6 kWh across all days, they cluster in Jun–Aug (46 % of the
7,207 non-compliant days at 2 kW) and Jan–Feb (18 %), and 59 % of the unmet
energy (63 % at 3 kW) falls in 22:00–24:00 with another 11 % (5 %) overnight —
a 5 kW battery cannot pull a 37 kWh household under a flat 2 kW line on a
cold night. No PV is curtailed in either set (no export cap is active).

**Network side** (full year, 100 monitors; baseline identical to §4.3):

| Dispatch | Violation-points | over (> +10 %) | under (< −6 %) | mean daily V min / worst | mean daily V max | mean daily peak TX (kW) | mean daily losses (kW) |
|---|---|---|---|---|---|---|---|
| Baseline (no battery) | 112,070 | 107,269 | 4,801 | 0.923 / 0.742 | 1.145 | 2,424 | 61.9 |
| QP, fit profiles (no DOE) | 84,927 (−24 %) | 69,799 | 15,128 | 0.847 / 0.648 | 1.127 | 3,746 | 272.1 |
| QP + 3 kW import cap | 76,862 (−31 %) | 66,452 | 10,410 | 0.872 / 0.678 | 1.127 | 3,316 | 209.7 |
| QP + 2 kW import cap | **66,422 (−41 %)** | 60,344 | 6,078 | 0.910 / 0.690 | 1.126 | 2,808 | 149.0 |

Four things the tables say:

1. **The import cap fixes what the QP broke.** Under-voltage points fall
   from 15,128 back to 6,078 (baseline 4,801), mean daily V min recovers
   from 0.847 to 0.910 pu (baseline 0.923), the mean daily feeder-head peak
   drops from 3.75 to 2.81 MW (baseline 2.42) and annual losses go from ×4.4
   to ×2.4 of baseline. The 22:00 step is now capped at 2 kW of grid import
   per household instead of load + 5 kW of charging.
2. **It also trims over-voltage** (69,799 → 60,344, −14 %) although it never
   touches export: with the night's charging capped at ~1 kW above load the
   battery cannot fill in the off-peak window, so more of its 10 kWh is
   charged under the midday PV peak, and less PV is exported.
3. **The cost is modest and measurable**: 12 % of the savings ($365 → $323
   per year) at 2 kW, 4 % at 3 kW, plus the unmet import above. Relative to
   the plain QP, the 3 kW cap buys 44 % of the 2 kW cap's violation
   reduction (8,065 of 18,505 points) for 35 % of its savings cost — the
   response is close to linear in the cap over this range, on both model
   epochs.
4. **What is left, from the 25-day attribution** (`diag_violation_attribution.py`
   on `fit_doe_none_imp2`): over-voltage is still 94 % feeder HP00007159
   (3,720 of 3,948 points) — and it runs later into the day
   (14:00–20:00 carries ~1.5× the points per half-hour of the morning)
   because the batteries discharge through the evening peak to hold imports
   under the cap, lifting that already-boosted feeder further. Under-voltage
   (457 points on the 25 days; baseline 301) is 89 % in 22:00–24:00 — every
   household still steps to *its* 2 kW cap at the same minute — 9 % in a
   thin all-night band (00:00–06:30) while the batteries recharge at the cap,
   2 % at 20:00–21:30, and **none between 17:00 and 19:30**, where the baseline
   has 47 % of its under-voltage (141 of 301): the batteries carry the
   evening peak. A flat cap has therefore removed the evening-peak
   under-voltage and halved the 22:00 block; what remains is the
   synchronisation itself, which a flat per-household number cannot address
   (§5, items 1–2).

## 5. What "zero violations" would actually take (decisions for the author)

Two of the original six items are done (2026-08-19): the **import-side
envelope** and **PV curtailment / import shortfall as decision variables**
(§4.1, §4.4). The flat 2 kW cap took the year from −26 % to −43 % of baseline
violation-points; what it leaves is (a) the same-minute step to the cap at
22:00 plus the all-night recharge band, (b) unmet import on the coldest and
hottest 13 % of customer-days, and (c) the HP00007159 boost feeder. The
remaining options, ordered by how much of the existing code each reuses:

1. **Shape the import envelope in time.** A flat number is simultaneously
   too tight for a 37 kWh winter day (38 MWh/yr of shortfall) and not tight
   enough at 22:00 (everyone steps to 2 kW together — 89 % of the residual
   under-points). Two shapes to run, both a scenario branch in
   `generate_doe_envelope()` (`doe_max` is currently flat): a tighter cap in
   22:00–24:00 that relaxes after midnight, and a per-interval cap derived
   from the feeder's residual headroom (feeder-head baseline power against
   its 22:00 peak), which is what a DNSP DOE actually is. Report the same
   three numbers as §4.4: violation-points, savings, unmet MWh.
2. **De-synchronise the tariff response.** Cheapest fix for the 22:00 step:
   randomised or staggered charge-start (a per-household offset in the
   heuristic weights, or a per-household start time on the import cap) — no
   network model needed, and it is what retailers' VPP platforms do.
   Quantify against item 1; the two compose.
3. **Network side: the HP00007159 tap.** Setting that transformer's
   `primary_voltage` to 11000 (neutral tap) in the model — one number in
   `TransformerConfigs.glm`, or an override in `build_elermorevale` — would
   remove ~90 % of the residual over-voltage at a stroke (still 94 % of it
   after the import cap, and now extending into 14:00–20:00 as the batteries
   discharge through the evening) and is what a DNSP would do first on a
   feeder that has grown 155 PV systems. Worth running as a sensitivity so
   the thesis can separate "what the batteries can do" from "what the tap
   does".
4. **Volt-var / volt-watt at the inverter (network side).** OpenDSS `InvControl`
   on the PV/battery elements would give the model the response the standard
   mandates; it changes the *baseline* too, so the comparison stays fair.
5. **Port `c` and `s` into the VPP layer.** `vpp_common.HouseholdSolver`
   still solves in `b` only, so `vpp/two_stage_doe_allocation` zeroes a
   household whose slice is infeasible (its README). The two relief
   variables from `osqp_daily_with_DOE.py` make every per-household DOE
   feasible and turn the two-stage gap into a curtailment / shortfall number
   instead of a failure count.
6. **Network-aware objective (voltage sensitivities).** Feed OpenDSS
   per-bus dV/dP into the QP (or into the DOE allocation) and iterate.
   Highest fidelity, largest change; only worth it after 1–3 have shown what
   a per-household envelope cannot reach.

The attribution numbers (95 % / 96 % for the plain QP; 94 % / 89 % after the
2 kW import cap) come from `network/diagnostics/diag_violation_attribution.py`,
which re-runs baseline and the dispatched profile set on every 15th day and
splits the violation-points by feeder and by hour.

## 6. Why the model over-states violations (read before quoting numbers)

- 152 clean Ausgrid customers are cycled over 1,785 loads (each profile ~12
  times), so load coincidence is that of 152 households, not 1,785, and
  every battery makes the same decision at the same minute.
- Distribution-transformer taps are at nominal; Ausgrid boosts them.
- Line impedances mirror GridLAB-D's Carson/concentric-neutral model
  since 2026-08-24 (network/line_impedance.py); Level 4 puts the
  agreement with GridLAB-D at ≤0.02 % — but see the model-epoch note at
  the top: the numbers in this document predate that rework.
- No inverter volt-var / volt-watt on the network side and no OLTC on the
  distribution transformers. PV curtailment exists only as the DOE
  scheduler's `c` variable, i.e. only when an export cap binds in the
  dispatch (zero in the plain and import-cap profile sets); the network
  model never curtails on its own.
- Only 100 of 1,785 loads are monitored; the count scales the *fraction*,
  and V min / V max are the extremes over that subset.

## 7. Reproduce

```bash
python -m pytest                                        # 121 tests incl. tests/test_doe_constraints.py (13)
python dispatch/osqp_daily_with_DOE.py --scenarios conservative tight --no-compare                  # export caps (§4.2)
python dispatch/osqp_daily_with_DOE.py --scenarios conservative --export-limit 1.5 0.75 --no-compare
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 2 --no-compare               # import caps (§4.4)
python dispatch/osqp_daily_with_DOE.py --scenarios none --import-limit 3 --no-compare
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --full --save
python network/elermorevale_openDSS.py --profiles outputs/profiles/net_profiles.csv --full --save --output-dir outputs/figures/net
python network/elermorevale_openDSS.py --profiles outputs/profiles/net_profiles.csv --save --output-dir outputs/figures/net --oltc
for s in conservative tight conservative_cap1.5 conservative_cap0.75 none_imp2 none_imp3; do
  python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_doe_$s.csv --save --output-dir outputs/figures/doe_$s          # representative days -> summaries.txt
  python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_doe_$s.csv --save --output-dir outputs/figures/doe_$s --full   # full year -> opendss_sweep_results.csv
done
python network/diagnostics/diag_violation_attribution.py --every 15                                    # plain QP attribution
python network/diagnostics/diag_violation_attribution.py --profiles outputs/profiles/fit_doe_none_imp2.csv --every 15
```

Roughly 20–25 min per DOE profile set (152 customers × 365 days ×
144-variable QP, multiprocessing) and 8–15 min per full-year network sweep
on the development machine. Two things to know when re-running a single
set: `--export-limit` with one value writes `<scenario>.csv` with no `_cap`
suffix (so `--scenarios conservative --export-limit 0.75` on its own would
overwrite `fit_doe_conservative.csv` — call `run_all()` / `save_profiles()`
with an explicit label instead), and each 16-worker pool needs a few GB of
headroom to spawn — a run started while another process held the memory
sat in a silent worker respawn loop for an hour (2026-08-19).

Outputs: `outputs/figures/<dir>/summaries.txt` (representative days, over/under
split, OLTC state in every block header), `outputs/figures/<dir>/opendss_sweep_results.csv`
(one row per day: `oltc, base_/qp_ v_min, v_max, violations, over, under,
peak_tx_kw, loss_kw`), `sweep_summary.png`.
