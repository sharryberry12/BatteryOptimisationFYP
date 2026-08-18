# Network-aware dispatch on Elermore Vale — can the batteries be made to enforce zero voltage violations?

*Working note, 2026-08-16. Answers the question raised after the 2026-08-13
sweeps: "how can I enforce 0 voltage violations? or do some violations occur
in real life?" Every number below comes from the model as fixed on
2026-08-16 (MODEL_VERIFICATION.md defects #2 and #6 closed; DOE constraint
actually enforced — see §4.1). Numbers from earlier runs are not comparable.*

## 0. TL;DR

- **Violations do occur in real life** and the model exaggerates them (§6).
  AS 60038 gives the DNSP a +10 % / −6 % window at the point of supply; Ausgrid
  manages it with off-load distribution-transformer tap boosts, dynamic
  export limits (DOEs) and the mandatory AS/NZS 4777.2:2020 inverter volt-var /
  volt-watt response — none of which the model has. Zero violations is a
  *design target for the batteries* worth quantifying, not a physical
  expectation for this feeder.
- **The R15 QP moves the problem, it does not remove it.** Over the year the
  QP dispatch cuts violation-points by 26–29 % (fit / net) — almost entirely
  over-voltage at midday — but the synchronised charging it induces raises
  under-voltage: mean daily V min falls 0.883 → 0.813 pu, mean daily peak
  substation power rises 2.4 → 3.7 MW, annual losses ×4.3 (§2).
- **The zone-substation OLTC cannot help.** With the regulator active the
  11 kV bus sits at 0.9955–1.0004 pu all day, inside its ±1 % band; the tap
  never moves and every result is identical to controls-off. The violations
  are LV-side, downstream of the distribution transformers (§3).
- **Export-cap DOEs on the QP are almost inert here** (§4.3): the tightest
  envelope removes 3.6 % of the QP's remaining over-voltage and none of its
  under-voltage, because the plain QP already charges into the PV peak
  (feeder head at noon: +59 kW import vs −900 kW export in the baseline).
  Note that the DOE constraint in `osqp_daily_with_DOE.py` had never
  actually been enforced before today (§4.1) — the DOE numbers here are the
  first real ones.
- **What is left is two specific things, not a diffuse problem.** 91 % of
  the QP's residual over-voltage points sit on the LV feeder of one
  transformer (`HP00007159`, +2.56 % boost tap in the GLM) that is above
  1.10 pu whenever it is lightly loaded — a DNSP tap-setting matter no
  dispatch can fix; and 82 % of the QP's under-voltage points fall in
  22:00–24:00, when every battery starts charging at 5 kW as the off-peak
  tariff begins (feeder-head peak 4.1 → 5.9 MW).
- **What zero violations would take** is therefore an *import-side*
  constraint or a de-synchronised charging window for the batteries (plus
  PV curtailment for the few days a cap is infeasible), and a tap change on
  HP00007159 on the network side; §5 lists the options in the order I would
  try them and what each needs from the code that already exists.

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

`python elermorevale_openDSS.py --profiles profiles/{fit,net}_profiles.csv --full --save`

| Metric (per day unless noted) | Baseline | QP (fit) | QP (net) |
|---|---|---|---|
| Violation-points, annual | 132,170 | 98,038 (−25.8 %) | 93,610 (−29.2 %) |
| Violation-points, daily mean / max | 362 / 1,023 | 269 / 456 | 257 / 431 |
| Days QP < baseline | — | 223 / 365 | 226 / 365 |
| Mean daily V min (pu) | 0.883 | 0.813 | 0.816 |
| Worst V min in the year (pu) | 0.705 | 0.622 | 0.623 |
| Mean daily V max (pu) | 1.150 | 1.131 | 1.127 |
| Mean daily peak substation P (kW) | 2,440 | 3,770 | 3,732 |
| Worst peak substation P (kW) | 5,653 | 7,584 | 7,506 |
| Mean daily losses (kW, day-sum of interval losses) | 71.9 | 312.1 | 308.4 |

The over/under split (added 2026-08-16) shows the mechanism on the winter
representative day (2010-07-01, net profiles): baseline 141 over / 126 under
→ QP 60 over / **182 under**. Every battery starts charging at 5 kW the
moment the off-peak tariff begins (22:00), and 1,785 copies of the same
decision put a new 5.9 MW peak on a feeder whose baseline peak was 4.1 MW.
This is the classic tariff-herding effect; it is not an artefact of the
translation, it is what a price-only objective does at scale.

## 3. Lever 1 — the zone-substation OLTC (negative result)

`--oltc` builds the GLM regulator as an OpenDSS RegControl (vreg 120 V on a
52.92 PT ratio = 1.0 pu at the 11 kV winding, ±1 % band, 16 raise / 10 lower
taps of 1.25 %) and runs the daily solves with `controlmode=static`. The
mechanism is verified: in the 3 kW-placeholder snapshot the tap moves to
1.025 (`tests/test_zone_regcontrol_*`).

With real profiles it never moves. On both representative days the 11 kV
bus stays between **0.9955 and 1.0004 pu** for all 48 intervals — the 132 kV
source is stiff and the zone transformer drop at 4 MW is < 0.5 % — so the
regulator sits at neutral and every metric in `figures/net_oltc/summaries.txt`
is byte-identical to `figures/net/summaries.txt`. The violations live across
the distribution transformers and LV feeders (2–3 % drop each way), out of
the zone regulator's reach. Real DNSPs fix that with off-load tap boosts on
the distribution transformers (typically +2.5 / +5 %) — a static offset the
model leaves at nominal (§6).

## 4. Lever 2 — DOE-constrained dispatch (export caps on the QP)

### 4.1 The constraint had never been enforced

`osqp_daily_with_DOE.py` extends the QP with `doe_min ≤ p_k ≤ doe_max`. Its
persistent OSQP workspace was set up with the 2T+1 base rows and each day
called `solver.update(A=<A with T extra rows>, l=…, u=…)`. `osqp` 1.x
`update(**kwargs)` silently ignores `A`, so the solver kept solving the
unconstrained problem: every scenario produced the same dispatch and the
same annual savings ($364.94) and "98.5 % compliant" only measured how often
the unconstrained dispatch happened to fit. **Any DOE result produced before
2026-08-16 is invalid.** The fix keeps the DOE identity rows in the
workspace from setup (inactive at ±∞) and updates only their bounds per day;
`tests/test_doe_constraints.py` pins that the rows bind, that no-envelope
reproduces `osqp_daily.py`, and that infeasible days fall back cleanly.

### 4.2 Scenarios and what the QP could honour

Envelopes are per household, on the grid flow p (kW, export negative);
`--export-limit` scales the `conservative` scenario.

| Profile set | Export cap | Days envelope feasible | Mean annual savings (fit) |
|---|---|---|---|
| `fit_profiles` (no DOE) | — | — | $364.94 |
| `fit_doe_conservative` | 2.4 kW flat | 99.4 % | $364.26 |
| `fit_doe_conservative_cap1.5` | 1.2 kW flat | 99.0 % | $362.29 |
| `fit_doe_conservative_cap0.75` | 0.6 kW flat | 98.0 % | $361.36 |
| `fit_doe_tight` | 1.5 / 0.9 / 0.45 kW (off-peak / shoulder / 14–20 h) | 98.3 % | $359.33 |

Feasibility stays high even at 0.6 kW because the QP has perfect foresight:
it pre-discharges in the morning so the full 10 kWh can absorb the midday
surplus. On the infeasible days the script falls back to the unconstrained
dispatch and flags the breach (`doe_compliant = 0`, `doe_slack_kw > 0`) —
**PV curtailment is not modelled**, so those days still export above the cap.
The savings cost of a cap is small (< 2 % even for `tight`) because the
constraint mostly re-times charging within the same tariff block.

### 4.3 What the network sees

`python elermorevale_openDSS.py --profiles profiles/fit_doe_<scenario>.csv --full --save --output-dir figures/doe_<scenario>`

Full year, 100 monitors × 48 intervals × 365 days; baseline (no battery)
is the same in every row: **132,170** violation-points = 120,698 over +
11,472 under. `qp_*` columns are the dispatched scenario.

| Dispatch | Violation-points | over (> +10 %) | under (< −6 %) | mean daily V min / worst | mean daily V max | mean daily peak TX (kW) |
|---|---|---|---|---|---|---|
| Baseline (no battery) | 132,170 | 120,698 | 11,472 | 0.883 / 0.705 | 1.150 | 2,440 |
| QP, fit profiles (no DOE) | 98,038 | 71,839 | 26,199 | 0.813 / 0.622 | 1.131 | 3,770 |
| QP, net profiles (no DOE) | 93,610 | 67,808 | 25,802 | 0.816 / 0.623 | 1.127 | 3,732 |
| QP + DOE 2.4 kW export cap | 97,765 | 71,560 | 26,205 | 0.813 / 0.622 | 1.130 | 3,770 |
| QP + DOE 1.2 kW export cap | 96,622 | 70,423 | 26,199 | 0.813 / 0.622 | 1.130 | 3,770 |
| QP + DOE 0.6 kW export cap | 96,042 | 69,850 | 26,192 | 0.813 / 0.622 | 1.130 | 3,769 |
| QP + DOE `tight` (1.5/0.9/0.45 kW) | 95,452 | 69,264 | 26,188 | 0.813 / 0.622 | 1.130 | 3,769 |

Three things the table says:

1. **The export caps are almost inert on this network** — the tightest one
   removes 3.6 % of the QP's over-voltage and 0 % of its under-voltage.
   Not because the constraint doesn't bind (it does; §4.1–4.2) but because
   the plain fit QP already charges into the PV peak: on the summer
   representative day the aggregate PV peak is 203 kW across the 152 profiles
   and the batteries are absorbing 103 kW at that moment, so the feeder-head
   flow at noon is +59 kW *import* (baseline: −900 kW export). Per household
   the QP's worst-interval export averages 0.25 kW; only 17 of 152 customers
   ever export more than 0.6 kW on that day. There is nothing left for a
   per-household export cap to cut.
2. **What the QP leaves behind on the over-voltage side is one transformer.**
   Attributing points on 25 days spread through the year: 91 % of the QP's
   residual over-voltage points sit on the 7 monitors behind
   `HP00007159GTX00000001_TX` (300 kVA, GLM `primary_voltage 10725` — a
   +2.56 % off-load boost tap; 433 V × 11000/10725 = 444 V, 256 V L-N =
   1.068 pu on the 240 V base before any load effect). Those loads sit above
   1.10 pu in every light-load interval (07:00–21:30 on the summer day: 210
   of the QP's 252 over-points that day) whatever the dispatch does. In the
   baseline the same feeder is 31 % of the over-points; PV export elsewhere
   is the rest, and that is the part the QP removes. **This is a tap-setting
   question for the DNSP, not a battery-scheduling one** — and it is a
   modelling one for the thesis: on the 240 V base the AS 60038 window is
   generous (1.10 = 264 V; the standard's nominal is 230 V), the 11 kV bus is
   held at exactly 1.0 pu, and the boost tap the source encodes was set for a
   feeder without 155 PV systems.
3. **What the QP adds on the under-voltage side is the 22:00 charging block.**
   82 % of the QP's under-voltage points fall in 22:00–24:00 (baseline:
   13 %); on the winter day 40 of the 100 monitors are below 0.94 pu at
   22:00 (V min 0.737) as every battery starts charging at 5 kW the moment the
   off-peak tariff begins, lifting the feeder-head peak from 4.08 MW
   (18:00, baseline) to 5.93 MW (22:00). Export caps cannot see this; it needs
   an *import*-side constraint or de-synchronisation (§5, items 1 and 3).

So the honest answer to "how do I enforce zero violations": on this model
you cannot get there from the export side, and the plain QP already does the
export side's job. The remaining ~96 k points/yr split into a static
tap-boost feeder (DNSP-side, ~2/3) and synchronised charging (dispatch-side,
~1/4) — the second is exactly what a network-aware objective or an
import envelope should target, and it is where the next experiment belongs.

## 5. What "zero violations" would actually take (decisions for the author)

Ordered by how much of the existing code each reuses. Items 1–2 are the
ones I would do first; they use `osqp_daily_with_DOE.py`'s (now working)
envelope rows unchanged.

1. **Import-side envelope (`doe_max` finite).** The under-voltage half is
   caused by charging, not exporting. A per-household import cap in the
   22:00 block (or a per-interval cap derived from the feeder's residual
   headroom) forces the QP to spread charging over the off-peak window
   instead of front-loading it. `generate_doe_envelope()` already returns
   `doe_max`; only a scenario branch is needed. Expect infeasibility on
   high-load winter evenings — hence item 2.
2. **PV curtailment as a decision variable.** Add `c_k ≥ 0` (curtailed PV,
   with `c_k ≤ g_k`) to the QP with a small linear penalty, so `p = l − g +
   c − b` and a hard export cap is always feasible. This is what an
   AS/NZS 4777.2 inverter does under a DOE in reality; without it the
   "tight" scenarios keep exporting on the days that matter most.
3. **De-synchronise the tariff response.** Cheapest fix for the 22:00 peak:
   randomised or staggered charge-start (a per-household offset in the
   heuristic weights) — no network model needed, and it is what retailers'
   VPP platforms do. Quantify against item 1.
4. **Volt-var / volt-watt at the inverter (network side).** OpenDSS `InvControl`
   on the PV/battery elements would give the model the response the standard
   mandates; it changes the *baseline* too, so the comparison stays fair.
5. **Network-aware objective (voltage sensitivities).** Feed OpenDSS
   per-bus dV/dP into the QP (or into the DOE allocation of
   `vpp/two_stage_doe_allocation`) and iterate. Highest fidelity, largest
   change; only worth it after 1–3 have shown what a per-household envelope
   cannot reach.
6. **Network side: the HP00007159 tap.** Setting that transformer's
   `primary_voltage` to 11000 (neutral tap) in the model — one number in
   `TransformerConfigs.glm`, or an override in `build_elermorevale` — would
   remove ~90 % of the QP's residual over-voltage at a stroke and is what a
   DNSP would do first on a feeder that has grown 155 PV systems. Worth
   running as a sensitivity so the thesis can separate "what the batteries
   can do" from "what the tap does".

The attribution numbers (91 % / 82 %) come from
`diag_violation_attribution.py`, which re-runs baseline and QP on every
15th day and splits the violation-points by feeder and by hour.

## 6. Why the model over-states violations (read before quoting numbers)

- 152 clean Ausgrid customers are cycled over 1,785 loads (each profile ~12
  times), so load coincidence is that of 152 households, not 1,785, and
  every battery makes the same decision at the same minute.
- Distribution-transformer taps are at nominal; Ausgrid boosts them.
- LV reactances are estimates (0.25 / 0.08 Ω/km); the balanced-line
  reduction discards mutual coupling (MODEL_VERIFICATION.md "Known
  approximations"). Level 4 puts the LV agreement with GridLAB-D at ~1 %.
- No inverter volt-var / volt-watt, no curtailment, no OLTC on the
  distribution transformers.
- Only 100 of 1,785 loads are monitored; the count scales the *fraction*,
  and V min / V max are the extremes over that subset.

## 7. Reproduce

```bash
python -m pytest                                        # 101 tests incl. tests/test_doe_constraints.py
python osqp_daily_with_DOE.py --scenarios conservative tight --no-compare
python osqp_daily_with_DOE.py --scenarios conservative --export-limit 1.5 0.75 --no-compare
python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv --full --save
python elermorevale_openDSS.py --profiles profiles/net_profiles.csv --full --save --output-dir figures/net
python elermorevale_openDSS.py --profiles profiles/net_profiles.csv --save --output-dir figures/net --oltc
for s in conservative tight conservative_cap1.5 conservative_cap0.75; do
  python elermorevale_openDSS.py --profiles profiles/fit_doe_$s.csv --save --output-dir figures/doe_$s
  python elermorevale_openDSS.py --profiles profiles/fit_doe_$s.csv --save --output-dir figures/doe_$s --full
done
```

Outputs: `figures/<dir>/summaries.txt` (representative days, over/under
split, OLTC state in every block header), `figures/<dir>/opendss_sweep_results.csv`
(one row per day: `oltc, base_/qp_ v_min, v_max, violations, over, under,
peak_tx_kw, loss_kw`), `sweep_summary.png`.
