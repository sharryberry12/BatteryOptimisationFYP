# Peak-Finding & VPP Duty-Cycle Analysis — Findings

Context and results for the Part B reframe: the VPP as a *virtual peaker
plant* whose value is firm capacity, not bill savings. Produced by
[peak_duty_analysis.py](peak_duty_analysis.py) (dataset-wide duty cycle) and
[replay_peak_event.py](replay_peak_event.py) (worst-event replay on the
Elermore Vale OpenDSS model). Figures in `figures/peak_duty/` and
`runs/peak_replay_*/figures/`.

## 1. Why this analysis exists

The supervisor notes reframe the project: the point of a VPP is to act like
a traditional power plant that turns on to offset grid stress. Financial
incentives (the Part A QP savings) are the *recruitment mechanism* that gets
households into the fleet — the research object is the capability itself.
The operative questions:

1. Find the peak load condition in the Ausgrid dataset.
2. If the network must never exceed some fraction of that peak (say 70 %),
   how many aggregated households does it take to cover the rest?
3. How often does that fleet actually have to run ("VPP mode": batteries
   discharging into the grid instead of load shedding)?

The wager behind the framing: peaks are rare and short, so a modest fleet
buys firm capacity with a tiny duty cycle. The numbers below test that.

## 2. Method

- **Data**: full 3-year Ausgrid solar-home dataset (`data_3_years.csv`,
  2010-07-01 → 2013-06-30, 300 customers, half-hourly). All 300 customers
  are used — the Ratnam cleaning rules (which drop customers for PV
  anomalies) would bias a *demand*-side aggregate; `--clean` exists as a
  sensitivity flag.
- **Aggregate series**: per-interval sum of net demand (GC + CL − GG),
  rescaled to a constant 300-household panel via the per-household mean
  (reporting count never drops below 299, so the rescale is cosmetic).
  Cached under `vpp/cache/peak_agg_data_3_years.csv`.
- **Events**: contiguous half-hour runs above a threshold *f* × peak.
  Gaps in the record (dropped DST days) break contiguity.
- **Fleet sizing** at each threshold, using the repo's battery model
  (5 kW / 10 kWh per household, `osqp_daily.P_MAX` / `E_MAX_DEFAULT`):
  - power-limited count: ⌈max deficit ÷ 5 kW⌉
  - energy-limited count: ⌈worst-event energy ÷ 10 kWh⌉
  - required fleet = max of the two.
- **Peaker assumptions** (deliberate, stated): fleet is pre-charged before
  an event (full 10 kWh usable; `--usable-frac 0.5` models the QP resting
  SOC instead); full recharge between events (violations counted and
  warned); no round-trip losses, consistent with `paper_context.md` §9.

## 3. The peak condition

**Aggregate net demand peaks at 772 kW (2.57 kW/household) on Saturday
2011-02-05 at 18:30** — the February 2011 NSW heatwave. The top-10 peak
days split into two regimes, all peaking 17:30–20:30:

| When | Day | Aggregate kW | kW/household |
|---|---|---|---|
| 2011-02-05 18:30 | Sat | 772.1 | 2.57 |
| 2011-02-01 19:00 | Tue | 744.7 | 2.48 |
| 2013-01-08 19:30 | Tue | 740.1 | 2.47 |
| 2013-01-18 18:30 | Fri | 677.9 | 2.26 |
| 2011-02-03 17:30 | Thu | 657.4 | 2.19 |
| 2010-07-02 17:30 | Fri | 622.5 | 2.08 |
| 2011-02-02 20:30 | Wed | 618.8 | 2.06 |
| 2010-08-02 18:00 | Mon | 583.9 | 1.95 |
| 2011-07-15 17:30 | Fri | 583.5 | 1.95 |
| 2011-07-19 18:00 | Tue | 575.8 | 1.92 |

Two observations that matter downstream:

- **PV is structurally absent at the peak.** Every top event is an evening;
  on the peak day gross load tops 840 kW at 18:00 while PV is already ~0
  (`figures/peak_duty/peak_day.png`). Solar alone cannot touch this peak —
  it must come from storage (or shedding).
- **Both seasons contribute.** Summer heatwave evenings dominate the
  extreme tail, but winter evenings (Jul–Aug) fill the top-10. A
  peaker-VPP is not a summer-only proposition on this feeder.

## 4. Duty cycle vs firm-capacity threshold

Full sweep in `figures/peak_duty/duty_cycle_summary.csv`; selected rows:

| Threshold | kW | Hours/yr above | Events/yr | Worst event | Fleet needed | Binding | Cycles/yr/battery |
|---|---|---|---|---|---|---|---|
| 60 % | 463 | 121.1 | 59.0 | 9.0 h, 1668 kWh | 167 (56 %) | energy | 3.3 |
| 70 % | 540 | 17.3 | 9.3 | 7.5 h, 232 kW, 1040 kWh | **104 (35 %)** | energy | 1.2 |
| 80 % | 618 | 6.3 | 2.3 | 5.5 h, 551 kWh | 56 (19 %) | energy | 0.8 |
| 90 % | 695 | 3.2 | 1.0 | 3.5 h, 179 kWh | 18 (6 %) | energy | 0.7 |
| 95 % | 734 | 1.8 | 1.3 | 2.5 h, 72 kWh | 8 (2.7 %) | power | 0.4 |

Headline findings:

1. **The duty cycle is tiny.** At the 70 % threshold the VPP runs
   17.3 h/year in ~9 events — 0.2 % of the year. The fleet sits idle
   99.8 % of the time, which is exactly the traditional-peaker profile.
   The "15 minutes over a few years" intuition from the meeting notes is
   literally true only near the very top of the curve (95 % threshold:
   1.8 h/yr, 8 households); the 70 % tail is fatter but still negligible.
2. **Sizing is energy-limited, not power-limited, at every threshold below
   95 %.** With 5 kW / 10 kWh batteries, 47 households would cover the
   70 %-threshold peak *power* deficit, but the 7.5-hour worst event needs
   104 households' worth of *energy*. Firm capacity on this feeder is
   bought in kWh, not kW — battery *duration*, not inverter rating, is the
   binding spec for a capacity-style VPP product.
3. **Degradation cost of the capability is negligible.** At 70 %, each
   fleet battery does ~1.2 full-cycle-equivalents/year in peaker duty.
   The capability barely consumes the asset — supporting the argument that
   household financial incentives (daily tariff arbitrage) and the VPP
   capability can coexist on the same battery.
4. **Events cluster hard.** Every required run falls between 16:00 and
   23:30 (`figures/peak_duty/event_calendar.png`), concentrated in
   Feb 2011, Jan 2013, and the mid-2010/2011 winters. FY 2011-12 has
   almost no events — a fleet sized on a mild year would be badly
   undersized, so multi-year (ideally heatwave-inclusive) data is a
   methodological requirement, not a nicety.

Caveats printed by the script: two event pairs at the 70 % threshold sit
closer together than the full-recharge time (per-event sizing is slightly
optimistic there), and the pre-charge assumption (`--usable-frac 1.0`)
presumes day-ahead notice of peak events — realistic for heatwave peaks,
which are forecastable, but worth stating.

## 5. Worst-event replay on the Elermore Vale feeder

`replay_peak_event.py` replays the worst event (2011-02-05 16:00–23:30) as
a physical power flow: the 300 customers' day profiles are mapped round-robin
onto all 1,810 Elermore Vale loads (×6.03 replication), the 104-household
fleet discharges pro-rata (per-household peak 2.23 kW, 10.0 kWh — i.e. each
battery fully drains exactly once), and both scenarios (no battery / VPP)
are solved as 48-step daily power flows. Threshold at feeder scale:
70 % × 772 kW × 6.03 = **3,261 kW** at the zone substation.

| Scenario | Peak feeder-head P | Over threshold | v_min | Voltage-violation points |
|---|---|---|---|---|
| No battery | 5,077 kW | +1,816 kW | 0.727 p.u. | 1,273 of 4,800 |
| VPP peaker fleet | 3,644 kW | +383 kW | 0.781 p.u. | 820 of 4,800 |

**Max measured shave: 1,438 kW** — the evening profile is flat-topped for
the full 7.5 h event (`runs/peak_replay_*/figures/feeder_head_relief.png`),
which is exactly the visual signature of a peaker plant holding a firm limit.

Three physical findings the kW-bookkeeping could not show:

1. **Losses drive a wedge between customer-level and feeder-head capacity.**
   The *injected* VPP aggregate sits exactly on the 3,261 kW threshold, but
   the *measured* feeder head rides ~383 kW (~9 %) above it — network
   losses at this loading (~380 kW at the feeder head during the event).
   The measured shave (1,438 kW) also exceeds the injected shave
   (231.6 kW × 6.03 = 1,397 kW) because reducing flow also avoids some
   losses. Implication: a firm-capacity product defined *at the zone
   substation* must be sized on measured power — roughly a losses-margin
   (~9 % here) more fleet than the customer-aggregate arithmetic suggests —
   or the threshold must be defined at the customer-aggregate level.
2. **The peak relief is also voltage relief.** During the event the
   no-battery feeder sags to 0.727 p.u. at the worst monitored load —
   far below the −6 % statutory limit, the kind of condition that precedes
   load shedding. The fleet lifts the evening minimum by ~0.05 p.u. and
   removes ~450 of the 1,273 violation points. It does not fix the feeder's
   midday overvoltage (up to 1.118 p.u. from replicated PV export) — that
   is the *export*-envelope problem the DOE methods in `vpp/` address, and
   the two are complementary: the same fleet charges midday, discharges at
   the evening peak.
3. **The stress is real but the mapping exaggerates it.** ×6 replication of
   300 all-solar households onto every load makes the simulated feeder
   heavily loaded (evening violations even in the VPP scenario). Absolute
   voltages/kW describe this synthetic loading, not the real Elermore Vale
   operating point; the *deltas* between scenarios are the meaningful
   result.

Run artifacts (dispatch CSVs, `replay_summary.csv`, `manifest.json` with
fleet IDs and sizing provenance) are under `runs/peak_replay_2011-02-05_*/`.

## 6. Limitations & next steps

- **Dataset ≠ feeder.** The 300 Ausgrid customers are all solar homes;
  Elermore Vale's ~1,785 loads are mapped round-robin from them
  (replication factor ≈ 6), so feeder-level kW are "300-panel kW × scale",
  not a measured zone-substation history. The per-household and
  duty-cycle *shape* results transfer; absolute feeder kW do not.
- **Thermal ratings not yet checked.** "Does something trip at peak"
  needs transformer kVA ratings from the GLM compared against simulated
  loading — the natural extension of the replay (the zone substation and
  23 distribution transformers are already monitored).
- **Firm threshold is relative to dataset peak.** A real DNSP threshold
  would come from asset ratings or a connection-agreement limit; the
  70 %-of-peak framing is a placeholder until real DOE / rating data is
  sourced (SA Power Networks Flexible Exports, Project EDGE, Project
  Symphony, DEIP DOE working group).
- **Transmission layer.** The replay stops at the zone substation. The
  planned Kundur two-area coupling would let a transmission contingency
  *trigger* VPP-mode, closing the loop on "VPP as a power plant that turns
  on to offset grid faults".
- **Perfect foresight & lossless batteries** inherited from the Part A
  formulation (`paper_context.md` §9).
