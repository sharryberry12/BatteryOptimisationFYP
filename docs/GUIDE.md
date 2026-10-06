# A Beginner's Guide to the Battery Optimisation FYP

*Written 2026-10-06 against the repository as it stood that day. Every number
in here comes from a file in the repo; the source is named next to it so you
can check it. This guide explains; it does not replace the detailed documents,
which it points to at the end of each part.*

---

## How to read this guide

You do not need to know power systems or optimisation to start. Each part
builds on the one before it, so read them in order the first time. Every part
ends with a "Where it lives" box naming the files, and a "Read next" line
naming the detailed document for that part.

If you only have ten minutes, read Part 0 and Part 9.

```
Part 0   The whole project on one page
Part 1   The physical world: a house, a tariff, a feeder, the data
Part 2   Part A: scheduling one battery for one day (the QP)
Part 3   The Elermore Vale network model
Part 4   Part B: many batteries under one limit (the VPP)
Part 5   The four kinds of operating envelope (DOE)
Part 6   The studies and what they found
Part 7   Reading and running the repository
Part 8   The paper and where things stand
Part 9   Common confusions, in one place
Appendix A   Glossary
Appendix B   Sign and symbol cheat sheet
Appendix C   Suggested reading order for the existing documents
```

---

## Part 0. The whole project on one page

**The question.** Hundreds of houses have rooftop solar and a home battery.
Each battery is programmed to save its owner money. What happens to the
street's electricity network when they all do that at once, and how should
they be coordinated so that the network stays safe without the owners losing
their savings?

**The answer the project reaches.** Left alone, every battery makes the same
decision at the same minute: charge at full power when the cheap tariff
starts at 22:00. On a real feeder model that creates a new evening peak and
drags customer voltages below the legal minimum. Limits on how much each house
may export do almost nothing about this, because the problem is import, not
export. A time-varying limit on import, called a Dynamic Operating Envelope
(DOE), fixes it. The project builds such envelopes three different ways, from
an assumed shape, from a real substation measurement, and from the physics of
the feeder, and shows what each one costs the households.

**The three layers.** The code is built in the order the physics happens:

```
  Ausgrid data (300 solar homes, one year, half-hourly)
          |
          v
  PART A  dispatch/     one house, one day: a quadratic program picks the
                        battery schedule that flattens the grid flow
          |
          v  outputs/profiles/*.csv  (48 numbers per house per day)
          |
  NETWORK network/      the real Elermore Vale 11 kV feeder in OpenDSS;
                        replay those schedules and read the voltages
          |
          v
  PART B  vpp/          many houses coupled by one feeder-level limit;
                        two ways to enforce it (one boss vs pre-sliced)
          |
          v
  STUDIES studies/      the experiments: peak duty, static vs DOE on the
                        heatwave day, 105-day sweep, where the batteries
                        sit, network-aware per-load envelopes
          |
          v
  PAPER   FYP_final_paper.tex + docs/PAPER_REVISION_BRIEF.md
```

**The one idea to hold onto.** There is a tension between the best answer
for the street and the practical answer for the households. The best answer
needs one controller that sees every house. The practical answer lets each
house decide alone, but then the allowances have to be cut cleverly, or the
street ends up worse off than if the batteries had never been installed.

---

## Part 1. The physical world

### 1.1 A house with solar and a battery

Four flows of power meet at the house's connection to the street. All are
in kilowatts (kW), averaged over a half-hour interval.

| Symbol in code | Symbol in paper | Meaning | Sign |
|---|---|---|---|
| `load` | ℓ | what the house consumes | always ≥ 0 |
| `pv` | g | what the roof generates | always ≥ 0 |
| `b` | β | battery power | **+ discharging, − charging** |
| `p` | π | grid power at the meter | **+ importing, − exporting** |

Power must balance, so

```
p = load − pv − b
```

Define `net = load − pv`, the grid flow the house would have with no battery.
Then `p = net − b`. This one line is used everywhere in the code.

Two worked examples with small numbers:

- Midday: load 1 kW, PV 3 kW, so `net = −2` (the house would export 2 kW).
  Charge the battery at 2 kW, so `b = −2`. Then `p = −2 − (−2) = 0`. The
  battery has absorbed the surplus and nothing is exported.
- Evening: load 2 kW, PV 0, so `net = +2`. Discharge at 2 kW, so `b = +2`.
  Then `p = 2 − 2 = 0`. The battery is carrying the house.

The battery has two limits. It can charge or discharge at most 5 kW. It holds
at most 10 kWh. Its state of charge (SOC, in kWh) starts the day at half,
5 kWh, and each half hour moves by half the power:

```
soc_k = 5 − 0.5 × (b_1 + b_2 + ... + b_k)
```

Charging at 2 kW for one half hour therefore raises the SOC from 5 to
6 kWh. The project also requires the battery to end the day where it started,
so that one day's schedule does not steal energy from the next. In code that
is `sum(b) = 0`.

### 1.2 The tariff and the two bills

Electricity is priced by time of day. The project uses the tariff of the
paper it reproduces (Ratnam, Weller and Kellett 2015, called R15 throughout):

| Band | Hours | Price |
|---|---|---|
| off-peak | 22:00 to 07:00 | $0.03/kWh |
| shoulder | 07:00 to 14:00 and 20:00 to 22:00 | $0.06/kWh |
| peak | 14:00 to 20:00 | $0.30/kWh |

Solar exported to the grid earns a feed-in tariff of $0.40/kWh. Notice that
this is more than any import price. That is historically accurate for the
2010 Ausgrid scheme and it has consequences later.

There are two ways a house can be metered, and the project computes both:

- **Gross feed-in, called `fit` in the code (metering topology 1).** The
  roof has its own meter. Every kWh of PV earns $0.40 whatever the house
  does with it. The house pays the tariff on `max(load − b, 0)`, meaning
  load not covered by the battery. A battery discharge that exceeds the load
  earns nothing.
- **Net metering, called `net` in the code (metering topology 2).** One
  meter on `p`. Imports pay the tariff, exports earn $0.40.

A day's **savings** is the no-battery bill minus the with-battery bill.

### 1.3 The feeder, and why voltage matters

The street is not a wire with infinite capacity. Power reaches houses through
a chain:

```
132 kV transmission
   -> Jesmond 132/11 kV zone substation (one big transformer)
      -> 11 kV feeder backbone (the "Elermore Vale feeder")
         -> 23 distribution transformers, 11 kV down to 433 V (250 V per phase)
            -> low-voltage (LV) street cables
               -> 1,785 houses, nominally 230 to 240 V each
```

Think of voltage as pressure in a water pipe. Every house drawing power pulls
the pressure down a little at every house downstream of it on the same cable.
Every house pushing power out (solar export) pushes the pressure up. A few
houses do not matter. Hundreds doing the same thing at the same minute do.

The legal window for supply voltage in Australia (AS 60038) is 230 V plus
10 % and minus 6 %, that is 216.2 V to 253 V. The code expresses voltage in
**per unit (p.u.) of 230 V**, so the window is 0.94 to 1.10 p.u. A reading
of 0.75 p.u. means 172 V, a serious brown-out.

> **A note on the base voltage.** The model code (`V_NOM` in
> `network/elermorevale_openDSS.py`) divides by 230 V and has done so since
> the OpenDSS model was first added. Several internal documents (CLAUDE.md,
> network/README.md, studies/NETWORK_AWARE_DISPATCH.md, docs/WALKTHROUGH.md)
> say "240 V base"; they are wrong about the code, and the arithmetic in
> NETWORK_AWARE_DISPATCH about the boost-tap transformer ("256 V = 1.068 pu")
> uses the wrong base. The paper's text (230 V, 216.2 to 253 V) matches the
> code. One consequence worth knowing: the distribution transformers deliver
> about 250 V per phase at no load, which is already 1.087 p.u. on the 230 V
> base. A lightly loaded LV cable sits within 1.2 % of the upper limit
> before any solar export, and a transformer with a +2.56 % boost tap sits
> above it. This explains most of the "over-voltage" the model reports.

A **violation point** is one monitored house at one half hour outside the
window. The model monitors 100 of the 1,785 houses, so a day has
100 × 48 = 4,800 points. The count measures how widespread and how long a
problem is, not how bad; it is always paired with the worst voltage seen.

### 1.4 The data

Three data files live in `data/`, none of them in git.

**`data.csv`, the one-year Ausgrid window.** 300 houses in the Ausgrid area
with rooftop solar, 1 July 2010 to 30 June 2011. Each row is one house, one
day, one of three channels, with 48 half-hourly values in kWh. The channels
are GC (general consumption), CL (controlled load, usually hot water) and GG
(gross generation). The code adds `load = GC + CL`, sets `pv = GG`, and
converts kWh per half hour to kW by dividing by 0.5 once at load time. Every
calculation after that is in kW.

Not every house has clean data. The dataset paper (Ratnam et al. 2017, called
R17) gives rules, which `dispatch/osqp_daily.py` applies: drop a house if on
any day its load never exceeds 6 W, or its PV never exceeds 60 W, or its
daily PV is tiny, or it reports PV before 5 am (a sign of a wiring error).
About **152 houses** survive. The original Part A paper said 145; the
thresholds that produced 145 were not recorded, and the two counts are an
open item for the final paper.

**`data_3_years.csv`.** The same houses, July 2010 to June 2013. Used only
by the peak-duty study, which wants as many peak events as possible.

**`Jesmond-132_11kV-FY2011.csv`.** Ausgrid's measured loading of the Jesmond
zone substation, which feeds Elermore Vale, in MW every 15 minutes. Only
13 January to 30 April 2011 is populated. This is what turns the DOE from an
assumed shape into a measured one.

> **Where it lives.** Data loading and cleaning: `dispatch/osqp_daily.py`
> (`load_dataset`, `identify_clean_customers`, `clean_dataset`). The vpp layer
> caches the cleaned day arrays as a pickle in `outputs/cache/` so the
> one-minute cleaning pass runs once. Substation file: `studies/static_vs_doe_replay.py`
> (`load_jesmond_day`). Paths: `paths.py`.
>
> **Read next.** `data/README.md`, then the "Data" section of
> `dispatch/FORMULATION.md`.

---

## Part 2. Part A: scheduling one battery for one day

### 2.1 What an optimisation problem is

An optimisation problem has three parts. **Decision variables** are the
numbers you are free to choose. The **objective** is a formula that scores a
choice; you want the lowest score. **Constraints** are rules a choice must
obey. A solver is a program that finds the choice with the lowest score
among all the choices that obey the rules.

A **quadratic program (QP)** is the special case where the objective is a
sum of squares (plus linear terms) and the constraints are linear. QPs are
**convex**, which means there is exactly one best answer and a solver will
find it reliably and fast. Almost everything in this repo is a QP, on
purpose. Adding on/off switches or integer choices would break that, and the
design documents warn against it.

### 2.2 The R15 QP

For one house on one day there are 48 decision variables, the battery power
in each half hour: `b = [b_1, ..., b_48]`.

The objective is

```
minimise   Σ_k  h_k × (net_k − b_k)²
```

that is, the weighted sum of the squared grid flow. Squaring punishes big
flows in either direction, import or export. The weights `h_k` are bigger in
expensive hours, so the battery works hardest to keep grid flow small when
power is dear. The weights start as the tariff divided by the cheapest
tariff: 1 in off-peak, 2 in shoulder, 10 in peak.

The constraints are the three families from Part 1:

```
−5 ≤ b_k ≤ 5                      rate limit, every k
 0 ≤ 5 − 0.5 Σ_{j≤k} b_j ≤ 10     SOC stays in the battery, every k
 Σ_k b_k = 0                      end the day where you started
```

The SOC rule is written with a lower-triangular matrix of ones (called `T`
in the paper) that turns "sum of all b up to k" into a matrix product, which
is what a QP solver needs.

**The objective is not dollars.** This is the single most important thing
to understand about Part A. The QP flattens the weighted grid flow; it never
sees the bill. Flattening and saving money usually agree, which is why the
method works, but not always. On a sunny summer day in `fit` mode the QP
charges at midday to flatten the export, paying the shoulder price on the
load meter, while the PV earns its $0.40 regardless. In the evening there is
not enough load to use the stored energy, and discharge beyond load earns
nothing. The house loses money that day. Annual savings are still positive
because winter days dominate. `docs/WALKTHROUGH.md` section 1.3 walks through
one such day with real numbers.

### 2.3 The weight heuristic (Algorithm 1 of R15)

Because the objective is a stand-in for the bill, R15 adds a greedy loop to
make it chase dollars. Start with the weights 1, 2, 10. Double the weights of
one tariff band, re-solve the QP, compute the real bill. If savings improved,
keep the change; otherwise discard it. Try each band in turn, repeat up to 20
rounds, stop when no band helps. Weights are capped at 1,000. This is
`optimise_H` in the code and it is why one house-day takes a handful of QP
solves, not one.

### 2.4 OSQP and the "build once, update bounds" rule

The solver is OSQP. It wants the problem written as

```
minimise  ½ xᵀ P x + qᵀ x    subject to   l ≤ A x ≤ u
```

For the R15 QP, `P` is a diagonal matrix of `2 h_k`, `q` is `−2 h_k net_k`,
and `A` stacks the three constraint families. The key observation is that
`A` is the same for every house and every day; only `P`'s diagonal, `q`, and
the bounds `l` and `u` change. So the code builds the OSQP workspace once per
worker process and calls `update(...)` with the new numbers every day. A full
year for one house solves in 9 to 11 seconds, against about 250 seconds
reported by R15 with a different solver.

This rule has a sharp edge. OSQP's `update()` **cannot change the shape of
`A`**, and version 1.x silently ignores an `A=` keyword rather than raising
an error. The first DOE implementation did exactly that, and for weeks every
DOE scenario produced the same, unconstrained schedule while reporting
"98.5 % compliant". It was caught on 2026-08-16. The fix is to put every row
you might ever need into `A` at setup, inactive at ±infinity, and only ever
move the bounds. Every solver in the repo now works that way.

### 2.5 What Part A produces

Running `python dispatch/osqp_daily.py` solves every house-day for both
metering modes in a multiprocessing pool and writes two long-format CSVs,
`outputs/profiles/fit_profiles.csv` and `net_profiles.csv`, with one row per
house, date and interval: load, PV, battery, grid, SOC, daily savings. These
CSVs are the contract between Part A and the network model.

Headline numbers (`dispatch/FORMULATION.md` section 6):

| Quantity | This repo | R15 |
|---|---|---|
| Mean annual savings, gross feed-in | $364.95 | $348 |
| Mean annual savings, net metering | $87.02 | $90 |

### 2.6 The DOE extension of Part A

`dispatch/osqp_daily_with_DOE.py` is a copy of `osqp_daily.py` (not an
import; fixes must be mirrored) that adds a per-household envelope on the
grid flow: `doe_min_k ≤ p_k ≤ doe_max_k`, with `doe_min ≤ 0` capping export
and `doe_max ≥ 0` capping import.

An envelope can be impossible for a battery alone. A house with 7 kW of PV at
noon cannot get under a 2 kW export cap with a battery that charges at 5 kW.
So the decision vector grows to three blocks, `x = [b | c | s]`:

- `c`, **curtailed PV**, between 0 and the PV available. This is what a real
  inverter does under a DOE. The grid flow becomes `p = load − pv + c − b`.
  With curtailment the export cap can always be met, so it is enforced hard.
- `s`, **import shortfall**, at least 0. You cannot switch off a house's
  load, so the import cap is soft: `p ≤ doe_max + s`, and `s` records by how
  much it was missed.

Both get a linear penalty of 100 × `h_k` per kW in the objective, far larger
than any flattening gain, so they are used only when the battery physically
cannot comply. The bill is computed on `pv − c`, so curtailed energy really
is lost income. With no envelope `c = s = 0` and the result equals the plain
QP to solver tolerance; `tests/test_doe_constraints.py` pins that.

> **Where it lives.** `dispatch/osqp_daily.py`: `build_constraints`,
> `solve_battery`, `build_tariff`, `bill_topology1/2`, `optimise_H`,
> `run_all`, `save_profiles`. `dispatch/osqp_daily_with_DOE.py`:
> `generate_doe_envelope`, the three-block `build_constraints`, `DispatchResult`.
>
> **Read next.** `dispatch/README.md`, then `dispatch/FORMULATION.md`
> sections 2 to 4 and 9, then `docs/WALKTHROUGH.md` Part 1 with a Python
> session open.

---

## Part 3. The Elermore Vale network model

### 3.1 What the model is

Ausgrid's Elermore Vale feeder, in Wallsend NSW, is distributed by CSIRO as a
GridLAB-D model: a folder of text files in `network/glm/` describing every
transformer, cable, switch and house. GridLAB-D is one power-flow simulator;
the project uses another, OpenDSS, because it runs inside the same Python
process as the scheduler. So `network/elermorevale_openDSS.py` **translates
the GridLAB-D files into OpenDSS commands at runtime** every time the model is
built. There is no static OpenDSS file to open and inspect; the translation
itself has to be verified.

The census of what gets built (`network/MODEL_VERIFICATION.md`):

| Thing | Count |
|---|---|
| Zone substation transformer, 132/11 kV | 1 (plus an on-load tap changer) |
| Distribution transformers, 11 kV / 433 V | 23 |
| Lines, overhead / underground | 1,743 / 422 |
| Residential loads | 1,785 |
| Rooftop PV systems in the source | 155 |
| Redflow batteries in the source | 40 |

The 155 PV systems and 40 batteries from the source files are only used in
"full" snapshot mode. In profile mode, the one that matters, they are skipped
and every house's PV and battery enter through its grid profile instead.

### 3.2 How a schedule is replayed

`simulate_scenario()` does one day of one scenario:

1. Rebuild the circuit from the GLM files (about a second).
2. Map the 152 houses onto the 1,785 loads **round-robin**: sorted load 1
   gets house 1, load 2 gets house 2, ..., load 153 gets house 1 again. Each
   profile is used about 11.7 times. This is the single biggest modelling
   choice in the project and is discussed in 3.4.
3. Attach a 48-point `LoadShape` to every load from its house's `grid`
   column (or `load − pv` for the no-battery baseline). Batteries and PV are
   not separate elements; the net flow is what the cable sees.
4. Put voltage monitors on 100 loads spread evenly along the list.
5. Run 48 half-hourly power-flow solves.
6. Read the 100 monitors, the zone-transformer power and the losses. Count
   points above 1.10 or below 0.94 p.u.
7. Refuse to continue if any monitor read zero volts (`DeadMonitorError`).
   The engine's "converged" flag has twice been shown to be true on a dead
   circuit, so it is never trusted alone.

Running the year is the same thing 365 times; `run_full_sweep()` writes one
row per day to `opendss_sweep_results.csv`.

### 3.3 How the model was verified, and what the bugs taught

`network/MODEL_VERIFICATION.md` describes a four-level pyramid: unit tests
on the pure translation functions, invariants between the source files and
the built circuit, known-answer physics (no load means flat voltage and zero
losses; source power equals load plus losses), and finally the same
operating point solved independently in GridLAB-D 5.3 and compared node by
node. After the 2026-08-24 impedance rework the two engines agree to 0.001 %
at 11 kV and 0.020 % worst case at LV.

The defect log is worth reading once because each entry is a lesson:

- Bare lengths in the GridLAB-D files are in **feet**, not metres. Every
  11 kV section was 3.28 times too long until 2026-08-13.
- OpenDSS reads a load's `kv` as line-to-neutral for one-phase loads but
  line-to-line for three-phase loads. Seven loads were silently running at
  2.4 times their commanded power.
- A one-phase line using a three-phase line code grew two phantom phases,
  energising about 1,900 nodes that did not exist.
- Twenty-five fuel-cell units declared as `load` objects on phases their
  service point does not carry floated at zero volts and put a fake
  `v_min = 0` into every sweep until 2026-08-16.
- Two or more OpenDSS `Storage` elements collapse this circuit to a dead
  state that still reports converged. The 40 source batteries are modelled
  as `Generator` elements instead.
- Line impedances were reduced incorrectly from the source matrices, running
  the 11 kV backbone at about twice its true impedance. Fixed 2026-08-24 by
  `network/line_impedance.py`, which reproduces GridLAB-D's own Carson
  equations.

The practical consequence: **network results have epochs.** Anything before
2026-08-16 is invalid, anything between then and 2026-08-24 is on the old
impedances, and everything quoted in the current documents was regenerated
on 2026-08-24 or later.

### 3.4 Why the model exaggerates

Read before quoting any violation count:

- 152 profiles cycled over 1,785 loads means the whole feeder makes the same
  decision at the same minute. Real houses are more diverse.
- Every house in the dataset is a solar home, so the feeder is modelled as
  100 % solar, far above 2011 reality.
- The no-load LV voltage is about 1.087 p.u. on the 230 V base (see the note
  in 1.3), and one transformer, `HP00007159`, carries a +2.56 % boost tap
  that puts its seven monitored houses above 1.10 p.u. whenever they are
  lightly loaded, whatever any battery does. Real networks fix that with a
  tap change, which the model does not do.
- No inverter volt-var or volt-watt response, which Australian standards now
  require and which would trim over-voltage in reality.
- Only 100 of 1,785 houses are monitored.

The guidance throughout the project is therefore to compare scenarios
against each other on the same model, and to read shapes and deltas rather
than absolute levels.

### 3.5 What the full-year replay found

`studies/NETWORK_AWARE_DISPATCH.md` is the write-up. The plain R15 schedule
replayed over the year, 100 monitors × 48 × 365 points:

| Dispatch | Violation points | over / under | Mean daily V min | Mean daily peak at zone TX | Savings per house per year |
|---|---|---|---|---|---|
| No battery | 112,070 | 107,269 / 4,801 | 0.923 | 2.42 MW | $0 |
| R15 QP (gross FiT) | 84,927 (−24 %) | 69,799 / 15,128 | 0.847 | 3.75 MW | $365 |
| QP + export cap, tightest | 80,366 (−28 %) | 65,245 / 15,121 | 0.847 | 3.74 MW | $328 |
| QP + 3 kW import cap | 76,862 (−31 %) | 66,452 / 10,410 | 0.872 | 3.32 MW | $350 |
| QP + 2 kW import cap | **66,422 (−41 %)** | 60,344 / 6,078 | 0.910 | 2.81 MW | $323 |

Three findings carry into everything after:

1. **The QP moves the problem rather than removing it.** It cuts midday
   over-voltage but triples under-voltage, because all 1,785 batteries start
   charging at 5 kW at 22:00. On a 25-day sample, 96 % of the QP's
   under-voltage points fall in 22:00 to 24:00.
2. **Export caps are nearly useless here.** The plain QP already charges into
   the solar peak, so the feeder imports at noon. Caps only clip a handful of
   very large PV systems, at a cost that lands almost entirely on those
   houses.
3. **An import cap is the lever that works.** A flat 2 kW per house takes
   under-voltage back near the no-battery level for 12 % of the savings.
   What it leaves is the same-minute step to the cap at 22:00, which a flat
   number cannot address. That is the motivation for Part B.

The zone substation's tap changer (`--oltc`) was also tried and never moves:
the 11 kV bus stays within 0.5 % of nominal all day, and the problems are
downstream on the LV cables, out of its reach.

> **Where it lives.** `network/elermorevale_openDSS.py` (`parse_glm`,
> `build_elermorevale`, `map_customers_to_network_loads`,
> `select_monitored_loads`, `attach_shapes`, `simulate_scenario`,
> `run_full_sweep`), `network/line_impedance.py`, `network/validation/`,
> `network/diagnostics/diag_violation_attribution.py`,
> `network/elermorevale_gui.py` (an HTML dashboard).
>
> **Read next.** `network/README.md`, then `network/MODEL_VERIFICATION.md`,
> then `studies/NETWORK_AWARE_DISPATCH.md`, then `docs/WALKTHROUGH.md`
> Part 2.

---

## Part 4. Part B: many batteries under one limit (the VPP)

### 4.1 Why houses do not simply add up

Part A solves each house alone, which is embarrassingly parallel. A Virtual
Power Plant (VPP) changes one thing: the network company issues a limit on
the **sum** of all the houses' grid flows, not on each house:

```
D_min,k  ≤  Σ_i p_i,k  ≤  D_max,k       for every half hour k
```

`D_min ≤ 0` caps the street's total export, `D_max ≥ 0` caps its total
import. That one line couples every house to every other house. The good news
is that the coupled problem is still a convex QP: a stacked objective, linear
constraints, one global optimum.

### 4.2 The b-space trick

Because each house's grid flow is `p_i = net_i − b_i`, the coupling can be
rewritten so that it only involves the batteries:

```
agg_net_k − D_max,k  ≤  Σ_i b_i,k  ≤  agg_net_k − D_min,k
```

where `agg_net` is the sum of every house's no-battery flow. Load, PV and
the envelope now enter only through the bounds, the coupling rows are just
`[I I ... I]`, and the quadratic matrix stays diagonal. This is why both VPP
methods keep the "build once, update bounds" speed of Part A, and why 152
households solve together in about two seconds.

### 4.3 Method A: centralised QP (the ground truth)

`vpp_common.solve_centralised()` stacks all N household constraint blocks
block-diagonally and appends 48 coupling rows. One OSQP solve gives the
exact best fleet schedule. Its **soft** mode adds two slack vectors,
`s_up` on the import side and `s_lo` on the export side, each with a linear
penalty of 1,000 per kW. The solve then never fails; when the 5 kW / 10 kWh
fleet physically cannot meet the envelope, the slacks say when and by how
much. The duals of the coupling rows are the envelope's shadow price: what
one more kW of headroom in that half hour would be worth to the fleet.

Assessment: optimal, but it needs every house's load, PV and objective in
one place, and one failure takes down the whole solve.

### 4.4 Method B: two-stage allocation (deployed practice)

This is what SA Power Networks actually did in its Flexible Exports trial.

**Stage 1.** The network company splits the feeder envelope into
per-household slices so that the slices add up to no more than the budget.
`allocate()` in `two_stage_doe_allocation.py` implements four rules:

| Rule | Export side | Import side |
|---|---|---|
| `equal` | 1/N each | 1/N each |
| `prorata_pv` | in proportion to each house's peak PV | 1/N each |
| `prorata_surplus` | in proportion to forecast surplus `(pv − load)⁺` | in proportion to forecast need `(load − pv)⁺` |
| `maxmin` | water-filling against each house's physical export cap | the same allowance for all, except houses whose unavoidable import exceeds it get that floor |

**Stage 2.** Every house solves its own Part A QP alone with its slice as a
per-household envelope. `vpp_common.HouseholdSolver` appends 48 identity
rows to the Part A block. A slice tighter than ±5 kW is relaxed to the
battery limit and the gap recorded. With `--soft` the house gets the same
two slacks as Method A, so a slice it cannot meet on energy is met
best-effort and the remainder reported as shortfall. Without `--soft`, an
infeasible house falls back to no battery at all and is counted in
`n_failed`; the early "300 to 900 % gap" numbers measured that fallback, not
the allocation, and have been superseded.

Assessment: suboptimal by construction, because the allowance is decided
before anyone reveals what they would do with it. The gap to Method A is the
headline measurement. Private, robust, realistic.

### 4.5 Frozen weights

Each household's weights `h_i` are taken from the uncoupled Part A heuristic
and then **frozen** for every coupled solve. If the heuristic were re-run
inside a coupled loop, each method would be minimising a different objective
and the gap numbers would mean nothing. The uncoupled schedule is also kept
as `b_uncoupled`, the "what happens today" baseline.

### 4.6 Fairness

Savings are reported as a vector, one entry per house, and summarised by two
indices. **Jain's index** is `(Σx)² / (N Σx²)`: 1 means everyone saved the
same, 1/N means one house got everything. The code clips negative savings to
zero first; the paper must say so (clipped 0.71 to 0.66 versus raw 0.67 to
0.60 on the test day). **Gini** is 0 for perfect equality and 1 for one
house taking all. This matters because AEMO's fairness report asks exactly
who pays for network relief, and the answer here is that the centralised
solve leaves 136 of 152 houses worse off.

### 4.7 The pipeline and the run folders

`vpp/run_vpp_network.py` chains everything: build the ensemble, run a method
through `vpp_registry.py`, export three CSVs (no battery, uncoupled, coupled)
through `vpp_export.py` in exactly the format the network script reads,
replay each on Elermore Vale, and plot measured feeder-head power against
the envelope. Every run gets a folder `outputs/runs/<id>/` with a
`manifest.json` recording every argument, the envelope, the git commit and
the results. The manifests are tracked in git; the CSVs are not. Any figure
in the paper can be traced to a manifest.

### 4.8 Methods that were removed

Four other coupling methods were built and deleted on 2026-09-11: dual
decomposition (price coordination), sharing ADMM, one-shot price-based
control, and FCAS co-optimisation. `vpp/VPP_EXTENSION.md` section 5 records
why. The one lesson that survived is a warning: anything that coordinates by
broadcasting one price herds every battery into the price trough, which is
the 22:00 problem all over again. The two retained methods enforce explicit
envelopes.

> **Where it lives.** `vpp/vpp_common.py` (`assemble_ensemble`,
> `feeder_envelope`, `HouseholdSolver`, `solve_centralised`, metrics),
> `vpp/centralised_qp/centralised_qp.py`,
> `vpp/two_stage_doe_allocation/two_stage_doe_allocation.py` (`allocate`,
> `water_fill`, `floor_fill`, `run_rule`), `vpp/vpp_registry.py`,
> `vpp/vpp_export.py`, `vpp/run_vpp_network.py`. Tests:
> `tests/test_vpp_methods.py`, `tests/test_two_stage_import_side.py`.
>
> **Read next.** `vpp/README.md`, then `vpp/VPP_EXTENSION.md` sections 1 to
> 4, then the two method READMEs, then `docs/WALKTHROUGH.md` Part 3.

---

## Part 5. The four kinds of operating envelope

"DOE" is used for four different things in this repo. Keep them apart.

### 5.1 Per-household synthetic shapes (Part A)

`generate_doe_envelope()` in `osqp_daily_with_DOE.py`. A scenario name and a
base export limit give a 48-value pair for one house. `conservative` is a
flat 80 % of the base (2.4 kW for a 3 kW base), `tight` is TOU-shaped at
50 / 30 / 15 % of the base, `rolling` is a hand-written curve. The import
side is a flat cap or infinity. These probe the scheduler; they know nothing
about the network. They drove the full-year sweeps in Part 3.5.

### 5.2 Feeder-level synthetic shapes (VPP)

`feeder_envelope()` in `vpp_common.py`. A per-household base times the
number of houses, shaped `static`, `tight_tou` or `dynamic_solar`, plus a
flat import cap times N. These are the `--scenario` options on every VPP
script and the fixtures in the VPP tests.

### 5.3 Zone-substation headroom DOE (the paper's DOE)

`zone_headroom_envelope()` in `studies/static_vs_doe_replay.py`. The first
envelope derived from a measurement. Treat the 152-house ensemble as one
slice of the Jesmond substation's load and everything else as fixed:

```
other_k    = L_jesmond,k − P_base,k      the rest of the substation
headroom_k = C_zone − other_k            what is left for the fleet
D_max,k    = min(headroom_k, C_feeder)
```

`C_zone` defaults to 95 % of the day's measured peak. `C_feeder` defaults to
the ensemble's no-battery peak, so the fleet may never create a new feeder
peak. Inside the substation's stress window the limit drops below the
no-battery flow and the fleet must discharge; outside it the limit relaxes
to the feeder cap and the fleet may charge. The export side stays a flat
1.5 kW per house in every regime, so "static" and "DOE" differ only through
this import limit. On 5 February 2011 the limit bottoms out at 101 kW
(0.66 kW per house) at 16:30 and sits at 438 kW for the other 36 intervals.

### 5.4 Network-aware per-load envelope (built 2026-10-05)

The newest piece, after Mahmoodi et al. (IEEE Trans. Smart Grid 2024),
reduced to real power. It differs from 5.3 in two ways: it knows about
voltage, not just substation power, and it is **per load** (1,785
customers), not per household, because a per-household cap would be
averaged over the dozen locations each profile is dealt to.

**Step 1, voltage sensitivities** (`network/voltage_sensitivity.py`). For
one half hour, set every load to its no-battery flow and solve. Then, one
load at a time, add 1 kW, solve again, and record how much each of the 100
monitored voltages moved. The result is a 100 × 1,785 matrix `dv_dp` in p.u.
per kW, the feeder's own linearisation at that operating point. Almost every
entry is negative (more import, lower voltage); about a third are small
positives from cross-phase coupling on the unbalanced network. The solver
tolerance is tightened to 1e-6 for this, because at the default 1e-4 the
solver noise was as large as the signal. This is repeated for every half
hour, because the sensitivities grow as the feeder gets more loaded.

**Step 2, worst-case rows** (`vpp/network_doe.py`). Each customer gets an
import cap `hi ≥ 0` and an export cap `lo ≤ 0`. The fleet must be safe for
any combination of flows inside the caps, so each limit is written for its
worst case:

```
under-voltage, monitor m:   Σ_j |D_mj|⁻ · hi_j  ≤  v_noload,m − 0.94
over-voltage,  monitor m:   Σ_j |D_mj|⁻ · lo_j  ≤  1.10 − v_noload,m
substation (optional):      Σ_j hi_j            ≤  zone headroom (from 5.3)
```

A row that zero exchange cannot satisfy, such as the boost-tap transformer
sitting above 1.10 at no load, is dropped and counted rather than forced
onto the customers.

**Step 3, projection.** The caps start at the widest thing each customer
could use (own flow plus the full 5 kW battery rate) and are pulled back as
little as possible, least-squares, until every row holds. OSQP would not
converge on this dense problem, so it is solved through its dual with
L-BFGS-B, which takes 0.4 to 2 seconds per half hour.

**Step 4, dispatch.** Each of the 1,785 loads solves alone in a soft
`HouseholdSolver` under its own caps. No allocation stage is needed, because
the envelope is already per customer. Export beyond a cap is clipped as PV
the inverter would spill and earns nothing; import beyond a cap is reported
as shortfall.

A deliberate choice: the positive cross-phase entries are left out by
default (`cross_phase=False`). Keeping them makes the caps safe for every
combination including one phase importing while another exports, but on
this feeder it left 61 % of loads unable to import their own demand at
midnight.

> **Where it lives.** 5.1 `dispatch/osqp_daily_with_DOE.py`; 5.2
> `vpp/vpp_common.py`; 5.3 `studies/static_vs_doe_replay.py`; 5.4
> `network/voltage_sensitivity.py`, `vpp/network_doe.py`,
> `studies/network_doe_study.py`, with tests `tests/test_voltage_sensitivity.py`,
> `tests/test_network_doe.py`, `tests/test_network_doe_study.py`.
>
> **Read next.** The module docstrings of `vpp/network_doe.py` and
> `network/voltage_sensitivity.py`, which are the design notes for 5.4.

---

## Part 6. The studies and what they found

Each study is a script in `studies/` that writes a run folder under
`outputs/runs/` with a `summary.csv`, a `manifest.json` and figures.

### 6.1 Peak duty: the VPP as a peaker plant

`peak_duty_analysis.py` and `replay_peak_event.py`;
write-up `studies/PEAK_DUTY_FINDINGS.md`. Over three years, the 300-house
aggregate demand peaks at 772 kW (2.57 kW per house) at 18:30 on Saturday
5 February 2011, the NSW heatwave. If the feeder must never exceed 70 % of
that peak, 104 houses (about one in three) with 5 kW / 10 kWh batteries
could cover the excess, running about 17 to 32 hours a year. The sizing is
set by **energy** not power: the worst event is a 7.5-hour evening plateau,
so battery kWh matters and inverter kW does not. Replayed on the feeder, the
fleet shaves 1,556 kW off the zone transformer and lifts the worst voltage
from 0.826 to 0.870 p.u.

### 6.2 Static limits versus the DOE on the heatwave day

`static_vs_doe_replay.py`, the paper's core experiment. Three regimes for
the 152 houses on 5 February 2011, all through the feeder model:

| Regime | Feeder peak | Peak time | Substation excess left | Fleet savings | Zone TX peak (×11.7) | Worst voltage | Violation points |
|---|---|---|---|---|---|---|---|
| No battery | 438 kW | 20:00 | 693 kWh | $0 | 5.64 MW | 0.749 | 340 |
| Static limits | 594 kW (+36 %) | 22:00 | 64 kWh | $260.5 | 7.66 MW | 0.648 | 297 |
| DOE (zone headroom) | 438 kW (0 %) | 21:00 | 0 kWh | $205.7 | 5.63 MW | 0.714 | 288 |

Under static limits the tariff alone relieves most of the substation excess
during the 14:00 to 20:00 peak band, and then every battery recharges at
once at 22:00, creating a new peak 36 % above the no-battery peak. The DOE
holds the feeder at its no-battery peak, removes the substation excess
entirely, and costs the households 21 % of their savings on that day,
unevenly: 136 of 152 are worse off.

The same script with `--two-stage-rules` adds Method B. With soft slices on
the same envelope: equal and max-min slices leave about 1.4 MWh of the
required reduction undelivered and re-create a 539 kW peak at 22:00;
need-proportional slices come within 203 kWh at the same household cost as
the centralised solve.

A zone-limit sweep (99 % down to 90 % of the measured peak) shows the
fleet's 1,520 kWh runs out between 95 % and 92 %; below that the solve
reports shortfall and the recharge spills into a new 22:00 peak.

### 6.3 Where the batteries sit

`battery_location_study.py`. The same DOE dispatch injected two ways: behind
every meter (the normal way), or as one big generator at the 11 kV feeder
head carrying the same total. Daily losses: 4,655 kWh with no battery,
4,220 kWh distributed (−9.4 %), 4,654 kWh aggregated (−0.03 %).
Under-voltage points: 338 / 287 / 339. The substation sees the same thing
either way; all the network value happens below the 11 kV bus. A VPP modelled
as one plant at the substation has none of the distribution-network value a
DOE is meant to unlock.

### 6.4 The whole season

`doe_day_sweep.py`. The 6.2 recipe on all 105 days the Jesmond record
covers, 14 January to 30 April 2011, zone limit at 95 % of each day's peak:

| Regime | Mean daily peak | Days above the no-battery peak | Substation excess left | Fleet savings | Under-voltage points | Losses |
|---|---|---|---|---|---|---|
| No battery | 180 kW | — | 17.0 MWh | $0 | 1,582 | 82.4 MWh |
| Static limits | 266 kW (+50 %) | 104 of 105 | 6.3 MWh | $13,650 | 3,297 | 90.3 MWh |
| Centralised DOE | 180 kW | 0 | 0.0 MWh | $12,586 (−7.8 %) | 1,530 | 75.5 MWh |
| Two-stage, max-min | 182 kW | 52 | 3.1 MWh | $9,357 (−31 %) | 1,243 | 61.4 MWh |
| Two-stage, need-proportional | 171 kW | 18 | 1.3 MWh | $9,816 (−28 %) | 1,229 | 65.1 MWh |

The herd is systematic (104 of 105 days). The centralised DOE is met every
day at 7.8 % of savings over the season, much less than the 21 % of the
hottest day. Pre-sliced allowances cost three to four times as much and
still fail to deliver. The two-stage rows post the best network figures,
but for the wrong reason: the slices over-constrain every house, the
batteries cycle less, and the cables carry less current, a cost borne
entirely by the households.

### 6.5 Network-aware per-load envelopes (four days, 2026-10-05)

`network_doe_study.py`, results in `outputs/runs/network_doe_*`. Under-voltage
points out of 4,800, then fleet savings per day:

| Day | Static | Zone DOE | Two-stage max-min | Network-aware import caps |
|---|---|---|---|---|
| 2011-02-13 | 36 / $179 | 0 / $153 | 0 / $115 | 0 / $117 |
| 2011-03-08 | 0 / $45 | 3 / $44 | 0 / $17 | 0 / $28 |
| 2011-04-08 | 20 / $137 | 19 / $131 | 0 / $95 | 0 / $110 |
| 2011-02-05 (peak) | 297 / $260 | 287 / $206 | 290 / $234 | 241 / $222 |

On the three ordinary days the per-load import caps give zero under-voltage
with only 9 to 11 kWh of import above cap, against 77 to 283 kWh for the
two-stage slices, at 63 to 84 % of the zone-DOE savings. On the peak day the
caps cannot be met: the no-battery voltage is already at 0.75 p.u., well
past where the linear model holds. Adding export caps removes every
over-voltage point the customers can influence; what remains is entirely the
seven monitors whose no-load voltage is above 1.10 p.u. On the high-solar
day the export caps curtail 78 kWh and savings go negative. Whether this
goes in the paper is an open decision.

### 6.6 Paper figures

`paper_figures.py` builds the print-quality figures from existing run
folders only, with a `checks.md` that recomputes every number the paper
quotes and reports expected / computed / match.

> **Read next.** `studies/README.md`, then `docs/NUMERICAL_SIMULATION_DRAFT.md`
> (the draft results section with every number's provenance in its
> appendix).

---

## Part 7. Reading and running the repository

### 7.1 Layout

```
paths.py          every location in one place; import from it, never hard-code
data/             inputs (gitignored)
dispatch/         Part A
network/          the feeder model, its GLM sources, validation, dashboard
vpp/              Part B: shared code, two method folders, pipeline
studies/          experiments and write-ups
docs/             this guide, the walkthrough, the paper working docs
tests/            pytest (187 tests, no data needed)
outputs/          everything generated (gitignored except run manifests)
  profiles/       Part A CSVs
  figures/        plots, one subfolder per run
  runs/           one folder per study or pipeline run, with manifest.json
  cache/          cleaned day arrays as pickles
```

`dispatch`, `network`, `vpp` and `studies` are Python packages; scripts
import each other as `from vpp import vpp_common`. Every entry script puts
the repo root on `sys.path` from its own location, so `python <folder>/<script>.py`
works from anywhere.

### 7.2 The commands that reproduce the story

```bash
pip install -r requirements.txt
python -m pytest                                   # ~15 s, proves the model and the QPs

python dispatch/osqp_daily.py                      # Part A, both metering modes (minutes)
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save
                                                   # replay representative days on the feeder
python network/elermorevale_openDSS.py --profiles outputs/profiles/fit_profiles.csv --save --full
                                                   # the full year (~10 min)

python vpp/centralised_qp/centralised_qp.py --n-households 20 --save
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save
python vpp/run_vpp_network.py centralised_qp --n-households 20 --scenario static

python studies/static_vs_doe_replay.py             # the paper's core day (needs the Jesmond CSV)
python studies/static_vs_doe_replay.py --two-stage-rules maxmin,equal
python studies/battery_location_study.py
python studies/doe_day_sweep.py                    # 105 days, ~50 min
python studies/network_doe_study.py --date 2011-04-08   # ~12 min per day
```

### 7.3 Reading a run folder

Open `outputs/runs/<id>/manifest.json` first. It records the arguments, the
date, the ensemble size, the envelope (with `inf` written as the string
`"inf"`), the git commit, and a summary. `summary.csv` has one row per
regime. `dispatch_<case>.csv` are the per-house schedules in the network
schema. `voltages_<case>.npy` are the 100 × 48 monitor readings.
`figures/` holds the plots.

### 7.4 What the tests protect

The suite runs without any data file. It covers the GLM translation
functions, the invariants between source files and built circuit, physics
goldens (a frozen reference solve: source power, losses, voltage extremes),
the GridLAB-D harness, the DOE rows actually binding, the no-envelope case
equalling the plain QP, cross-method consistency on a synthetic ensemble
(Method A hard equals soft when feasible; duals non-zero exactly where the
cap binds; Method B never better than A), and the new sensitivity and
envelope code. Everything else is verified by running scripts and reading
the logged metrics.

### 7.5 Gotchas that have cost real time

- `osqp.update(A=...)` is silently ignored. Pre-allocate rows and move bounds.
- `osqp_daily_with_DOE.py` duplicates `osqp_daily.py`. Fix both.
- Both QP scripts use `multiprocessing.Pool`; on Windows every worker
  re-imports the module, so module top level must be side-effect free.
- `--export-limit` with a single value writes `<scenario>.csv` without a
  `_cap` suffix and will overwrite an earlier set.
- A 16-worker pool needs several GB free; under memory pressure it sits in a
  silent respawn loop.
- The engine's `Converged=True` is necessary, not sufficient. Check for dead
  monitors.
- Interval order is the CSV column order 1 to 48, never parsed clock times,
  because "0:00" is the end of the day.
- The 152 versus 145 customer count is unresolved.
- Internal docs say the voltage base is 240 V; the code uses 230 V (Part 1.3).

> **Read next.** `README.md`, then `CLAUDE.md` (the agent context, which
> doubles as the most compact map of conventions).

---

## Part 8. The paper and where things stand

`FYP_final_paper.tex` is the final paper draft. It reads: introduction;
problem formulation (the household QP in the paper's `[π; β]` stacking, the
OSQP mapping, and an envelope-calculation section that is now out of date);
numerical simulation (data, replication, the feeder envelope from Jesmond,
static versus DOE on 5 February, network validation, battery location,
sensitivity and limitations); conclusion.

Three working documents go with it:

- `docs/PAPER_REVISION_BRIEF.md` lists every correction by line number: the
  introduction promises transmission-distribution co-simulation and
  line-current checks that do not exist; Section II-C says the import side
  is split evenly, which is no longer true; the "300 to 900 %" two-stage gap
  is superseded; the 96 % figure is a 25-day sample, not the full year;
  Jain's index must say "clipped"; `references.bib` is missing. It also
  supplies text-ready paragraphs for the two-stage results, the season
  sweep and the zone-limit sensitivity.
- `docs/NUMERICAL_SIMULATION_DRAFT.md` is the draft results section with an
  appendix mapping every number to a run folder.
- `docs/VPP_COMPARISON_AND_FIGURES.md` plans the regime comparison and
  describes an untested "hybrid" (the network company runs the centralised
  solve and issues each house's optimal profile as its envelope). The brief
  says to mention it only as future work.

The brief's recommended regime names: **No battery**, **Static limits**,
**Centralised VPP** (what the current text calls "DOE"), and **Two-stage
DOE (rule)**.

**Committed 2026-10-06:** the network-aware envelope code
(`network/voltage_sensitivity.py`, `vpp/network_doe.py`,
`studies/network_doe_study.py`, three test files), the manifests and figures
of its four run folders, and the README and CLAUDE.md lines that map them
(commits 0cda6fa, 425f896, 2720f77, 227e6f2).

**Open decisions for the author:** settle 152 versus 145; decide whether the
network-aware envelope is a paper result (it needs the 105-day sweep if so);
cut or soften the co-simulation claims; correct the 240 V statements in the
internal docs; supply the bibliography.

---

## Part 9. Common confusions, in one place

**"Which way is positive?"** Battery `b` is positive when discharging. Grid
`p` is positive when importing. `net = load − pv`. `p = net − b`. Export is
negative `p`. An export cap is a negative number (`D_min ≤ 0`); an import cap
is a positive number (`D_max ≥ 0`).

**"The paper's QP has `[π; β]` but the code only has `b`."** Same problem.
The code substitutes `π = net − b` out, leaving 48 variables instead of 96.
The paper's six-block `A₁` and the code's three-block `A` are the same
constraints.

**"The paper's printed `A₁` has the SOC rows the other way round."** The
paper's inline derivation is right and its printed matrix has a typo. The
code follows the derivation and the tests pin it by forward-simulating the
SOC. `dispatch/FORMULATION.md` section 3 records this.

**"`fit` or `net`?"** `fit` is gross feed-in (PV metered separately, always
paid $0.40). `net` is one meter on the grid flow. Every VPP and study result
in the paper is `fit`.

**"Why can savings be negative?"** The QP flattens; it does not bill. In
`fit` mode discharge beyond load is worthless and charging costs the
shoulder rate, so a sunny low-load day loses money.

**"Which DOE?"** Part 5. Per-household synthetic (Part A sweeps),
feeder-level synthetic (VPP scenarios), zone-substation headroom (the paper),
network-aware per-load (new).

**"Curtailment `c` and shortfall `s`, or slacks `s_up` and `s_lo`?"** Two
different relief mechanisms. Part A's DOE script carries `c` and `s` inside
the household QP. The VPP layer never imports that script; it uses the slack
pair in `HouseholdSolver` and `solve_centralised`. The network DOE study
applies curtailment after the fact to export beyond a cap.

**"Ensemble scale or feeder scale?"** Ensemble scale is the 152 houses
(hundreds of kW). Feeder scale is after replication over 1,785 loads, about
11.7 times larger (MW). The modelled feeder at 5.64 MW is bigger than the
whole real substation at 4.26 MW; compare shapes, not levels.

**"Violation points or voltage?"** Points count breadth and duration; they
can fall while the worst voltage gets worse, as static limits show on 5
February (340 to 297 points, 0.749 to 0.648 p.u.). Always read both.

**"Hard or soft?"** Hard means the envelope is an exact constraint and an
impossible one makes the solve fail. Soft means a penalised slack absorbs
the impossible part and reports it. Every headline VPP result is soft, and
the reported shortfall is a result, not an error.

**"Which model epoch?"** Network numbers before 2026-08-16 are invalid,
between then and 2026-08-24 are on the old impedances, and everything in the
current documents is on the reworked model. DOE numbers before 2026-08-16
are invalid because the constraint was never enforced.

---

## Appendix A. Glossary

| Term | Meaning |
|---|---|
| AS 60038 | The Australian standard setting supply voltage at 230 V, +10 % / −6 % |
| Ausgrid | The NSW distribution network company whose data and feeder model are used |
| b-space | Writing the QP in the battery power alone, with grid flow substituted out |
| Convex | A problem with one global optimum that solvers find reliably |
| DNSP | Distribution Network Service Provider, the network company |
| DOE | Dynamic Operating Envelope: time-varying import and export limits at a connection |
| Dual (shadow price) | The marginal value of relaxing a constraint by one unit |
| Elermore Vale | The real 11 kV feeder in Wallsend NSW used as the test network |
| Ensemble | The set of households solved together on one day |
| Epoch | A period during which the network model gave consistent numbers |
| Feeder | The 11 kV line and everything downstream of it |
| Feed-in tariff (FiT) | The price paid for exported solar, $0.40/kWh here |
| GLM | GridLAB-D model file format; the source of the feeder model |
| GridLAB-D | A power-flow simulator; the source format and the cross-validation reference |
| Headroom | Spare capacity between what a network element carries and its limit |
| Herding | Many controllers making the same decision at the same minute |
| Jain's index | A fairness measure from 1/N (all to one) to 1 (all equal) |
| Jesmond | The 132/11 kV zone substation feeding Elermore Vale; its measured load is in the data |
| Loadshape | An OpenDSS time series attached to a load |
| Net metering | One bidirectional meter on the grid flow |
| OLTC | On-load tap changer; the zone substation's automatic voltage regulator |
| OpenDSS | The power-flow engine used, driven from Python through dss-python |
| OSQP | The quadratic-program solver used throughout |
| p.u. | Per unit: a voltage divided by its nominal, 230 V here |
| QP | Quadratic program |
| R15, R17 | Ratnam et al. 2015 (the algorithm paper) and 2017 (the dataset paper) |
| Replication | Cycling 152 profiles over 1,785 loads, about 11.7 copies each |
| Shortfall | Import above a cap that the battery could not remove |
| Slack | An extra variable that absorbs a constraint violation at a penalty |
| SOC | State of charge, kWh in the battery |
| Static limits | Today's practice: a fixed export cap per house and no import cap |
| Surrogate objective | A stand-in objective (flattening) used instead of the true one (the bill) |
| TOU | Time-of-use tariff |
| Two-stage | The DNSP slices the envelope first; households solve alone second |
| VPP | Virtual Power Plant: many small resources coordinated as one |
| Warm start | Reusing the last solution as the starting point of the next solve |
| Zone substation | The 132/11 kV transformer station at the top of the feeder |

## Appendix B. Sign and symbol cheat sheet

```
Interval k = 1..48, each 0.5 h.  T = 48, DT = 0.5.
load, pv ≥ 0 (kW)                        net = load − pv
b  > 0 discharge, < 0 charge (kW)        |b| ≤ P_MAX = 5
p  > 0 import, < 0 export (kW)           p = net − b            (Part A, VPP)
                                         p = net + c − b        (Part A DOE, c = curtailed PV)
soc_k = 5 − DT × Σ_{j≤k} b_j (kWh)       0 ≤ soc ≤ E_MAX = 10;  Σ b = 0
h_k ≥ 1 weights                          h0 = 1 / 2 / 10 by tariff band
D_min ≤ 0 export cap, D_max ≥ 0 import cap
Feeder coupling:  D_min ≤ Σ_i p_i ≤ D_max
   in b-space:    agg_net − D_max ≤ Σ_i b_i ≤ agg_net − D_min
OSQP duals of the coupling rows: negative where an import cap binds,
   positive where an export cap binds; shadow price μ ≈ −y
Voltage limits: 0.94 ≤ V ≤ 1.10 p.u. on 230 V  (216.2 V to 253 V)
Ensemble scale: 152 houses.  Feeder scale: × ~11.7.
```

## Appendix C. Suggested reading order for the existing documents

1. `README.md` (the map)
2. `dispatch/README.md`, then `dispatch/FORMULATION.md` sections 2 to 4
3. `docs/WALKTHROUGH.md` Part 1, with a Python session open
4. `network/README.md`, then `network/MODEL_VERIFICATION.md`
5. `studies/NETWORK_AWARE_DISPATCH.md` sections 0 to 4
6. `vpp/README.md`, `vpp/VPP_EXTENSION.md` sections 1 to 4, the two method READMEs
7. `docs/WALKTHROUGH.md` Parts 2 and 3
8. `studies/README.md`, `studies/PEAK_DUTY_FINDINGS.md`
9. `docs/NUMERICAL_SIMULATION_DRAFT.md`, then `docs/PAPER_REVISION_BRIEF.md`
10. `CLAUDE.md`, as a compact reference once the rest makes sense
