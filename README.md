# Code Walkthrough: QP Battery Scheduling and OpenDSS Network Validation

## Reference Key

Throughout this document, the following shorthand references are used:

- **[R15]** — Ratnam, Weller & Kellett, *"An optimization-based approach to scheduling residential battery storage with solar PV: Assessing customer benefit,"* Renewable Energy 75, 2015, pp. 123–134. (The algorithm paper.)
- **[R17]** — Ratnam, Weller, Kellett & Murray, *"Residential load and rooftop PV generation: an Australian distribution network dataset,"* International Journal of Sustainable Energy 36(8), 2017, pp. 787–806. (The dataset paper.)
- **[OSQP]** — Stellato et al., *"OSQP: An operator splitting solver for quadratic programs,"* Mathematical Programming Computation 12, 2020, pp. 637–672.

---

# Part 1: `osqp_daily_v2.py` — Battery Dispatch Optimisation

This file implements the core QP-based battery scheduling algorithm from [R15] and applies it to the Ausgrid residential load and PV dataset described in [R17].

---

## 1.1 Constants (lines 73–84)

```python
DT = 0.5            # hours per interval
T = 48              # intervals per day
P_MAX = 5.0         # kW charge/discharge limit
E_MAX_DEFAULT = 10.0 # kWh battery capacity
FIT_RATE = 0.40     # $/kWh export credit
H_BAR = 1000.0      # cap on heuristic weights
```

**Paper reference:** [R15] Section 6.1 defines the simulation parameters. The time window is T = 24 hours, the interval length is Δ = 0.5 hours, giving s = T/Δ = 48 intervals per day. The battery constraints are specified as c₀ = 0.5C (initial SOC at half capacity), and B̄ = B = 5 kW (symmetric charge/discharge power limit). The battery capacity used in Sections 6.2–6.4 is C = 10 kWh. The feed-in tariff compensation rate is ηc(M₁) = ηc(M₃) = $0.40/kWh (Section 6.1, Figure 3).

H̄ = 1000 is the saturation cap used in the heuristic (Section 5, equation 35) to prevent numerical instability in the QP solver when weights become very large.

---

## 1.2 Data Loading: `load_dataset()` (lines 100–123)

This function ingests the raw Ausgrid CSV. The CSV structure is documented in [R17] Section 2 and Table 1: each row represents one customer on one day for one consumption category, with 48 half-hourly energy readings (in kWh) as columns labelled 0:30 through 0:00.

The function performs three transformations:

1. **Melt** — The wide-format CSV (48 time columns) is unpivoted into long format with one row per (customer, date, time, consumption category) tuple.

2. **Pivot** — The three consumption categories (GC, CL, GG) are spread into separate columns. GC is General Consumption (residential load from the gross meter), CL is Controllable Load (utility-switched hot water, present for 137 of 300 customers per [R17] Section 2.1), and GG is Gross Generation (PV output measured directly from the inverter).

3. **Derive load and pv** — Total residential load is computed as `load = GC + CL`, matching the power balance in [R17] equation (1): d(j) = gc(j) − gg(j) + cl(j). The PV generation is simply `pv = GG`. CL is filled with zeros for customers without controllable load, consistent with [R17]'s note that "customers that do not have a controllable load do not have a meter recording CL data."

---

## 1.3 Data Cleaning: `clean_dataset()` (lines 126–140) and `CLEAN_CUSTOMER_IDS` (lines 92–97)

**Paper reference:** [R17] Section 3 describes the cleaning methodology in detail.

The hardcoded list of 55 customer IDs corresponds to the 54 customers in [R17] Table 4 (the "clean dataset" surviving all anomaly checks across 3 years) plus Customer 200 (used as a case study in [R15] Sections 6.2 and 6.5 despite not being in the strict clean set).

The `clean_dataset()` function applies a minimal additional filter: any (customer, date) group that does not have exactly 48 rows is discarded. This catches days corrupted by the daylight saving transitions described in [R17] Section 2.2, where the AEST-to-AEDT changeover produces zero entries in the 2–3 am slot.

Note that this is substantially less aggressive than [R17]'s full cleaning, which also removes customers with any day where max load < 6 W ([R17] Section 3.1), max PV generation < 60 W ([R17] Section 3.2 Category 1), total daily generation < 0.325 kWh with peak < 101 W (Category 2), or pre-dawn generation > 20 Wh (Category 3).

---

## 1.4 Pre-extraction: `extract_day_arrays()` (lines 143–160)

This is a performance optimisation with no mathematical content. It walks the cleaned DataFrame once and produces a plain Python dictionary mapping each customer ID to a list of `(date_string, load_array, pv_array)` tuples, where each array is a contiguous 48-element `float64` numpy vector.

The purpose is to ensure the inner optimisation loop (which runs tens of thousands of times across all customers and days) never touches pandas. Pandas indexing overhead is measurable at this scale — roughly 100 μs per `.values` access versus effectively zero for a pre-extracted numpy array.

---

## 1.5 Battery Constraint Matrices: `build_constraints()` (lines 167–196)

This is the structural heart of the QP. It constructs the constraint matrix A and bound vectors l, u that encode all physical battery constraints from [R15] Section 2.1.

**Decision variable.** The QP optimises over the battery profile b ∈ ℝ⁴⁸. The sign convention follows [R15] equation (1): lₖ = pₖ + gₖ + bₖ, rearranged to pₖ = lₖ − gₖ − bₖ. Positive bₖ means the battery is discharging (delivering power to the load), negative bₖ means it is charging.

**Note on variable elimination.** The paper formulates the QP with decision variable x = [p; b] ∈ ℝ²ˢ and an explicit power-balance equality constraint p + b = l − g ([R15] Lemma 1, equation 13). This code eliminates p algebraically using p = l − g − b, reducing the problem to b ∈ ℝˢ only. This halves the number of decision variables from 96 to 48 and removes s + 1 equality constraints, making the QP substantially cheaper to solve. The solutions are mathematically identical.

**Constraint block 1 — Power limits** ([R15] equation 2):

```
I · b ∈ [−P_MAX, P_MAX]
```

This is the 48 × 48 identity matrix with symmetric bounds, encoding −5 ≤ bₖ ≤ 5 kW for all k. The paper writes this as −B𝟙 ≤ β ≤ B̄𝟙.

**Constraint block 2 — State of charge** ([R15] equations 3–5):

```
−A_soc · b ∈ [−c₀, C − c₀]
```

where A_soc is the 48 × 48 lower-triangular matrix with entries Δ = 0.5, implementing the cumulative sum χₖ = χ₀ − Σⱼ₌₁ᵏ βⱼΔ from [R15] equation (3). The bounds enforce 0 ≤ χₖ ≤ C (equation 4) after subtracting the initial SOC c₀ = 0.5C.

To see why: −A_soc · b gives −Σⱼ₌₁ᵏ bⱼΔ for each k. The SOC at time k is χₖ = c₀ + (−Σbⱼ·Δ). Requiring 0 ≤ χₖ ≤ C is equivalent to −c₀ ≤ −Σbⱼ·Δ ≤ C − c₀.

**Constraint block 3 — Terminal SOC** ([R15] equation 6):

```
𝟙ᵀ · b = 0
```

This enforces χ₄₈ = χ₀, meaning the battery ends the day at the same SOC it started. The paper motivates this in Section 2.1: "In order to avoid an energy-shifting bias in these results, we insist that the state of charge of the battery at the end of a day is the same as the state of charge of the battery at the beginning of the day." This is handled as an equality constraint via OSQP's l = u = 0 on the last row.

The three blocks are vertically stacked into a single sparse constraint matrix A of shape (48 + 48 + 1) × 48 = 97 × 48.

---

## 1.6 OSQP Solver Caching: `_get_solver()` and `solve_battery()` (lines 199–245)

**Paper reference:** The paper does not specify a particular QP solver ([R15] uses MATLAB's `quadprog`). This implementation uses OSQP [OSQP], an operator-splitting method for convex QPs that is well-suited to the problem structure: the Hessian P is diagonal, the constraints are sparse, and the problem is solved repeatedly with only the objective changing between days.

`_get_solver()` constructs the OSQP workspace once per battery capacity. The constraint matrix A and bounds l, u are invariant across all days and all customers (they depend only on E_MAX, P_MAX, and c₀, all of which are fixed). The workspace is cached in `_SOLVER_CACHE` so subsequent calls skip the expensive setup phase.

`solve_battery()` is called for every day of every customer. It takes the load and PV arrays plus the diagonal of H and:

1. Computes the net load: `net = l − g` (the grid profile when b = 0).
2. Forms the quadratic objective. The paper's objective is min Σₖ hₖ · pₖ² ([R15] equation 10). Substituting p = net − b and expanding:

   Σ hₖ(netₖ − bₖ)² = bᵀ diag(h) b − 2(h ⊙ net)ᵀb + const

   OSQP minimises ½xᵀPx + qᵀx, so P = 2·diag(h) and q = −2·(h ⊙ net).

3. Calls `solver.update(Px=..., q=...)` to inject the new objective data without rebuilding the factorisation. This is the key speedup — OSQP's ADMM iterations warm-start from the previous solution, which is typically very close for consecutive days of the same customer.

4. Returns the optimal battery profile b* ∈ ℝ⁴⁸.

The `polish=True` setting enables an iterative refinement step that improves solution accuracy beyond what the ADMM iterations alone achieve, at negligible additional cost for this problem size.

---

## 1.7 Tariff: `build_tariff()` (lines 248–260)

**Paper reference:** [R15] Section 6.1 and Figure 3.

Returns a length-48 vector encoding the time-of-use billing profile ηᵇ(M₂) = ηᵇ(M₃):

| Period | Intervals | Hours | Rate ($/kWh) |
|--------|-----------|-------|-------------|
| Off-peak | 1–14, 45–48 | 00:00–07:00, 22:00–24:00 | 0.03 |
| Shoulder | 15–28, 41–44 | 07:00–14:00, 20:00–22:00 | 0.06 |
| Peak | 29–40 | 14:00–20:00 | 0.30 |

The peak rate is ×5 the shoulder rate and ×10 the off-peak rate, which gives the QP a strong incentive to shift load out of the peak window. The paper notes that peak billing rates are assumed to coincide with generation shortages or peak grid demand ([R15] Section 1).

---

## 1.8 Billing Functions (lines 263–294)

Two functions implement the two metering topologies defined in [R15] Section 3.1.

### `bill_topology1()` — Gross Feed-in Tariff (Metering Topology 1)

**Paper reference:** [R15] Section 3.1 (Metering topology 1), Section 3.3 equation (21), Figure 1.

In this topology, two gross (unidirectional) meters are installed:

- **M₁** measures raw PV generation gₖ and credits the customer at the flat FiT rate of $0.40/kWh. Since M₁ is gross and PV generation satisfies gₖ ≥ 0, it simply records all PV output regardless of where that power goes (battery, load, or grid).

- **M₂** measures power from node 1 to the load+battery (lₖ − bₖ when positive) and bills at the TOU rate. Since M₂ is gross, it records zero when power flows in the reverse direction (i.e., when battery discharge exceeds load: lₖ − bₖ < 0).

The bill is therefore:

```
S^C(H₁) = Δ · Σₖ max(lₖ − bₖ, 0) · tariffₖ − Δ · Σₖ gₖ · FIT_RATE
```

This matches [R15] equation (21) with s(M₁) = ηc(M₁) and s(M₂) = ηb(M₂).

### `bill_topology2()` — Net Metering (Metering Topology 2)

**Paper reference:** [R15] Section 3.1 (Metering topology 2), Section 3.3 equations (23)–(25).

A single bi-directional meter M₃ is installed at the point of common coupling (PCC). It measures pₖ = lₖ − gₖ − bₖ in both directions:

- When pₖ > 0 (importing from grid): billed at TOU rate.
- When pₖ < 0 (exporting to grid): credited at the net-metering rate ($0.40/kWh).

```
S^C(H₂) = Δ · Σₖ max(pₖ, 0) · tariffₖ − Δ · Σₖ max(−pₖ, 0) · net_credit
```

This matches [R15] equation (25) with s(M₃) defined per equation (23).

---

## 1.9 Heuristic for H: `build_H0_diag()` and `optimise_H()` (lines 297–350)

**Paper reference:** [R15] Section 5 (entire section), equations (35)–(38), and the pseudocode in Algorithm 1.

The weighting matrix H = diag(h₁, …, h₄₈) is the primary design parameter in the QP. When all hₖ = 1, the QP simply minimises total squared grid flow (load curve smoothing). The heuristic increases weights at time intervals where the TOU billing rate is high, which makes the QP prioritise reducing grid imports during expensive peak periods while tolerating larger grid flows during cheap off-peak periods.

### Base-line H₀

`build_H0_diag()` implements [R15] equations (36)–(38):

1. h̃ₖ = Σ_M ηᵇₖ(M) — the sum of billing rates across all meters at interval k. For a single-tariff setup this is just the TOU tariff vector.

2. h⁺ = minₖ h̃ₖ — the minimum non-zero billing rate (off-peak = $0.03).

3. H₀ = diag(sat₁^{H̄}(h̃ₖ / h⁺)) — each weight is the ratio of billing rate to minimum billing rate, clamped to [1, H̄]. With the tariff values from Section 6.1:
   - Off-peak intervals: 0.03/0.03 = 1
   - Shoulder intervals: 0.06/0.03 = 2
   - Peak intervals: 0.30/0.03 = 10

### Greedy search

`optimise_H()` implements Algorithm 1 from [R15] Section 5. The algorithm progressively doubles weights starting from the largest tier and working down to the smallest:

1. Group indices by their initial H₀ weight value. With the tariff above, there are three tiers: {peak indices at weight 10}, {shoulder indices at weight 2}, {off-peak indices at weight 1}.

2. Starting from the highest-weight tier, double all weights in that tier (capped at H̄ = 1000).

3. Solve the QP with the trial weights and compute savings. If savings improved, accept the new weights and continue. If not, try the next tier down.

4. Repeat until a full pass through all tiers produces no improvement.

The rationale ([R15] Section 5): "increase base-line weights when electricity billing is high and decrease base-line weights when electricity billing is low, and continue increasing/decreasing so long as the daily residential savings increase."

---

## 1.10 Single-Day Simulation: `simulate_day()` (lines 353–360)

Orchestrates one day for one customer:

1. Calls `optimise_H()` to find the best weighting matrix and corresponding battery profile.
2. Computes the grid profile p = l − g − b ([R15] equation 1).
3. Returns the daily savings (S⁰ − Sᶜ, [R15] equation 27), the battery profile b, the grid profile p, and the optimal weights h.

---

## 1.11 Parallel Execution: `_worker()` and `run_all()` (lines 363–417)

`_worker()` processes all days for a single customer sequentially, accumulating the annual savings Q^C(H) ([R15] Section 4.1 — the sum of daily savings J^C(H) across 365 days). It also collects the full half-hourly profiles (load, PV, battery, grid, SOC) for every day, which are needed downstream by the OpenDSS simulation.

`run_all()` distributes customers across CPU cores using Python's `multiprocessing.Pool`. Each worker process gets its own copy of the `_SOLVER_CACHE` (via fork), so the OSQP workspace is set up once per worker and then reused for all days of that worker's assigned customer. The function returns sorted arrays of customer IDs and annual savings, plus the full profiles dictionary.

---

## 1.12 Profile Export: `save_profiles()` (lines 420–482)

Writes two output formats:

1. **Long-format CSV** (`profiles/{mode}_profiles.csv`) — one row per (customer, date, interval) with columns: customer, date, interval (1–48), hour (0.0–23.5), load_kw, pv_kw, battery_kw, grid_kw, soc_kwh, daily_savings. This is the input consumed by `opendss_daily.py`.

2. **Per-customer day files** (`profiles/{mode}/cust_{id}/{date}.csv`) — each file contains exactly 48 values (the grid profile pₖ in kW, one per line). These are directly usable as OpenDSS `LoadShape` multiplier files without any transformation.

---

## 1.13 Figures (lines 485–647)

Each figure function reproduces a specific figure from [R15].

### `figure2_example_day()` — [R15] Figure 2

Shows load and PV generation profiles (top panel) and the QP-dispatched grid and battery profiles (bottom panel) for one customer on one day. The paper uses this figure to demonstrate how QP energy-shifting smooths the grid profile by charging the battery during periods of excess PV and discharging during peak load.

### `figure5_soc()` — [R15] Figure 5

Plots the battery state of charge trajectory χₖ = χ₀ − Σⱼ₌₁ᵏ bⱼΔ for Customer 75 on 9 January 2011 and Customer 200 on 5 July 2010. The paper uses these two customers to illustrate why some customers lose money with QP scheduling (Customer 75: battery charges during peak and discharges during off-peak, yielding negative savings) while others benefit (Customer 200: battery charges during off-peak and discharges during peak).

### `figure6_daily_savings()` — [R15] Figure 6

Histogram of daily savings J^C(H) for selected customers across all days in the dataset. [R15] Section 6.2 uses this to show the distribution: Customer 75's histogram is skewed negative (loses money most days under metering topology 1), while Customer 200's is skewed positive.

### `figure7_annual_savings()` — [R15] Figure 7

Two-panel histogram showing the distribution of annual savings Q^C(H) across all customers in the ensemble, one panel per metering topology. [R15] Section 6.3 reports mean annual savings of ~$350/yr for topology 1 (FiT) and ~$100/yr for topology 2 (net metering), with 9 and 50 customers respectively losing money.

### `figure8_capacity_sweep()` — [R15] Figure 8

Annual savings plotted against battery capacity C ∈ {0.1, 1, 2, 4, 6, 8, 10, 15, 20, 30} kWh for selected customers. [R15] Section 6.5 uses this to demonstrate that savings approach an asymptote: "a 30 kWh battery providing minimal additional savings over a 15 kWh battery," and that not all customers benefit from larger batteries (Customer 75's losses increase with capacity).

---

## 1.14 Main Pipeline (lines 650–683)

The execution sequence:

1. Load and clean the Ausgrid CSV.
2. Pre-extract day arrays into numpy.
3. Run all customers under both metering topologies (FiT and Net) in parallel.
4. Save half-hourly profiles to disk for downstream OpenDSS consumption.
5. Generate all paper figures.

---
---

# Part 2: `opendss_daily.py` — Network Power Flow Validation

This file takes the battery dispatch results from `osqp_daily_v2.py` and validates them against an actual distribution network power flow model. The paper [R15] claims in its introduction that QP scheduling reduces voltage swings from reverse power flow and peak demand, but never quantifies this at the network level. This module closes that gap.

**Additional references for Part 2:**

- **[DSS]** — Dugan, R.C. & Montenegro, D., *"Reference Guide: The Open Distribution System Simulator (OpenDSS),"* EPRI, 2019. (OpenDSS documentation.)
- **[AS60038]** — Standards Australia, *AS 60038:2012 Standard voltages.* (Australian voltage limits.)
- **[Stetz13]** — Stetz, T., Marten, F. & Braun, M., *"Improved low voltage grid-integration of photovoltaic systems in Germany,"* IEEE Trans. Sustainable Energy 4(2), 2013, pp. 534–542. (Cited in [R15] for LV network voltage issues.)

---

## 2.1 Constants (lines 56–88)

### Voltage limits

```python
V_NOM = 230.0       # V phase-to-neutral
V_UPPER_PU = 1.10   # +10%
V_LOWER_PU = 0.94   # −6%
```

These are the statutory voltage limits from [AS60038], which specifies 230 V +10%/−6% for Australian LV networks. [R15] Section 1 discusses voltage rise (leading to upper limit violations) and voltage dip (lower limit violations) as the key adverse consequences of high PV penetration, citing [Stetz13] and others.

### Transformer parameters

```python
TX_KVA = 200.0
TX_PRIMARY_KV = 11.0
TX_SECONDARY_KV = 0.433  # line-to-line
TX_R_PCT = 1.5
TX_X_PCT = 4.0
```

A 200 kVA, 11 kV / 433 V distribution transformer with Dyn11 (delta primary, star-grounded secondary) winding connection. This is a standard Ausgrid pole-top or pad-mount transformer for residential LV feeders. The 1.5% resistive and 4% reactive impedance are representative of this transformer class. The Ausgrid network operates at 11 kV on the MV side (as stated in [R17] Section 2: "The Ausgrid distribution network covers 22,275 km²").

### Cable parameters

```python
CABLE_R_PER_KM = 0.32   # Ω/km
CABLE_X_PER_KM = 0.073  # Ω/km
```

These represent a 95 mm² aluminium XLPE underground cable, which is the standard Ausgrid LV backbone conductor. The high R/X ratio (~4.4) is characteristic of LV cables and means that voltage drop/rise is dominated by active power flow rather than reactive power. This is why the paper's QP (which controls active power via battery dispatch) is effective at managing voltages — in LV networks, active power control is more impactful than reactive power control.

### Feeder geometry

```python
N_BACKBONE_NODES = 10
SEGMENT_LENGTH_M = 30.0   # 300 m total
SERVICE_LENGTH_M = 15.0
```

A 300 m backbone with 10 equally-spaced nodes is representative of a suburban Australian LV feeder. Customers are connected via 15 m single-phase service drops. The 54 customers in the clean dataset [R17, Table 4] map naturally onto this geometry with roughly 5–6 customers per backbone node across 3 phases.

---

## 2.2 DSS Engine: `get_dss()` (lines 91–103)

Imports the `DSS` singleton from the `dss-python` package. This is the Python binding to the OpenDSS engine [DSS], which provides a full unbalanced three-phase power flow solver. The engine runs in-process (no external executable) and exposes all OpenDSS functionality through property assignments and the `dss.Text.Command` interface.

---

## 2.3 Customer-to-Bus Mapping: `assign_customers_to_buses()` (lines 106–123)

Distributes the 54 customers across the feeder using a round-robin scheme:

- Customer i gets phase `(i mod 3) + 1` (cycling through phases 1, 2, 3).
- Customer i gets backbone node `(i ÷ 3) mod 10 + 1` (distributing evenly along the feeder).

This produces approximately 18 customers per phase and 5–6 customers per backbone node. The round-robin phase assignment ensures reasonable phase balance, which is important because the power flow is solved as a full three-phase unbalanced model — severe phase imbalance would produce misleadingly pessimistic voltage results.

In a real network the phase allocation would come from the utility's GIS data. The round-robin approach is a reasonable proxy for a first-pass analysis.

---

## 2.4 Network Builder: `build_network()` (lines 126–224)

This is the core function that defines the entire LV circuit using OpenDSS command strings. Every element is created programmatically — no external `.dss` files are needed. The function issues commands in this sequence:

### Circuit and voltage source

```
New Circuit.AusLV basekv=11 pu=1.0 phases=3 bus1=sourcebus Isc3=10000 Isc1=10500
```

Creates a balanced three-phase voltage source at 11 kV representing the upstream MV network. The short-circuit currents (Isc3 = 10,000 A, Isc1 = 10,500 A) define the source impedance, modelling a moderately stiff MV bus. This means voltage at the transformer primary will be nearly constant regardless of LV loading — a realistic assumption for a single LV feeder on a large MV network.

### Line codes

```
New Linecode.backbone nphases=3 r1=0.32 x1=0.073 r0=0.96 x0=0.219 units=km
New Linecode.service nphases=1 r1=0.64 x1=0.08 r0=1.92 x0=0.24 units=km
```

Two line types are defined using positive-sequence (r1, x1) and zero-sequence (r0, x0) impedances. The zero-sequence values are set to 3× positive-sequence, which is a standard approximation for cables without explicit neutral modelling. The backbone is three-phase (95 mm² Al) and the service drop is single-phase (smaller conductor, higher resistance).

### Transformer

```
New Transformer.MV_LV phases=3 windings=2
  buses=[sourcebus, lv_node_0]
  conns=[delta, wye]
  kvs=[11, 0.433]
  kvas=[200, 200]
  %Rs=[0.75, 0.75]
  xhl=4.0
```

A three-phase two-winding transformer: delta on the 11 kV primary, star (wye) on the 433 V secondary. The `conns=[delta, wye]` configuration is the Australian standard Dyn11 connection. The winding resistance is split equally between primary and secondary (%Rs = 0.75% each, totalling 1.5%), and the leakage reactance is 4%.

The transformer secondary bus `lv_node_0` is the head of the LV feeder backbone.

### Backbone segments

```
New Line.seg_0 bus1=lv_node_0 bus2=lv_node_1 linecode=backbone length=0.03
New Line.seg_1 bus1=lv_node_1 bus2=lv_node_2 linecode=backbone length=0.03
...
```

Ten segments of 30 m each create the feeder backbone from `lv_node_0` (transformer secondary) to `lv_node_10` (feeder end). Customers near `lv_node_10` experience the largest voltage deviations because they are electrically furthest from the voltage source — this is the "voltage rise is particularly pronounced when large numbers of rooftop PV generators are connected in close proximity" effect described in [R15] Section 1, citing [Stetz13].

### Customer connections

For each customer, two elements are created:

```
New Line.svc_{cid} bus1=lv_node_{node}.{phase} bus2=cust_{cid}.{phase}
  linecode=service length=0.015

New Load.load_{cid} bus1=cust_{cid}.{phase} phases=1
  kv=0.2500 kw=1 pf=1 model=1 status=variable
```

The service line connects one phase of the backbone to the customer's meter bus. The Load element has `kw=1` as a base value that gets multiplied by the LoadShape (so the shape carries actual kW values), `pf=1` (unity power factor — the QP only manages active power), `model=1` (constant P+Q regardless of voltage), and `status=variable` which allows the load shape to contain negative values, representing net export to the grid.

### Monitors

```
New Monitor.tx_power element=Transformer.MV_LV terminal=2 mode=1 ppolar=no
New Monitor.v_{cid} element=Load.load_{cid} terminal=1 mode=0
```

OpenDSS Monitors record time-series data during the simulation [DSS]. Mode 0 records voltage magnitudes, mode 1 records power flows. `ppolar=no` gives rectangular (P, Q) rather than polar (S, θ) coordinates. These are read back after the simulation completes to extract results.

---

## 2.5 LoadShape Attachment: `attach_loadshapes()` and `attach_baseline_shapes()` (lines 227–295)

These functions create a LoadShape object for each customer containing the 48 half-hourly power values for a specific day, then assign it to the customer's Load element via the `daily` property.

The QP scenario uses `day["grid"]` — the optimised grid profile pₖ = lₖ − gₖ − bₖ from `osqp_daily_v2`. The baseline scenario uses `day["load"] − day["pv"]` — the grid profile with no battery (bₖ = 0), i.e., p⁰ₖ = lₖ − gₖ, which is the base-line grid profile defined in [R15] equation (16).

The LoadShape command embeds the 48 values directly as a comma-separated list:

```
New Loadshape.shape_{cid} npts=48 minterval=30 mult=(v1,v2,...,v48)
```

`minterval=30` sets the interval to 30 minutes, matching the Ausgrid dataset resolution.

---

## 2.6 Simulation Execution: `run_daily()` (lines 301–306)

```
Set mode=daily stepsize=30m number=48
Set controlmode=static
```

This configures OpenDSS to solve 48 sequential power flows at 30-minute intervals. At each step, OpenDSS evaluates the LoadShape multiplier for that time step, applies it to each Load element's base kW, and solves the full three-phase unbalanced power flow using its Newton-Raphson solver [DSS]. `controlmode=static` means voltage regulators and capacitor banks (if any) operate at their fixed settings — appropriate here since the synthetic feeder has no voltage control devices.

---

## 2.7 Results Collection (lines 309–360)

### `collect_voltages()`

Reads each customer's voltage monitor (mode 0, channel 1 = V₁ magnitude) and converts to per-unit by dividing by the nominal 230 V phase-to-neutral. Returns a dictionary mapping customer ID to a length-48 voltage array.

### `collect_tx_power()`

Reads the transformer monitor (mode 1, channels 1–6 = P₁, Q₁, P₂, Q₂, P₃, Q₃) and sums across phases to get total active and reactive power at the transformer secondary. Positive values indicate power flowing from MV to LV (normal load); negative values indicate reverse power flow from LV to MV (net PV export exceeding feeder load). This reverse flow is exactly what [R15] Section 1 identifies as the cause of voltage rise.

### `collect_losses()`

Reads total circuit losses from `dss.ActiveCircuit.Losses`, which returns cumulative (watts, vars). These represent I²R and I²X losses across all lines and the transformer. The QP's load-smoothing effect should reduce peak currents and hence reduce losses.

---

## 2.8 Scenario Comparison: `simulate_scenario()` and `simulate_day_comparison()` (lines 363–399)

`simulate_scenario()` runs the full pipeline for one day under one scenario:

1. Build the network (fresh each time to clear previous state).
2. Attach the appropriate LoadShapes (baseline or QP).
3. Run the 48-step daily power flow.
4. Collect all results and compute summary statistics: V_min, V_max across all (customer, interval) pairs, count of voltage violations (outside [0.94, 1.10] p.u.), peak transformer power.

`simulate_day_comparison()` calls this function twice (baseline then QP) for the same day, returning both result dictionaries for side-by-side comparison.

---

## 2.9 Plotting Functions (lines 405–514)

### `plot_voltage_envelope()`

Plots the min and max voltage across all 54 customers at each of the 48 half-hour intervals, as shaded bands. The baseline band (salmon) and QP band (blue) are overlaid with the statutory limits drawn as dashed red lines. The expected result: the QP band should be narrower than the baseline band, particularly during midday (where PV export causes voltage rise in baseline) and evening peak (where high load causes voltage dip).

### `plot_substation_power()`

The transformer secondary power curve over the day. The baseline curve shows negative values during midday (reverse power flow from PV export) and a sharp evening peak. The QP curve should be smoother — batteries absorb excess PV during the day and discharge during the evening, reducing both the reverse flow magnitude and the peak import. This is the network-level manifestation of the "load curve smoothing" objective described in [R15] Section 2.2.

### `plot_voltage_heatmap()`

A 2D colour map with customers on the y-axis and time on the x-axis, coloured by voltage. Red/orange regions indicate voltages near or exceeding the upper limit (voltage rise from PV export); green/yellow regions are near nominal; pale regions indicate low voltage. This visualisation makes it easy to identify which customers at which times are most affected — typically customers at the feeder end (`lv_node_10`) during high PV periods.

### `plot_daily_summary_table()`

Prints a formatted text table comparing key metrics between scenarios: V_min, V_max, violation count, peak transformer power, and total losses.

---

## 2.10 Full-Year Sweep: `run_full_sweep()` and `plot_sweep_results()` (lines 517–610)

`run_full_sweep()` iterates over every day in the dataset (up to 365 or a user-specified cap), runs the day comparison, and collects one summary row per day into a DataFrame. This takes a few minutes on a modern machine (each daily power flow solves in well under a second for a 54-customer feeder).

`plot_sweep_results()` produces four panels showing the daily V_max, V_min, peak transformer power, and violation counts across the full year for both scenarios. Seasonal patterns should be visible: summer days with high PV and high load produce the most extreme voltages, while winter days are generally less stressed.

---

## 2.11 Profile Loading: `load_profiles_from_csv()` (lines 613–642)

Reads the long-format CSV produced by `osqp_daily_v2.save_profiles()` and reconstructs the profiles dictionary. Groups by (customer, date), sorts by interval within each group, and extracts the six data columns into numpy arrays. This is the interface contract between the two files — the CSV acts as a serialisation boundary so they can be run independently.

---

## 2.12 Main Entry Point (lines 645–721)

Parses command-line arguments, loads the profile CSV, builds the customer-to-bus mapping, initialises the DSS engine, and runs either representative days (default: one summer and one winter day) or a full-year sweep (`--full` flag). The default day indices assume the dataset starts on 1 July 2010 ([R17] Section 2), so day index 0 ≈ winter (July) and day index 190 ≈ summer (early January).

---

## Overall Data Flow

```
data.csv (Ausgrid raw)
    │
    ▼
osqp_daily_v2.py
    ├── load_dataset()     ← [R17] Section 2
    ├── clean_dataset()    ← [R17] Section 3
    ├── build_constraints()← [R15] Section 2.1, eqs. (2)-(9)
    ├── build_tariff()     ← [R15] Section 6.1, Fig. 3
    ├── optimise_H()       ← [R15] Section 5, Algorithm 1
    ├── solve_battery()    ← [R15] Lemma 1, eq. (11), via OSQP
    ├── bill_topology1/2() ← [R15] Section 3.3, eqs. (21)/(25)
    ├── simulate_day()     ← [R15] Section 4, eq. (27)
    ├── run_all()          ← parallel across customers
    └── save_profiles()    ── writes profiles/fit_profiles.csv
                                        │
                                        ▼
                              opendss_daily.py
                                  ├── load_profiles_from_csv()
                                  ├── build_network()     ← synthetic Ausgrid LV feeder
                                  ├── attach_loadshapes() ← baseline p⁰ or QP p^C
                                  ├── run_daily()         ← 48-step power flow via OpenDSS
                                  ├── collect_voltages()  ← per-node V(t) in p.u.
                                  ├── collect_tx_power()  ← substation P(t) in kW
                                  └── compare & plot      ← voltage compliance, losses
```
