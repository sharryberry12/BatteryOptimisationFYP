# CLAUDE.md — QP Battery Scheduler with Dynamic Operating Envelopes

Context for Claude Code sessions on this repository. Derived from the FYP Part A paper
*"A Quadratic-Programming Optimisation Approach for Virtual Power Plants Utilising Dynamic
Operating Envelopes"* (Satheeshkumar, Ah Sang, Dias — Monash ECSE).

Companion doc: `docs/VPP_EXTENSION.md` — multi-household VPP coupling methodologies.

---

## 1. What this project is

A day-ahead **residential battery dispatch scheduler** formulated as a convex QP, solved with
**OSQP**, then **validated on a real LV distribution feeder** in OpenDSS against Australian
standards. It replicates Ratnam et al. (2015) [1] and extends it by adding **Dynamic Operating
Envelope (DOE)** import/export constraints.

Three layers, in dependency order:

```
Layer 1  Scheduler      Ausgrid data -> QP (OSQP) -> dispatch profile (pi, beta)
Layer 2  Validator      dispatch -> OpenDSS loadshapes -> QSTS solve -> compliance checks
Layer 3  Dashboard      Flask web service wrapping 1 and 2, three coordinated views
```

The end goal (Part B) is **aggregate VPP dispatch**: many households coupled through a shared
feeder-level DOE budget, plus FCAS/wholesale market participation.

**Design intent that should not be broken:** the DOE extension is *only* two extra inequality
block-rows. Setting `D_max -> +inf`, `D_min -> -inf` must recover the Ratnam baseline exactly.
Any refactor must preserve that switch.

---

## 2. Notation and sign conventions

**These signs are load-bearing. Getting one wrong silently produces a plausible-looking but wrong
schedule.**

| Symbol | Code name (suggested) | Meaning | Sign convention |
|---|---|---|---|
| `s` | `S` | number of intervals per day | 48 |
| `Δ` | `DELTA_H` | interval length in hours | 0.5 |
| `ℓ` | `load` | residential load, kW avg over interval | ≥ 0 |
| `g` | `pv` | PV generation, kW | ≥ 0 |
| `β` | `beta` | battery power, kW | **positive on DISCHARGE** |
| `π` | `pi_grid` | grid power at PCC, kW | **positive on IMPORT**, negative on export |
| `χ_k` | `soc` | state of charge at time kΔ, kWh | 0 ≤ χ ≤ C |
| `χ₀` | `soc_0` | initial SOC | 0.5·C |
| `C` | `CAPACITY_KWH` | battery energy capacity | 10 kWh |
| `B̅` | `B_MAX` | max discharge power | +5 kW |
| `B̲` | `B_MIN` | max charge power (signed) | −5 kW |
| `D̅` | `doe_import` | DOE max import profile ∈ ℝˢ | **element-wise ≥ 0** |
| `D̲` | `doe_export` | DOE max export profile ∈ ℝˢ | **element-wise ≤ 0** |
| `h_k` | `h` | tariff weight in cost | ≥ 1 |

Horizon: `T = 24 h`, `s·Δ = T`, `s = 48`, `Δ = 30 min`.

Decision variable stacking (**do not reorder — the whole constraint matrix depends on it**):

```
x = [pi; beta]  in R^{2s},  pi = x[0:s],  beta = x[s:2s]
```

---

## 3. The optimisation problem (the contract)

### Physical coupling

```
ell_k = pi_k + g_k + beta_k          for all k in 1..s        (power balance at PCC)
```

Rearranged for the equality block: `pi + beta = ell - g`.

### Constraints

1. **Charge/discharge rate**: `B_MIN * 1 <= beta <= B_MAX * 1`
2. **SOC recursion**: `chi_k = chi_0 - DELTA * sum_{j<=k} beta_j`
   With `T` the lower-triangular all-ones matrix (`T[i,j] = 1 for i >= j`), `sum_{j<=k} beta_j = (T*beta)_k`.
   Capacity bound `0 <= chi_k <= C` becomes:
   ```
   -C_lo <= -T*beta <= C_hi,     C_lo = (chi_0/DELTA)*1,     C_hi = ((C - chi_0)/DELTA)*1
   ```
3. **Daily SOC neutrality**: `1^T beta = 0` (battery returns to initial SOC each day, prevents drift)
4. **DOE**: `D_min <= pi <= D_max`

### Stacked inequality `A1 x <= b1`, with `A1 ∈ R^{6s x 2s}`, `b1 ∈ R^{6s}`

Block-rows 1–4 act on the `beta` block, 5–6 on the `pi` block:

```
[ 0   I ] [pi   ]   [  B_MAX * 1 ]
[ 0  -I ] [beta ]   [ -B_MIN * 1 ]
[ 0  -T ]         <= [  C_hi      ]      <-- SEE WARNING BELOW
[ 0   T ]           [  C_lo      ]
[ I   0 ]           [  D_max     ]
[-I   0 ]           [ -D_min     ]
```

> **⚠ Known sign ambiguity — verify against the code before trusting the paper.**
> The printed `A1` in the paper shows the SOC block-rows as `[0 T]` paired with `C_hi` and
> `[0 -T]` paired with `-C_lo`. That contradicts the inline derivation `-C_lo <= -T*beta <= C_hi`
> immediately above it, which requires `-T*beta <= C_hi` **and** `T*beta <= C_lo`. The version
> written in the block above is the one consistent with the physics (β positive on discharge
> drains SOC). **Action for any agent working here: check what the code actually assembles, run a
> forward SOC simulation on a solved dispatch, and confirm `0 <= chi_k <= C` holds. If the code
> matches the paper's printed matrix, the paper has a typo and needs correcting for Part B.**

### Cost function

```
min  sum_{k=1}^{s}  h_k * pi_k^2
```

`h_k >= 1` from the greedy heuristic in [1, §5]. Larger `h_k` in peak-tariff intervals makes grid
exchange expensive there; the squared term also penalises large reverse power flow (voltage rise).

Embedded as `H_cal = [[H, 0], [0, 0]]` with `H = diag(h_1..h_s)` — zeros on the `beta` block, so
**`P` is diagonal**. That matters for the VPP extension (see §3 of the VPP doc).

> **⚠ This is a surrogate objective, not dollars.** Annual savings are computed *post hoc* under
> the two metering topologies. Do not assume the QP is minimising the reported dollar figure.
> This distinction becomes critical for market participation — see `docs/VPP_EXTENSION.md` §7.

### Equality block `A2 x = b2`

```
A2 = [ 0^T   1^T ]        b2 = [   0    ]
     [ I     I   ]             [ ell - g ]
```

Row 1 = daily neutrality. Rows 2..s+1 = per-sample power balance.

### OSQP canonical form

OSQP solves `min ½ x^T P x + q^T x  s.t.  l <= A_c x <= u`. Two mappings:

- Cost rewritten as `½ x^T (2·H_cal) x`  →  **`P = 2 * H_cal`, `q = 0`**
- Equality encoded as the double inequality `b2 <= A2 x <= b2`

```
A_c = [A1; A2]
l   = [-inf * 1 ; b2]
u   = [b1       ; b2]
```

---

## 4. Why OSQP, and the warm-start invariant

The **sparsity pattern of `P` and `A_c` is identical for every customer and every day of the year.**
Only `l`, `u` (and `q`, if a linear term is ever added) change between days as load, generation and
DOE forecasts change.

Consequence: **factorise the OSQP workspace once, warm-start every subsequent day.**
This is the entire reason OSQP was chosen over an interior-point solver.

- Full year of 48-interval dispatches for one customer: **9–11 s** on a single Intel i7 core with
  warm-starting, vs **≈250 s** reported in [1] for `quadprog`.
- Further parallelised across customers via a CPU pool.

**Invariants any agent must not break:**
- Do not rebuild the OSQP problem inside the per-day loop. Use `update(l=..., u=...)`.
- `P` and `A_c` must be **CSC sparse** matrices, and their structural nonzeros must not change
  between days. If you ever need to change a *value* in `P` or `A_c`, use `update_P` / `update_A`
  with an index vector — not `setup()`.
- Warm-starting across *different customers* is a correctness risk if the first day's solve is
  seeded from an unrelated customer's solution. It's still valid (OSQP converges regardless) but
  makes runtimes non-reproducible. Reset the warm start at customer boundaries or document that
  you don't.
- Always check `res.info.status == 'solved'`. `solved_inaccurate` should be logged, not silently
  accepted.

---

## 5. Data

**Ausgrid "Solar Home Electricity" dataset** [6]:
- 30-minute interval measurements, 300 residential customers, Ausgrid distribution area
- Period: **1 July 2010 – 30 June 2011**
- Channels: gross PV generation (GG), general consumption (GC), controllable load (CL)
- **Gross FiT metering** is required — forecasts need separately recorded PV and load, which
  modern bidirectional net metering cannot provide

**Cleaning protocol** (from [1]): keep only customers with **≥ 5 W of both load and PV generation
on every day of the year** → **145-customer ensemble**.

**Storage**: consolidated into a single **SQLite** database keyed by `(customer_id, timestamp)`,
so a full customer-day record is one SQL query.

**Battery / tariff parameters** ([1, §6.1]):
- `C = 10 kWh`, `chi_0 = 0.5*C`, `B_MAX = -B_MIN = 5 kW`
- Import tariff bands: off-peak **$0.03/kWh**, shoulder **$0.06/kWh**, peak **$0.30/kWh**
- Export compensation: flat **$0.40/kWh**
- Two financial policies: **FiT (metering topology 1)** and **net metering (topology 2)**

> Note the export rate ($0.40) **exceeds every import rate**. That is unusual and has real
> consequences for any attempt to switch to a linear/dollar objective — see VPP doc §7.

---

## 6. Reference results (regression targets)

Any refactor of the scheduler must reproduce these. Treat them as golden values.

| Quantity | This implementation | Ratnam [1] |
|---|---|---|
| Mean annual savings, FiT | **$364.95/yr** | $348/yr |
| Mean annual savings, net metering | **$87.02/yr** | $90/yr |
| Solve time, 1 customer-year | 9–11 s (i7, warm start) | ≈250 s (quadprog) |
| Ensemble size | 145 customers | 145 customers |

Distribution shape is preserved including the **long left tail of customers with negative savings
under net metering**. Differences vs [1] attributed to OSQP (ADMM, default parameters) vs the
quadprog interior-point solver.

---

## 7. OpenDSS network validation

**Feeder**: Elermore Vale 11 kV/LV feeder, Wallsend NSW (CSIRO Representative LV Networks package [7]).
- 1 zone substation
- 23 distribution transformers
- ~1,785 residential and small-commercial loads
- 155 rooftop PV systems
- 40 Redflow ZBM2 batteries

**Porting**: native model is **GridLAB-D**; ported to **OpenDSS** via `dss-python`. OpenDSS chosen
because it has a Python-native interface (feeder sim runs in the same process as the OSQP
scheduler), handles unbalanced LV analysis, and is what Australian DNSPs and CSIRO use for DER
hosting-capacity studies — keeping results comparable to industry work.

**Injection**: each QP dispatch profile `(pi, beta)` exported as a **48-point loadshape** injected at
the corresponding customer bus. Battery → `Storage` element following the dispatched loadshape.
PV → `PVSystem` element following its measured generation profile. Daily **quasi-static time-series
(QSTS)** solve at the same 30-minute resolution as the dispatch.

**Compliance checks** (three, all must pass):

| Check | Standard | Limit |
|---|---|---|
| Bus voltages | **AS IEC 60038:2022** [8] | 230 V nominal, +10%/−6% → **216.2–253 V**; 400 V phase-to-phase same tolerance |
| Transformer apparent-power loading | **AS/NZS 60076.7:2013** [9] | nameplate rating + loading guidance |
| Line currents | **AS/NZS 3008.1.1:2017** [10] | cable current-carrying capacity |

**Result so far**: across 145 replicated customer-days, baseline dispatches (DOE rows suppressed)
produced **no voltage excursions, no transformer overloads, no line overcurrents**. Conclusion:
under historical 2010–11 PV and load conditions, household-level QP optimisation alone does not
threaten network safety on this feeder. The OpenDSS model is therefore ready to act as the
validation system once DOE rows are re-enabled and contemporary penetration levels are stress-tested.

> **Open modelling question in the repo:** the dataset has 145 customers; the feeder has ~1,785
> loads. How profiles are mapped/replicated/sampled onto feeder buses materially changes the
> voltage results. Whatever the current strategy is, it should be explicit, seeded, and documented
> — it is the single biggest lever on the network-compliance conclusions.

---

## 8. Flask dashboard

Wraps scheduler + validator. User selects a customer and a day; the QP is solved, the dispatch is
fed to OpenDSS, and **three coordinated views** render:

1. **Network topology view** — each bus colour-coded by instantaneous voltage (red = over-voltage,
   blue = under-voltage), annotated with substation-to-feeder power flow in real time
2. **Per-bus voltage heatmap** — 24-hour view of voltage propagation under the chosen dispatch
3. **Substation power-flow plot** — baseline vs QP-optimised across the 48-interval day, alongside
   a histogram of voltage violations across the feeder

Part B target: extend into a **consumer-facing** interface showing battery operation, solar exports,
energy usage and projected savings — addressing prosumer transparency concerns raised in Australian
VPP trials [12, 13].

---

## 9. Modelling gaps (deliberate simplifications — know them before extending)

These are all fine for Part A replication fidelity. Several must be revisited before VPP/market work.

| Gap | Impact | Priority for Part B |
|---|---|---|
| **No round-trip efficiency** — no `eta_charge` / `eta_discharge` in the SOC recursion | Overstates achievable savings; `1^T beta = 0` implies lossless cycling | **High** — trivial to add, changes all savings numbers |
| **No degradation cost** | Solver will cycle the battery as hard as constraints allow | Medium |
| **Perfect day-ahead forecasts** of `ell` and `g` assumed | Savings are an upper bound; no robustness | **High** — refs [4],[5] give the RL/uncertainty path |
| **Daily SOC neutrality `1^T beta = 0`** | Blocks multi-day arbitrage; blocks holding energy for FCAS delivery across a day boundary | **High** — relax to a terminal SOC band `chi_lo <= chi_s <= chi_hi` |
| **Surrogate quadratic objective**, not settlement dollars | Cannot price market participation directly | **High** for FCAS/wholesale |
| **Single-phase / balanced treatment at the QP layer** | OpenDSS is unbalanced; DOE allocation is per-phase in reality | Medium — matters for feeder-level DOE budgeting |
| **No inverter apparent-power limit** (`sqrt(P^2+Q^2) <= S_rated`) | Ignores reactive capability and inverter sizing | Medium — becomes a second-order cone if Q is added, breaking pure-QP form |
| **No reactive power / volt-var** | Voltage support left entirely to real power | Medium |

---

## 10. Testable invariants

Any solved dispatch `(pi, beta)` must satisfy these. Build them as assertions in a
`validate_dispatch()` helper and call it in tests, not just at the end of a run.

```python
TOL = 1e-6

# 1. Power balance
assert np.allclose(pi + pv + beta, load, atol=TOL)

# 2. Daily SOC neutrality
assert abs(beta.sum()) < TOL

# 3. SOC stays in band (forward-simulated, NOT read from the constraint matrix)
soc = soc_0 - DELTA_H * np.cumsum(beta)
assert (soc >= -TOL).all() and (soc <= CAPACITY_KWH + TOL).all()

# 4. Rate limits
assert (beta >= B_MIN - TOL).all() and (beta <= B_MAX + TOL).all()

# 5. DOE (when enabled)
assert (pi >= doe_export - TOL).all() and (pi <= doe_import + TOL).all()

# 6. Solver status
assert res.info.status == 'solved'
```

Additional property tests worth having:

- **Baseline recovery**: with `doe_import = +inf`, `doe_export = -inf`, the objective and dispatch
  must match the DOE-rows-removed formulation to solver tolerance.
- **Monotonicity**: relaxing the DOE (widening the envelope) must never increase the optimal cost.
- **Infeasibility handling**: a DOE tight enough that `ell - g` cannot be met within battery limits
  must return `primal infeasible`, and the pipeline must degrade gracefully (curtail / relax to a
  soft constraint with a slack penalty) rather than crash. **This will happen constantly in VPP
  mode** — decide the policy now.
- **Ensemble regression**: mean annual savings within ~$5 of $364.95 (FiT) and $87.02 (net).

---

## 11. Working conventions for this repo

- **Language/stack**: Python. OSQP (QP), `dss-python` (OpenDSS), SQLite (data), Flask (dashboard),
  NumPy/SciPy sparse. Confirm actual versions from `requirements.txt` / lockfile before assuming.
- Build sparse matrices with `scipy.sparse` and convert to **CSC** before handing to OSQP.
- Keep the constraint-assembly function **pure**: `(params, forecasts) -> (P, q, A_c, l, u)`. It
  should be unit-testable without touching the database or OpenDSS.
- Keep the **DOE on/off switch explicit** — a boolean or an envelope-provider object, not commented-out
  code. Part A results depend on being able to reproduce the DOE-suppressed baseline.
- Don't hardcode `s = 48`. Derive it from `T / DELTA_H` so a 15-minute or 5-minute variant (5-min is
  the NEM dispatch interval — relevant for FCAS) is a parameter change, not a rewrite.
- Paper is IEEE two-column format. Figure numbering in the current draft: Fig. 1 topology,
  Fig. 2 annual savings comparison, Fig. 3 dashboard screenshot.
- The paper carries an **AI Use Acknowledgement**: generative AI used for boilerplate code
  generation, refactoring, routine debugging and formatting; all research design, algorithmic
  decisions, analysis and written content by the authors. Keep this true — agent-generated content
  should stay in the boilerplate/refactor category, and any algorithmic suggestion should be
  reviewed and owned by the authors before it lands in the paper.

---

## 12. References (as numbered in the paper)

1. E. L. Ratnam, S. R. Weller, C. M. Kellett, "An optimization-based approach to scheduling residential battery storage with solar PV: assessing customer benefit," *Renew. Energy*, vol. 75, pp. 123–134, Mar. 2015. — **the paper being replicated**
2. B. Liu, J. H. Braslavsky, "Robust dynamic operating envelopes for DER integration in unbalanced distribution networks," *IEEE Trans. Power Syst.*, 2024.
3. SA Power Networks, "Flexible exports for solar PV: trial report," 2022.
4. M. T. Sarker et al., "AI-Driven Optimization Framework for Smart EV Charging Systems...," *WEVJ*, vol. 16, no. 7, p. 385, Jul. 2025.
5. Z. Dou et al., "Innovative energy solutions: Evaluating reinforcement learning algorithms for battery storage optimization in residential settings," *Process Saf. Environ. Prot.*, vol. 191, pp. 2203–2221, Oct. 2024.
6. E. L. Ratnam et al., "Residential load and rooftop PV generation: an Australian distribution network dataset," *Int. J. Sustain. Energy*, vol. 36, no. 8, pp. 787–806, 2017. — **the dataset**
7. F. Geth, T. Brinsmead, "The Representative Low Voltage Networks data and models package – introduction manual," CSIRO, Jan. 2021. — **the feeder model**
8. AS IEC 60038:2022 — Standard voltages
9. AS/NZS 60076.7:2013 — Power transformers, loading guide
10. AS/NZS 3008.1.1:2017 — Cable selection
11. AEMO, "The fairness in dynamic operating envelope objectives report," 2023. — **fairness problem statement**
12. AEMO, "VPP demonstrations: knowledge sharing report," 2021.
13. SA Power Networks, "Advanced VPP grid integration: Final knowledge sharing report," ARENA G00854, 2021.
