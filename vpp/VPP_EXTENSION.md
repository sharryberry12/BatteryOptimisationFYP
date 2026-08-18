# VPP Extension — Coupling Methodologies

Design context for extending the single-household QP scheduler to **coordinated multi-household
Virtual Power Plant dispatch**. Read `dispatch/FORMULATION.md` first for notation and the base formulation.

Everything below preserves the core property that makes the existing code fast: **the per-household
QP keeps the same sparsity pattern, so OSQP stays warm-startable.** Any method that destroys that
should be treated as a last resort.

---

## 1. Why the current formulation doesn't extend for free

Right now every household solves in isolation. Household `i` solves

```
min  sum_k h_k * pi_{i,k}^2
s.t. A1_i x_i <= b1_i        (battery + its own DOE)
     A2_i x_i  = b2_i        (power balance + neutrality)
```

`N` independent QPs, embarrassingly parallel. That's exactly what the CPU pool does today.

A VPP introduces a **shared feeder-level resource**. The DNSP no longer issues an independent
envelope per connection — it has a feeder-level headroom budget that must be split:

```
sum_{i=1}^{N} pi_{i,k}  <=  D_feeder_max_k       for all k
sum_{i=1}^{N} pi_{i,k}  >=  D_feeder_min_k       for all k
```

This single coupling constraint destroys separability. `N` independent problems become one problem
with `2sN` variables. Everything in this document is a strategy for handling that coupling.

The good news: **the coupled problem is still convex** (quadratic objective, linear constraints,
continuous variables). Strong duality holds, so decomposition methods converge to the true global
optimum — no duality gap to worry about. That is not true if you later add binaries (on/off
commitment, mode switching), so avoid integer variables if you can.

---

## 2. The coupled problem, stated once

Let `X = [x_1; x_2; ...; x_N] ∈ R^{2sN}`. Let `E_i ∈ R^{s x 2s}` select the `pi` block of `x_i`
(i.e. `E_i x_i = pi_i`, so `E = [I_s  0_s]`).

```
min_X   sum_i  x_i^T H_cal_i x_i
s.t.    A1_i x_i <= b1_i                       for all i     (local)
        A2_i x_i  = b2_i                       for all i     (local)
        sum_i E_i x_i <= D_feeder_max                        (COUPLING)
       -sum_i E_i x_i <= -D_feeder_min                       (COUPLING)
```

Only `2s` coupling rows (`s` upper, `s` lower). Everything else is block-diagonal. That structure
is what every method below exploits.

---

## 3. Method A — Centralised monolithic QP

**Stack everything into one OSQP problem.**

```
P     = blkdiag(2*H_cal_1, ..., 2*H_cal_N)                      # diagonal, 2sN x 2sN
A_c   = [ blkdiag(A1_i) ; blkdiag(A2_i) ; [E_1 ... E_N] ]
```

### Assessment

| | |
|---|---|
| **Optimality** | Exact global optimum, single solve |
| **Complexity** | ~6sN + 2sN + s rows. For N=1000, s=48: ~400k rows, 96k variables |
| **Sparsity** | Excellent — block diagonal plus `2s` dense-ish coupling rows |
| **Warm start** | Still works; pattern is day-invariant |
| **Privacy** | None — aggregator sees every household's load and PV |
| **Robustness** | Single point of failure; one infeasible household kills the whole solve |

### Verdict

**Do this first.** It is the least code, it gives you the ground-truth optimum that every
decomposition method must be benchmarked against, and OSQP genuinely handles sparse problems of
this size. Only move to decomposition once you have measured that it is too slow or once privacy /
architecture arguments demand it.

Practical notes:
- Build blocks with `scipy.sparse.block_diag` and `scipy.sparse.vstack`; convert to CSC once.
- Test scaling empirically: N = 10, 50, 145, 500, 1785. Plot solve time. Expect a knee somewhere.
- Per-household infeasibility must be caught *before* the stack. Pre-screen each household's local
  feasibility, or add slack variables with large linear penalties so the master problem is always
  feasible and tells you *who* is being violated.

---

## 4. Method B — Two-stage DOE allocation (decoupled)

**Stage 1**: DNSP splits the feeder budget into per-household envelopes.
**Stage 2**: every household solves its existing QP independently, unchanged.

```
Stage 1:  D_feeder_max_k  ->  {D_max_{i,k}}   such that   sum_i D_max_{i,k} <= D_feeder_max_k
Stage 2:  N independent QPs, exactly the code that exists today
```

This is what **SA Power Networks actually did** in the Flexible Exports trial [3] and it is the
architecture Australian DNSPs are heading toward. It is also **the closest to zero new code**: your
scheduler already accepts `D_max`, `D_min` per household. Only the allocator is new.

### Allocation rules (this is where the research contribution lives)

| Rule | Formula | Notes |
|---|---|---|
| **Equal split** | `D_i = D_feeder / N` | Trivial. Wastes capacity on households that can't use it |
| **Pro-rata connection capacity** | `D_i ∝ S_rated_i` | Regulatorily defensible, ignores actual need |
| **Pro-rata forecast net demand** | `D_i ∝ (ell_i - g_i)` | Efficient, but rewards big consumers |
| **Max-min fair (progressive filling)** | maximise the smallest allocation, then the next | Classic networking result; implementable as iterative LP |
| **Proportional fair** | `max sum_i log(u_i)` | **Not a QP** — needs a conic solver (Clarabel/ECOS/SCS) or an iterative water-filling that stays LP |
| **Outcome-fair** | equalise *savings*, not *envelope* | The AEMO [11] point — allocating equal envelopes does not produce equal benefit |
| **Price/auction-based** | households bid for headroom | Efficient, but a market design problem in itself |

### Assessment

| | |
|---|---|
| **Optimality** | **Suboptimal** — allocation is made before households reveal what they'd do with it. Gap vs Method A is the headline number to measure |
| **Complexity** | Same as today. Fully parallel |
| **Privacy** | Good — DNSP needs only aggregate/forecast info |
| **Robustness** | Excellent — one household failing affects nobody else |
| **Realism** | **Highest** — matches deployed Australian practice |

### Verdict

**Do this second.** The comparison "centralised optimum vs allocated envelopes, across allocation
rules" is a clean, publishable, and directly policy-relevant result. Report both efficiency loss
(total savings gap) **and** fairness (Jain's index or Gini coefficient on per-household annual
savings) — the AEMO fairness question [11] is explicitly about the tension between those two.

---

## 5. Method C — Dual decomposition (price coordination)

Relax **only** the coupling constraint into the objective with a multiplier `lambda_k >= 0`
(interpretable as a **shadow price of feeder headroom in interval k**, in $/kW).

Lagrangian separates:

```
L(X, lambda) = sum_i [ x_i^T H_cal_i x_i + lambda^T E_i x_i ]  -  lambda^T D_feeder
```

Each household solves, **independently and in parallel**:

```
min  x_i^T H_cal_i x_i + lambda^T E_i x_i    s.t.  local constraints only
```

which in OSQP terms is **your existing problem with `q = E^T lambda` instead of `q = 0`**.

Master update (projected subgradient / dual ascent):

```
lambda^{t+1} = max(0,  lambda^t + alpha_t * (sum_i pi_i^t - D_feeder))
```

### Mapping onto the existing code — this is the appealing part

- `P` unchanged. `A_c` unchanged. **Sparsity pattern untouched.**
- Only `q` changes each iteration → `osqp.update(q=...)`, the cheapest possible update.
- Warm start from the previous iteration's solution → later iterations solve in a handful of ADMM steps.
- The aggregator only ever sees `sum_i pi_i` — **strong privacy**.

### Assessment

| | |
|---|---|
| **Optimality** | Converges to global optimum (convex, zero duality gap) |
| **Convergence** | **Slow and step-size sensitive.** Subgradient methods are O(1/sqrt(t)). Needs a diminishing step rule or Nesterov acceleration |
| **Communication** | `s` numbers down (prices), `s` numbers up (aggregate) per iteration |
| **Interpretability** | Excellent — `lambda_k` *is* a congestion price, directly meaningful to a DNSP |

### Verdict

Attractive economically (the multipliers are the product, not just a numerical device) but
practically fiddly. **If you want decomposition, prefer Method D unless the price interpretation is
itself the research contribution.**

---

## 6. Method D — Consensus / sharing ADMM ⭐ recommended decomposition

The **sharing problem** (Boyd et al., *Distributed Optimization and Statistical Learning*, §7.3) is
almost exactly your structure: separable objectives, one coupled sum constraint.

Introduce `z_i` copies of each household's grid profile and split. Each ADMM iteration:

```
# 1. Local (parallel across N):
x_i^{t+1} = argmin  x_i^T H_cal_i x_i
                  + (rho/2) * || E_i x_i - E_i x_i^t + pi_bar^t - z^t + u^t ||_2^2
            s.t. local constraints

# 2. Aggregate (cheap, one projection onto the feeder envelope):
z^{t+1}  = projection of (pi_bar^{t+1} + u^t) onto {z : N*z <= D_feeder}

# 3. Dual:
u^{t+1}  = u^t + pi_bar^{t+1} - z^{t+1}
```

where `pi_bar = (1/N) sum_i pi_i`.

### Why this fits the existing code unusually well

The local subproblem's quadratic term becomes `2*H_cal_i + rho * E_i^T E_i`.

`H_cal` is **diagonal** (zeros on the beta block). `E^T E` is **also diagonal** (identity on the pi
block, zero on beta). So:

> **`P` stays diagonal. Its sparsity pattern does not change — only the values on the `pi` block
> shift by `rho`.** Use `osqp.update_P(Px=...)` once when `rho` changes, and `osqp.update(q=...)`
> each iteration. Same factorisation strategy, same warm-start advantage.

Also worth noting: OSQP is itself an ADMM solver, so you are running ADMM-over-ADMM. This is fine
and common, but **do not run the inner OSQP to tight tolerance in early outer iterations** — use a
loose `eps_abs`/`eps_rel` early and tighten as the outer residuals fall. This is usually a 3–5x
speedup and is the single most impactful implementation detail here.

### Assessment

| | |
|---|---|
| **Optimality** | Converges to global optimum |
| **Convergence** | **Much more robust than dual decomposition.** Typically tens of iterations for engineering tolerance. Less step-size sensitive (`rho` matters but is forgiving; adaptive `rho` helps) |
| **Communication** | `s` numbers each way per iteration; only the *aggregate* leaves the households |
| **Parallelism** | Step 1 is fully parallel — reuse the existing CPU pool |
| **Warm start** | Across days *and* across ADMM iterations. Both help |

### Verdict

**The best decomposition choice for this project.** Same solver, same sparsity, same warm-start
story, same parallel infrastructure. The extension is genuinely ~150 lines of coordination logic
around code that already exists.

Stopping criteria: primal residual `||pi_bar - z||` and dual residual `rho*||z^{t+1} - z^t||`, both
below tolerance. Log both — a plot of residuals vs iteration is a good figure for Part B.

---

## 7. Method E — Market-mediated / indirect control

No explicit envelope allocation and no iteration to convergence. The DNSP or aggregator **broadcasts
a price signal** and households respond selfishly. This is the "transactive energy" family.

The objective must change from the surrogate `sum h_k pi_k^2` to something denominated in dollars.

### ⚠ The tariff problem you will hit immediately

Actual settlement is piecewise-linear:

```
cost = DELTA * sum_k [ p_import_k * max(pi_k, 0)  -  p_export_k * max(-pi_k, 0) ]
```

The standard trick is to split `pi = pi_plus - pi_minus` with both `>= 0` and solve an LP. **That
relaxation is only tight when `p_import_k >= p_export_k` at every interval** — otherwise the
optimum wants simultaneous import and export.

In this project's tariff set: export is a **flat $0.40/kWh** while import is at most **$0.30/kWh**
(peak) and as low as **$0.03/kWh** (off-peak). **Export compensation exceeds import price in every
interval.** So:

- The LP relaxation is **not tight**. You would need binaries (MILP) or a complementarity
  constraint to prevent simultaneous import/export.
- Even with a single net `pi`, a purely linear objective becomes a pure arbitrage problem: charge
  from the grid at $0.03 off-peak, export at $0.40. It will slam every constraint boundary.
- **This is almost certainly why [1] uses the `h_k * pi_k^2` surrogate in the first place.** The
  quadratic term regularises the solution away from bang-bang arbitrage and simultaneously
  penalises reverse power flow.

**Action:** document this explicitly in Part B. It is a real and defensible modelling justification,
not a limitation to hide. If you want dollar-denominated results, either (a) keep the quadratic as a
regulariser and add a linear price term (`q != 0`, still a QP — easy), or (b) use contemporary
tariffs where export < import and the LP relaxation is tight.

### Assessment

| | |
|---|---|
| **Optimality** | No constraint guarantee — households may collectively violate the feeder envelope |
| **Risk** | **Synchronisation / herding.** Every battery responding to the same price signal creates a new peak at the price trough. Well documented in demand-response literature |
| **Verdict** | Interesting as a comparison baseline showing *why* explicit envelopes are needed. Not a primary method |

---

## 8. Method F — Receding-horizon MPC (orthogonal, composable)

This is not an alternative to A–E; it wraps around any of them.

Instead of one 48-interval day-ahead solve, re-solve over a **shrinking or rolling horizon** as
actual load and PV are realised:

```
at interval k:  solve over [k, k+H], apply only interval k, advance, repeat
```

- Directly addresses the **perfect forecast** assumption flagged in `dispatch/FORMULATION.md` §9
- 48 solves per day instead of 1 — but each is smaller, and warm-starting makes this cheap
- The `1^T beta = 0` neutrality constraint must be reworked into a **terminal SOC target or band**;
  otherwise the shrinking horizon makes it progressively infeasible
- Uncertainty variants:
  - **Stochastic**: scenario-based, `M` scenarios → `M` copies of the QP with non-anticipativity
    constraints. Stays a QP but scales by `M`
  - **Robust**: enforce constraints for worst case in an uncertainty set. Box uncertainty stays a
    QP; ellipsoidal becomes a **second-order cone program** — no longer OSQP-solvable, needs
    Clarabel/ECOS
  - **Learning-augmented**: refs [4], [5]. RL supplies the forecast or a terminal value function;
    the QP still enforces feasibility. This is the **safest hybrid** — the optimiser guarantees
    constraint satisfaction, the learner only improves the objective. Explicitly flagged as a
    future direction in the paper

---

## 9. FCAS and wholesale market participation

Stated future work in the paper: VPPs respond far faster than conventional generators [12], static
export limits have been shown to constrain FCAS delivery [12], and SAPN demonstrated dynamic limits
can **more than double available export capacity at key intervals** [13].

### Formulation — stays a QP

Add non-negative reserve enablement variables `r_k^raise`, `r_k^lower` for each service.

**Headroom constraints** (the battery must be able to move by the enabled amount):
```
beta_k + r_k^raise  <=  B_MAX          # discharge headroom for raise
beta_k - r_k^lower  >=  B_MIN          # charge headroom for lower
```

**Energy adequacy** (must sustain the response for the service duration `tau`):
```
chi_k          >=  r_k^raise * tau     # enough stored energy to deliver raise
C - chi_k      >=  r_k^lower * tau     # enough empty capacity to absorb lower
```
`tau` = 6 s / 60 s / 5 min for contingency services; regulation is continuous and needs a different
(and more conservative) energy treatment.

**⚠ The DOE–FCAS interaction — this is the interesting result:**
```
pi_k - r_k^raise  >=  D_min_k          # raise response must fit inside the export envelope
pi_k + r_k^lower  <=  D_max_k          # lower response must fit inside the import envelope
```

This is precisely the mechanism by which static export limits throttle FCAS capability [12], and
precisely what dynamic envelopes relieve [13]. **Quantifying enabled FCAS capacity under static vs
dynamic envelopes on the Elermore Vale feeder is the single strongest result available to this
project.** It is directly comparable to the SAPN finding and uses infrastructure you already have.

**Objective** becomes energy cost minus reserve revenue:
```
min  sum_k h_k pi_k^2  -  sum_k ( p_k^raise * r_k^raise + p_k^lower * r_k^lower )
```
Quadratic + linear, all constraints linear → **still a convex QP**. `P` unchanged in pattern (the
new variables have zero quadratic cost), `q != 0`. OSQP handles it.

### Aggregation

An individual 10 kWh household battery is far below any NEM registration threshold. FCAS enablement
is a **VPP-level** quantity: `R_k = sum_i r_{i,k}`. So this is another coupling constraint and slots
straight into Method A or D. Bid the aggregate; disaggregate the dispatch instruction back to
households.

### Practical notes
- NEM dispatch interval is **5 minutes**, not 30. Don't hardcode `s = 48` (see `dispatch/FORMULATION.md` §11).
- 8 contingency FCAS markets plus 2 regulation markets. Start with **contingency raise only** — one
  service, one price series, clean story.
- FCAS prices are extremely spiky. Expected-value optimisation over historical prices will be
  dominated by a handful of intervals; report medians alongside means.

---

## 10. Comparison summary

| Method | Optimal? | Code delta | Scales to N=1785? | Privacy | Matches industry? | Do it? |
|---|---|---|---|---|---|---|
| **A. Centralised QP** | ✅ exact | Small | Probably, test it | ✗ | ✗ | **1st — ground truth** |
| **B. Two-stage allocation** | ✗ measurable gap | Smallest | ✅ trivially | ✅ | ✅✅ SAPN/AEMO | **2nd — the realistic case** |
| **C. Dual decomposition** | ✅ (slow) | Medium | ✅ | ✅ | Partial | Only if prices are the point |
| **D. Sharing ADMM** | ✅ | Medium | ✅ | ✅ | Emerging | **3rd — best decomposition** |
| **E. Price-based indirect** | ✗ no guarantee | Medium | ✅ | ✅ | Research | Baseline / counterexample |
| **F. MPC + uncertainty** | n/a (wrapper) | Medium–large | inherits | inherits | ✅ | Layer on later |
| **FCAS co-optimisation** | ✅ (still QP) | Medium | inherits | inherits | ✅✅ | **High-value result** |

---

## 11. Recommended implementation sequence

1. **Close the modelling gaps first** (`dispatch/FORMULATION.md` §9). Add round-trip efficiency and relax
   `1^T beta = 0` to a terminal SOC band. Both are small edits and both change every downstream
   number — do them before generating results you'd have to regenerate.
2. **Method A on a small ensemble** (N = 10, 50, 145). Establishes the ground-truth optimum and the
   scaling curve. Validate every solution against the `validate_dispatch()` invariants.
3. **Method B with 3–4 allocation rules.** Measure efficiency gap vs A *and* fairness (Jain / Gini
   on per-household annual savings). This directly answers the AEMO [11] fairness question and is
   the most policy-legible output.
4. **OpenDSS validation with DOE rows enabled** and contemporary PV penetration. The paper's current
   "no violations" result is under 2010–11 penetration — the interesting result is where it breaks.
5. **Method D** if and only if A hits a wall at feeder scale, or if the distributed architecture is
   itself part of the contribution.
6. **FCAS layer**, static vs dynamic envelope comparison. This is the strongest single result and it
   plugs into whichever coupling method is working by then.
7. **Consumer-facing dashboard**, addressing the prosumer transparency concerns from [12], [13].

---

## 12. Things that will bite

- **Infeasibility is the default in VPP mode.** A feeder envelope tight enough to be interesting
  will make some households infeasible. Decide the policy up front: soft constraints with slack and
  a large linear penalty is usually right, and the slack values then tell you *who* is constrained
  and *when* — which is a result, not an error.
- **Warm-start staleness.** When the coupling term changes between ADMM iterations, a stale warm
  start can slow convergence rather than help. Measure it; don't assume.
- **`rho` tuning.** OSQP's adaptive `rho` and the outer ADMM's `rho` are different parameters.
  Naming them distinctly in code will save an afternoon.
- **Three-phase reality.** OpenDSS is unbalanced; the QP layer is single-phase. A feeder envelope
  allocated without regard to phase can be satisfied at the QP layer and still cause a phase-specific
  voltage excursion in OpenDSS. Either allocate per-phase or state the limitation explicitly.
- **Profile-to-bus mapping** (`dispatch/FORMULATION.md` §7). 145 profiles onto ~1,785 loads. Whatever the
  replication strategy is, it dominates the network results. Seed it and document it.
- **Reproducibility.** Once ADMM and CPU pooling are combined, floating-point non-determinism from
  reduction ordering can make results non-bitwise-reproducible. Fix the reduction order or accept
  and document a tolerance.
