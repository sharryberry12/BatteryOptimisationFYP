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

## 5. Methods considered but not carried forward

Four further ways of enforcing the §2 coupling were designed and prototyped
against the same stacked problem, then removed from the repository on
2026-09-11 so the VPP layer ships only the two architectures the project
compares: A (what an aggregator with full visibility could achieve) and B
(what a DNSP actually deploys). They are recorded here so the choice is
traceable; the formulations and their tests are in git history before that
date.

- **Method C — dual decomposition (price coordination).** Lagrange-relax the
  coupling rows, broadcast a per-interval price `mu_k`, let each household
  solve its unchanged QP with a linear price term, update `mu` by projected
  subgradient. The prices converge to A's coupling duals (`mu -> -y`), but at
  O(1/sqrt(t)) and with a step size that needed retuning per instance, and
  nothing downstream consumed the prices.
- **Method D — sharing ADMM.** Boyd's sharing form: local proximal QPs with
  `P + rho*I`, a scalar clip onto the envelope, a dual update; tens of
  iterations to engineering tolerance with A's sparsity and warm start intact.
  It is a distributed route to A's optimum, not a different architecture, so
  it answered a question the comparison does not ask.
- **Method E — one-shot price-based indirect control.** Broadcast one price
  shape and let households respond selfishly. With A's duals it reproduces A
  exactly; with a retail TOU shape it herds every battery into the price
  trough. That lesson survives as a one-line warning in §8.
- **FCAS contingency-raise co-optimisation.** A reserve variable per household
  and interval (headroom `b + r <= P_max`, SOC adequacy `SOC >= tau*r`, and the
  DOE-interaction row `sum_i (pi_i - r_i) >= D_min`) to quantify the raise
  capacity static versus dynamic export limits leave. Still a QP, but a
  market-participation study rather than a coupling method, and it needs FCAS
  price data the repository does not carry.
- **Method F — receding-horizon MPC.** Never implemented: a wrapper around any
  coupling method, not an alternative to one.

---

## 6. Comparison summary

| Method | Optimal? | Code delta | Scales to N=1785? | Privacy | Matches industry? | Do it? |
|---|---|---|---|---|---|---|
| **A. Centralised QP** | ✅ exact | Small | Probably, test it | ✗ | ✗ | **1st — ground truth** |
| **B. Two-stage allocation** | ✗ measurable gap | Smallest | ✅ trivially | ✅ | ✅✅ SAPN/AEMO | **2nd — the realistic case** |

---

## 7. Recommended implementation sequence

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
5. **Consumer-facing dashboard**, addressing the prosumer transparency concerns from [12], [13].

---

## 8. Things that will bite

- **Infeasibility is the default in VPP mode.** A feeder envelope tight enough to be interesting
  will make some households infeasible. Decide the policy up front: soft constraints with slack and
  a large linear penalty is usually right, and the slack values then tell you *who* is constrained
  and *when* — which is a result, not an error.
- **Price signals herd.** Any scheme that coordinates through a broadcast price alone (retail TOU
  included) synchronises every battery at the price trough and creates a new peak there — the
  uncoupled QP's 22:00 charging block is exactly this. Explicit envelopes, not prices, are what
  both retained methods enforce.
- **Three-phase reality.** OpenDSS is unbalanced; the QP layer is single-phase. A feeder envelope
  allocated without regard to phase can be satisfied at the QP layer and still cause a phase-specific
  voltage excursion in OpenDSS. Either allocate per-phase or state the limitation explicitly.
- **Profile-to-bus mapping** (`dispatch/FORMULATION.md` §7). 145 profiles onto ~1,785 loads. Whatever the
  replication strategy is, it dominates the network results. Seed it and document it.
