# Method C — Dual Decomposition (Price Coordination)

**Coordinate through shadow prices.** Only the feeder coupling constraint is
relaxed into the objective with multipliers `lambda >= 0` — directly
interpretable as the **shadow price of feeder headroom** (per interval, per kW).
The Lagrangian separates, so each iteration every household solves its existing
QP independently with a price-shifted linear term, then a master updates the
prices from the aggregate imbalance (VPP_EXTENSION.md §5).

## How it works

With `mu = lambda_up - lambda_lo` (price on grid power `pi`):

```
household i :  min  sum_k h_ik (net_ik - b_ik)^2 - mu' b_i     s.t. local constraints only
master      :  lambda_up <- max(0, lambda_up + alpha_t (sum_i pi_i - D_max))
               lambda_lo <- max(0, lambda_lo + alpha_t (D_min - sum_i pi_i))
               alpha_t = alpha0 / sqrt(t)      (diminishing step)
```

The appeal in this codebase: the household subproblem is **the existing OSQP
workspace with a q-only update** (`HouseholdSolver.solve(q_extra=-mu)`) — the
sparsity pattern and warm start are untouched, and the aggregator only ever
needs `sum_i pi_i` (strong privacy).

Subgradient iterates are not primal-feasible in general, so the script tracks
and reports the **ergodic (running-average) primal**, which converges for
convex problems. The convergence figure also overlays the final `mu` against
the centralised solve's coupling duals (`-y`) — at convergence they coincide,
which is the cleanest correctness check available.

## Tuning the step (measured 2026-08-18)

`alpha0` has to be commensurate with the size of the duals it is trying to
find. On 8-household Ausgrid instances the binding coupling duals are
~68 $/kW-interval (winter, 2 kW/household import cap) and ~66 (summer,
0.05 kW/household export cap):

| alpha0 | winter: avg-primal violation / gap / \|mu\| after 300 it | summer: violation / gap / \|mu\| |
|---|---|---|
| 0.5 (old default) | 1.46 kW / −0.66 % / 36 | 0.39 kW / −3.3 % / 7 |
| 5 | 0.15 kW / −0.12 % / 66 | 0.22 kW / −2.5 % / 43 |
| 10 (default) | 0.08 kW / −0.01 % / 73 | — |
| 20 | 0.00 kW / +0.17 % / 82 | 0.05 kW / −0.7 % / 65 |
| 50 | — | 0.01 kW / −0.15 % / 66 |

The prices converge quickly once the step is right; the **ergodic primal**
is what converges at O(1/√t) (a negative "gap" means the averaged dispatch
still slightly violates the cap — on a strongly binding cap a 0.1 kW
residual is worth several % of the objective). Too large a step makes the
last iterate oscillate (18–27 kW violation at alpha0 10–20 on the winter
case) while the average is fine, which is why `--iterate avg` is the
default. `tests/test_vpp_methods.py` pins mu → −y and the improvement with
iterations on a synthetic ensemble.

## Run

```bash
python vpp/dual_decomposition/dual_decomposition.py --save
python vpp/dual_decomposition/dual_decomposition.py --iters 500 --alpha0 20 --save
```

Outputs: violation-vs-iteration (last iterate and ergodic average), gap vs
centralised, the recovered price profile, `figures/dual_convergence.png`.

## Assessment (from VPP_EXTENSION.md §5)

| | |
|---|---|
| Optimality | Converges to the global optimum (zero duality gap), but O(1/sqrt(t)) — slow |
| Sensitivity | Step size `alpha0` matters; too big oscillates, too small crawls |
| Communication | `T` prices down, `T` aggregates up per iteration |
| Interpretability | Excellent — `lambda_k` *is* a congestion price a DNSP could publish |
| Role here | Prefer Method D unless the price signal itself is the contribution |
