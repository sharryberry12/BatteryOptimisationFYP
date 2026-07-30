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

## Run

```bash
python vpp/dual_decomposition/dual_decomposition.py --save
python vpp/dual_decomposition/dual_decomposition.py --iters 500 --alpha0 1.0 --save
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
