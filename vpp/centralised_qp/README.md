# Method A — Centralised Monolithic QP

**The ground truth.** Every household's battery QP is stacked into one OSQP
problem with the feeder envelope as coupling rows and solved exactly, once.
Every other method in `vpp/` reports its optimality gap against this solve
(`vpp_common.solve_centralised`, which this script drives).

## How it works

Per household `i` the local problem is unchanged from `osqp_daily.py`
(b-space: rate limits, SOC band via the lower-triangular integrator, daily
neutrality `1'b = 0`). The stacked problem is:

```
min   sum_i sum_k h_ik (net_ik - b_ik)^2          P = blkdiag(2 diag(h_i))  — diagonal
s.t.  local constraints per household              A_local = blkdiag(A_i)
      agg_net - D_max <= sum_i b_i <= agg_net - D_min      [I I ... I]  coupling rows
```

Only `T = 48` coupling rows tie the blocks together; everything else is
block-diagonal, which is exactly the sparsity OSQP exploits.

**Soft mode** (`--soft`): adds slack `s_up, s_lo >= 0` on the coupling with a
linear penalty (default `1e3`). The solve is then always feasible and the slack
values tell you *when* and *by how much* the envelope cannot be met — a result,
not an error (VPP_EXTENSION.md §12). Hard mode reports OSQP primal
infeasibility instead.

## Run

```bash
python vpp/centralised_qp/centralised_qp.py --n-households 20 --save
python vpp/centralised_qp/centralised_qp.py --scenario tight_tou --soft --save
python vpp/centralised_qp/centralised_qp.py --scaling 10,20,50,145 --save   # solve-time knee
```

Outputs: the standard result block (objective, solve time, violation, savings),
per-household invariant validation, and `figures/centralised_aggregate.png`
(uncoupled behaviour vs coupled optimum vs envelope). `--scaling` produces the
empirical solve-time curve recommended in VPP_EXTENSION.md §3.

## Assessment (from VPP_EXTENSION.md §3)

| | |
|---|---|
| Optimality | Exact global optimum, single solve |
| Privacy | None — the aggregator sees every household's load and PV |
| Robustness | Single point of failure; one infeasible household kills the solve (hence `--soft`) |
| Role here | Ground truth + scaling study; do this first |
