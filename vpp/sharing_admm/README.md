# Method D — Consensus / Sharing ADMM ⭐

**The recommended decomposition** (VPP_EXTENSION.md §6). The VPP problem is a
textbook *sharing problem* (Boyd et al., §7.3): separable household objectives
plus one coupled aggregate constraint. ADMM alternates parallel local solves, a
trivial aggregate projection, and a dual update — and converges in tens of
iterations rather than the hundreds a subgradient method needs.

## How it works (scaled form, averages)

```
1. local (parallel over N):
     pi_i <- argmin f_i(pi_i) + (rho/2) || pi_i - pi_i^t + pibar^t - zbar^t + u^t ||^2
2. aggregate projection (elementwise clip):
     zbar <- clip(pibar + u, D_min/N, D_max/N)
3. dual:
     u <- u + pibar - zbar
```

Why it fits this codebase unusually well: in b-space the proximal term becomes
`(rho/2)||(net - v) - b||^2`, so the local subproblem is the **existing
household QP with the P diagonal shifted by rho** — set once at workspace
construction (`HouseholdSolver(hh, rho=rho)`) — and a **q-only update per
iteration**. Same solver, same sparsity, same warm start, and step 1 is
embarrassingly parallel (currently sequential; the CPU pool from
`osqp_daily.py` is the obvious upgrade).

Stopping: primal residual `r = N*max|pibar - zbar|` and dual residual
`s = rho*N*max|zbar - zbar_prev|`, both below `--tol-kw` (interpretable in kW
at the feeder head). Both are logged and plotted — residual curves are the
standard ADMM diagnostic figure.

Note: OSQP is itself an ADMM solver, so this is ADMM-over-ADMM — fine and
common. The known speed-up of loosening the inner tolerance in early outer
iterations is documented in VPP_EXTENSION.md §6 but not implemented (fixed
`eps=1e-6` keeps the solver API surface identical to the rest of the repo).

## Run

```bash
python vpp/sharing_admm/sharing_admm.py --save
python vpp/sharing_admm/sharing_admm.py --rho 5 --iters 100 --save
```

Outputs: iterations to convergence, gap vs centralised, envelope violation,
savings, `outputs/figures/vpp/sharing_admm/admm_convergence.png` (residuals + aggregate profile).

## Tuning

- `--rho` is the **outer** ADMM penalty, deliberately named distinctly from
  OSQP's internal rho (VPP_EXTENSION.md §12). Scale it against the local
  curvature `2h` — the heuristic pushes `h` to ~10³ in peak intervals, so
  ρ ≈ 50–200 is the useful range when the envelope binds (verified: ρ=100
  converges in ~100 iterations to 0.1% of the centralised optimum, while
  ρ=2 makes the proximal term negligible and progress crawls).
- If residuals plateau, first check the envelope is feasible at all
  (`centralised_qp.py` hard mode) — ADMM cannot converge to a nonexistent
  feasible point.

## Assessment (from VPP_EXTENSION.md §6)

| | |
|---|---|
| Optimality | Converges to the global optimum |
| Convergence | Robust; typically tens of iterations for engineering tolerance |
| Privacy | Only the aggregate leaves the households |
| Role here | Best decomposition; use when centralised hits scale/privacy limits |
