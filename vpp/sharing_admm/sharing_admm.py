"""
sharing_admm.py -- Method D: consensus/sharing ADMM (recommended)
=================================================================

The sharing problem (Boyd et al., Distributed Optimization and
Statistical Learning, Section 7.3) matches the VPP structure exactly:
separable household objectives plus one coupled aggregate constraint.
Each iteration (scaled form, averages):

  1. local, parallel over households:
       pi_i <- argmin f_i(pi_i) + (rho/2)||pi_i - pi_i^t + pibar^t
                                           - zbar^t + u^t||^2
  2. aggregate projection (elementwise clip):
       zbar <- clip(pibar + u, D_min/N, D_max/N)
  3. dual: u <- u + pibar - zbar

Why it fits this codebase unusually well (VPP_EXTENSION.md Section 6):
in b-space the proximal term becomes (rho/2)||(net - v) - b||^2, so the
local subproblem is the existing household QP with P shifted by rho on
the diagonal (set once at setup) and a q-only update per iteration --
same sparsity, same warm start, same parallelisable structure.

Run from the repo root, e.g.:
    python vpp/sharing_admm/sharing_admm.py --save
    python vpp/sharing_admm/sharing_admm.py --rho 5 --iters 100
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.admm")


def run_admm(households, d_min, d_max, rho, iters, tol_kw):
    """Sharing ADMM loop. Returns (B, history, n_iterations_used)."""
    N = len(households)
    solvers = [vc.HouseholdSolver(hh, rho=rho) for hh in households]

    B = np.vstack([hh.b_uncoupled for hh in households])
    Pi = np.vstack([hh.net - b for hh, b in zip(households, B)])
    pibar = Pi.mean(axis=0)
    zbar = np.clip(pibar, d_min / N, d_max / N)
    u = np.zeros(vc.T)

    hist = {"r": [], "s": [], "viol": []}
    for t in range(1, iters + 1):
        for i, (hh, s) in enumerate(zip(households, solvers)):
            v = Pi[i] - pibar + zbar - u
            q_extra = -rho * (hh.net - v)
            B[i], _status = s.solve(q_extra=q_extra)
            Pi[i] = hh.net - B[i]

        pibar = Pi.mean(axis=0)
        zbar_new = np.clip(pibar + u, d_min / N, d_max / N)
        r = N * float(np.abs(pibar - zbar_new).max())      # primal, kW
        s_res = rho * N * float(np.abs(zbar_new - zbar).max())  # dual
        u = u + pibar - zbar_new
        zbar = zbar_new

        agg_pi = N * pibar
        hist["r"].append(r)
        hist["s"].append(s_res)
        hist["viol"].append(
            vc.envelope_violation(agg_pi, d_min, d_max)["max_kw"])

        if t % max(1, iters // 10) == 0:
            logger.info("iter %4d: r=%.4f kW  s=%.4f  viol=%.4f kW",
                        t, r, s_res, hist["viol"][-1])
        if r < tol_kw and s_res < tol_kw:
            logger.info("converged at iteration %d (r=%.4f, s=%.4f)",
                        t, r, s_res)
            return B, hist, t

    logger.warning("ADMM hit the iteration cap (%d) without reaching "
                   "tol=%.3f kW -- if the envelope is infeasible the "
                   "residuals cannot converge; check the scenario or "
                   "tune --rho.", iters, tol_kw)
    return B, hist, iters


def main():
    parser = vc.standard_argparser(
        "Method D: consensus/sharing ADMM (best decomposition)")
    parser.add_argument("--rho", type=float, default=50.0,
                        help="ADMM penalty parameter (outer rho, distinct "
                             "from OSQP's internal rho). Scale with the "
                             "weight magnitude h -- too small vs 2h makes "
                             "the prox term negligible and progress crawls")
    parser.add_argument("--iters", type=int, default=200,
                        help="Maximum ADMM iterations")
    parser.add_argument("--tol-kw", type=float, default=0.05,
                        help="Stop when primal and dual residuals (kW, "
                             "aggregate scale) fall below this")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the centralised ground-truth solve")
    args = parser.parse_args()

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    logger.info("Day %s, scenario %s, rho=%.2f", date_iso, args.scenario,
                args.rho)

    B, hist, n_used = run_admm(households, d_min, d_max,
                               args.rho, args.iters, args.tol_kw)
    obj = vc.objective_surrogate(households, B)
    agg_pi = vc.aggregate_pi(households, B)
    viol = vc.envelope_violation(agg_pi, d_min, d_max)
    savings = vc.savings_vector(households, B, tariff, args.mode)
    vc.validate_ensemble(households, B)

    obj_star = None
    if not args.no_benchmark:
        res = vc.solve_centralised(households, d_min, d_max, soft=True)
        obj_star = res.objective

    logger.info("=== Method D: sharing ADMM ===")
    logger.info("  iterations used     : %d", n_used)
    logger.info("  surrogate objective : %.4f", obj)
    if obj_star is not None:
        logger.info("  gap vs centralised  : %.3f %%",
                    (obj - obj_star) / obj_star * 100.0)
    logger.info("  envelope violation  : %.4f kW max", viol["max_kw"])
    logger.info("  savings total/mean  : $%.2f / $%.2f per day",
                savings.sum(), savings.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    it = np.arange(1, len(hist["r"]) + 1)
    axes[0].semilogy(it, np.maximum(hist["r"], 1e-9), label="primal r")
    axes[0].semilogy(it, np.maximum(hist["s"], 1e-9), label="dual s")
    axes[0].axhline(args.tol_kw, color="k", ls=":", label="tolerance")
    axes[0].set_xlabel("ADMM iteration")
    axes[0].set_ylabel("residual (kW)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hours = vc.hours_axis()
    _B_unc, agg_unc = vc.uncoupled_baseline(households)
    axes[1].plot(hours, agg_unc, color="tab:gray", label="uncoupled")
    axes[1].plot(hours, agg_pi, color="tab:blue", label="ADMM")
    if np.isfinite(d_min).any():
        axes[1].plot(hours, d_min, "r--", label="envelope (export)")
    if np.isfinite(d_max).any():
        axes[1].plot(hours, d_max, "r--", label="envelope (import)")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("feeder-head power (kW)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Method D -- N={args.n_households}, {args.scenario}, "
                 f"rho={args.rho}")
    vc.finish_figure(fig, args, "admm_convergence.png", __file__)


if __name__ == "__main__":
    main()
