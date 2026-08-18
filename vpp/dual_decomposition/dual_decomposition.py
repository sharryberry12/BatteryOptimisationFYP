"""
dual_decomposition.py -- Method C: dual decomposition (price coordination)
==========================================================================

Relax only the feeder coupling constraint into the objective with
multipliers lambda >= 0 -- interpretable as a shadow price of feeder
headroom in $/kW per interval. The Lagrangian separates, every household
solves its existing QP with a price-shifted linear term, and a master
projected-subgradient loop updates the prices from the aggregate
imbalance (VPP_EXTENSION.md Section 5).

Per household the only change is the linear term:
    q_i = -2 h_i . net_i - mu,     mu = lambda_up - lambda_lo
(the price mu is on grid power pi; in b-space it enters with sign -mu).
The OSQP workspace, sparsity pattern and warm start are untouched --
each iteration is a q-only update.

Master update (alpha_t = alpha0 / sqrt(t), diminishing):
    lambda_up <- max(0, lambda_up + alpha_t (sum_i pi_i - D_max))
    lambda_lo <- max(0, lambda_lo + alpha_t (D_min - sum_i pi_i))

Subgradient iterates are not feasible in general; the ergodic (running
average) primal iterate is also tracked, which converges for convex
problems and is what gets reported.

Step size: alpha0 must be commensurate with the dual scale (~50-120 on the
Ausgrid instances). Measured 2026-08-18 on 8 households: alpha0=0.5 left
|mu| at 36 of 68 (winter import cap) and 7 of 66 (summer export cap) after
300 iterations; alpha0=10-50 recovers the centralised duals to ~1 %. The
ergodic PRIMAL is the slow part (O(1/sqrt t)); the prices are not.

Run from the repo root, e.g.:
    python vpp/dual_decomposition/dual_decomposition.py --save
    python vpp/dual_decomposition/dual_decomposition.py --iters 500 --alpha0 20
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from vpp import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.dual")


def run_dual(households, d_min, d_max, iters, alpha0):
    """Projected-subgradient dual loop. Returns dict of trajectories."""
    N = len(households)
    solvers = [vc.HouseholdSolver(hh) for hh in households]
    lam_up = np.zeros(vc.T)
    lam_lo = np.zeros(vc.T)
    finite_up = np.isfinite(d_max)
    finite_lo = np.isfinite(d_min)

    B_sum = np.zeros((N, vc.T))
    hist = {"viol_last": [], "viol_avg": [], "obj_avg": [], "mu_norm": []}
    B = np.zeros((N, vc.T))

    for t in range(1, iters + 1):
        mu = lam_up - lam_lo
        for i, s in enumerate(solvers):
            B[i], _status = s.solve(q_extra=-mu)

        agg_pi = vc.aggregate_pi(households, B)
        g_up = np.where(finite_up, agg_pi - d_max, 0.0)
        g_lo = np.where(finite_lo, d_min - agg_pi, 0.0)
        alpha = alpha0 / np.sqrt(t)
        lam_up = np.maximum(0.0, lam_up + alpha * g_up)
        lam_lo = np.maximum(0.0, lam_lo + alpha * g_lo)

        B_sum += B
        B_avg = B_sum / t
        agg_avg = vc.aggregate_pi(households, B_avg)
        hist["viol_last"].append(
            vc.envelope_violation(agg_pi, d_min, d_max)["max_kw"])
        hist["viol_avg"].append(
            vc.envelope_violation(agg_avg, d_min, d_max)["max_kw"])
        hist["obj_avg"].append(vc.objective_surrogate(households, B_avg))
        hist["mu_norm"].append(float(np.abs(mu).max()))

        if t % max(1, iters // 10) == 0:
            logger.info("iter %4d: viol(last)=%.3f kW viol(avg)=%.3f kW "
                        "|mu|_inf=%.2f", t, hist["viol_last"][-1],
                        hist["viol_avg"][-1], hist["mu_norm"][-1])

    return {"B_last": B.copy(), "B_avg": B_sum / iters,
            "mu": lam_up - lam_lo, "hist": hist}


def main():
    parser = vc.standard_argparser(
        "Method C: dual decomposition via projected subgradient")
    parser.add_argument("--iters", type=int, default=300,
                        help="Subgradient iterations")
    parser.add_argument("--alpha0", type=float, default=10.0,
                        help="Initial step size (alpha_t = alpha0/sqrt(t)). "
                             "Scale it with the dual magnitude: on the "
                             "Ausgrid instances the binding coupling duals "
                             "are ~50-120 $/kW-interval and 0.5 leaves the "
                             "prices at half their value after 300 "
                             "iterations; 10-50 converges (README)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the centralised ground-truth solve")
    args = parser.parse_args()

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    logger.info("Day %s, scenario %s, %d iterations, alpha0=%.3f",
                date_iso, args.scenario, args.iters, args.alpha0)

    out = run_dual(households, d_min, d_max, args.iters, args.alpha0)
    B_avg = out["B_avg"]
    obj = vc.objective_surrogate(households, B_avg)
    agg_pi = vc.aggregate_pi(households, B_avg)
    viol = vc.envelope_violation(agg_pi, d_min, d_max)
    savings = vc.savings_vector(households, B_avg, tariff, args.mode)

    obj_star, y_star = None, None
    if not args.no_benchmark:
        res = vc.solve_centralised(households, d_min, d_max)
        if "solved" in res.status:
            obj_star, y_star = res.objective, res.y_couple

    logger.info("=== Method C: dual decomposition (ergodic average) ===")
    logger.info("  surrogate objective : %.4f", obj)
    if obj_star is not None:
        logger.info("  gap vs centralised  : %.3f %%",
                    (obj - obj_star) / obj_star * 100.0)
    logger.info("  envelope violation  : %.4f kW max (%d intervals)",
                viol["max_kw"], viol["n_intervals"])
    logger.info("  savings total/mean  : $%.2f / $%.2f per day",
                savings.sum(), savings.mean())
    logger.info("  price |mu|_inf      : %.3f", np.abs(out["mu"]).max())

    hist = out["hist"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    it = np.arange(1, args.iters + 1)
    axes[0].semilogy(it, np.maximum(hist["viol_last"], 1e-9),
                     label="last iterate", alpha=0.6)
    axes[0].semilogy(it, np.maximum(hist["viol_avg"], 1e-9),
                     label="ergodic average")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("envelope violation (kW)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hours = vc.hours_axis()
    axes[1].plot(hours, out["mu"], label="dual prices mu")
    if y_star is not None:
        axes[1].plot(hours, -y_star, "--",
                     label="centralised duals (-y)")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("shadow price of headroom")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Method C convergence -- N={args.n_households}, "
                 f"{args.scenario}")
    vc.finish_figure(fig, args, "dual_convergence.png", __file__)


if __name__ == "__main__":
    main()
