"""
centralised_qp.py -- Method A: centralised monolithic QP
========================================================

Stack every household's battery QP into ONE OSQP problem with the feeder
envelope as coupling rows, and solve it exactly. This is the ground-truth
optimum that every other coupling method in vpp/ is benchmarked against
(VPP_EXTENSION.md Section 3).

In the code's reduced b-space formulation the coupling is
    agg_net - D_max  <=  sum_i b_i  <=  agg_net - D_min
so the stacked problem is block-diagonal local constraints plus T
identity coupling rows, and P stays diagonal.

Run from the repo root, e.g.:
    python vpp/centralised_qp/centralised_qp.py --n-households 20 --save
    python vpp/centralised_qp/centralised_qp.py --scaling 10,20,50,100 --save
    python vpp/centralised_qp/centralised_qp.py --scenario tight_tou --soft
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from vpp import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.centralised")


def report(households, res, d_min, d_max, tariff, mode):
    """Log the standard result block and return the summary dict."""
    if "solved" not in res.status:
        logger.error("Solve failed (%s). A hard envelope can be "
                     "infeasible -- retry with --soft.", res.status)
        return None

    agg_pi = vc.aggregate_pi(households, res.B)
    viol = vc.envelope_violation(agg_pi, d_min, d_max)
    savings = vc.savings_vector(households, res.B, tariff, mode)
    vc.validate_ensemble(households, res.B)

    logger.info("=== Method A: centralised QP ===")
    logger.info("  variables/constraints : %d / %d",
                res.n_variables, res.n_constraints)
    logger.info("  solve time            : %.3f s", res.solve_time)
    logger.info("  surrogate objective   : %.4f", res.objective)
    logger.info("  envelope violation    : %.4f kW max, %.4f kWh",
                viol["max_kw"], viol["energy_kwh"])
    if res.slack_up is not None:
        logger.info("  soft slack (imp/exp)  : %.4f / %.4f kWh",
                    res.slack_up.sum() * vc.DT, res.slack_lo.sum() * vc.DT)
    logger.info("  savings total/mean    : $%.2f / $%.2f per day",
                savings.sum(), savings.mean())
    return {"objective": res.objective, "agg_pi": agg_pi,
            "savings": savings, "violation": viol,
            "solve_time": res.solve_time}


def plot_aggregate(households, summary, d_min, d_max, args):
    """Aggregate feeder profile: uncoupled behaviour vs coupled optimum."""
    _B_unc, agg_unc = vc.uncoupled_baseline(households)
    hours = vc.hours_axis()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, agg_unc, label="uncoupled (today)", color="tab:gray")
    ax.plot(hours, summary["agg_pi"], label="centralised optimum",
            color="tab:blue")
    if np.isfinite(d_min).any():
        ax.plot(hours, d_min, "r--", label="feeder envelope")
    if np.isfinite(d_max).any():
        ax.plot(hours, d_max, "r--")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("feeder-head power (kW, +import)")
    ax.set_title(f"Method A -- N={args.n_households}, "
                 f"{args.scenario} envelope")
    ax.legend()
    vc.finish_figure(fig, args, "centralised_aggregate.png", __file__)


def scaling_study(args, sizes):
    """Empirical solve-time scaling curve (VPP_EXTENSION.md Section 3)."""
    day_arrays = vc.load_day_arrays(args.data)
    times, objs = [], []
    for n in sizes:
        households, _date, _tariff = vc.assemble_ensemble(
            day_arrays, n, args.date, mode=args.mode, e_max=args.e_max)
        d_min, d_max = vc.feeder_envelope(
            args.scenario, n, export_limit_kw=args.export_limit,
            import_limit_kw=args.import_limit)
        res = vc.solve_centralised(households, d_min, d_max,
                                   soft=args.soft, penalty=args.penalty)
        times.append(res.solve_time)
        objs.append(res.objective)
        logger.info("N=%4d : %d vars, %.3f s, status %s",
                    n, res.n_variables, res.solve_time, res.status)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, times, "o-")
    ax.set_xlabel("households N")
    ax.set_ylabel("OSQP solve time (s)")
    ax.set_title("Centralised QP scaling (single day)")
    ax.grid(True, alpha=0.3)
    vc.finish_figure(fig, args, "centralised_scaling.png", __file__)


def main():
    parser = vc.standard_argparser(
        "Method A: centralised monolithic VPP QP (ground truth)")
    parser.add_argument("--soft", action="store_true",
                        help="Soften the feeder envelope with penalised "
                             "slack (always feasible)")
    parser.add_argument("--penalty", type=float, default=1e3,
                        help="Linear slack penalty in soft mode")
    parser.add_argument("--scaling", default=None,
                        help="Comma-separated N list for a scaling study, "
                             "e.g. 10,20,50,145")
    args = parser.parse_args()

    if args.scaling:
        scaling_study(args, [int(n) for n in args.scaling.split(",")])
        return

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    logger.info("Day %s, scenario %s", date_iso, args.scenario)

    _B_unc, agg_unc = vc.uncoupled_baseline(households)
    viol_unc = vc.envelope_violation(agg_unc, d_min, d_max)
    logger.info("Uncoupled behaviour violates envelope by %.2f kW max "
                "(%.2f kWh) -- this is why coordination is needed.",
                viol_unc["max_kw"], viol_unc["energy_kwh"])

    res = vc.solve_centralised(households, d_min, d_max,
                               soft=args.soft, penalty=args.penalty)
    summary = report(households, res, d_min, d_max, tariff, args.mode)
    if summary is not None:
        plot_aggregate(households, summary, d_min, d_max, args)


if __name__ == "__main__":
    main()
