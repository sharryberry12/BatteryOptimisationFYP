"""
price_based_control.py -- Method E: market-mediated / indirect control
======================================================================

No explicit envelope and no iteration to convergence: a price signal is
broadcast ONCE and every household responds selfishly. This is the
"transactive energy" family, implemented here as the comparison baseline
that shows WHY explicit envelopes are needed (VPP_EXTENSION.md
Section 7): nothing guarantees the aggregate respects the feeder limits,
and identical batteries responding to an identical signal herd into new
peaks at the price trough.

The quadratic h.(net-b)^2 term is deliberately KEPT and the price enters
as a linear term (q-shift). A purely linear dollar objective under this
project's tariffs (export $0.40 > every import rate) degenerates into
bang-bang arbitrage -- the documented reason Ratnam et al. use the
quadratic surrogate in the first place.

Signals:
  none    zero price          -> the uncoupled baseline
  tou     gamma * TOU tariff  -> naive retail-shaped signal (herding demo)
  shadow  -y from the centralised solve's coupling duals -> the "right"
          congestion prices; a one-shot broadcast of them approximately
          reproduces the centralised optimum because the household
          objectives are strictly convex

Run from the repo root, e.g.:
    python vpp/price_based_control/price_based_control.py --save
    python vpp/price_based_control/price_based_control.py --gamma 200
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.price")

SIGNALS = ["none", "tou", "shadow"]


def build_signal(name, gamma, tariff, y_couple):
    """Price mu on grid power pi (positive discourages import/export)."""
    if name == "none":
        return np.zeros(vc.T)
    if name == "tou":
        return gamma * tariff
    if name == "shadow":
        if y_couple is None:
            raise RuntimeError(
                "shadow signal needs the centralised benchmark duals; "
                "run without --no-benchmark and with a feasible envelope")
        return -np.asarray(y_couple)
    raise ValueError(f"unknown signal {name!r}")


def respond(households, mu):
    """One-shot selfish response of every household to price mu."""
    B = np.zeros((len(households), vc.T))
    for i, hh in enumerate(households):
        solver = vc.HouseholdSolver(hh)
        B[i], _status = solver.solve(q_extra=-mu)  # mu on pi -> -mu on b
    return B


def main():
    parser = vc.standard_argparser(
        "Method E: one-shot price-based indirect control (baseline)")
    parser.add_argument("--signals", default=",".join(SIGNALS),
                        help=f"Comma-separated signals from {SIGNALS}")
    parser.add_argument("--gamma", type=float, default=50.0,
                        help="Scale for the TOU-shaped price signal")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the centralised solve (disables the "
                             "'shadow' signal)")
    args = parser.parse_args()
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    logger.info("Day %s, scenario %s, signals %s",
                date_iso, args.scenario, signals)

    obj_star, y_star = None, None
    if not args.no_benchmark:
        res = vc.solve_centralised(households, d_min, d_max)
        if "solved" in res.status:
            obj_star, y_star = res.objective, res.y_couple
        else:
            logger.warning("benchmark not solved (%s); 'shadow' signal "
                           "unavailable", res.status)

    hours = vc.hours_axis()
    fig, ax = plt.subplots(figsize=(10, 5))
    logger.info("=== Method E: one-shot price response ===")
    logger.info("  %-8s %10s %8s %9s %9s %10s",
                "signal", "objective", "gap%", "viol kW", "viol kWh",
                "peak exp kW")
    for name in signals:
        try:
            mu = build_signal(name, args.gamma, tariff, y_star)
        except RuntimeError as exc:
            logger.warning("skipping %s: %s", name, exc)
            continue
        B = respond(households, mu)
        obj = vc.objective_surrogate(households, B)
        agg_pi = vc.aggregate_pi(households, B)
        viol = vc.envelope_violation(agg_pi, d_min, d_max)
        gap = (obj - obj_star) / obj_star * 100.0 \
            if obj_star is not None else np.nan
        logger.info("  %-8s %10.3f %8.2f %9.3f %9.3f %10.2f",
                    name, obj, gap, viol["max_kw"], viol["energy_kwh"],
                    -agg_pi.min())
        ax.plot(hours, agg_pi, label=f"signal: {name}")

    if np.isfinite(d_min).any():
        ax.plot(hours, d_min, "r--", label="feeder envelope (not enforced)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("feeder-head power (kW, +import)")
    ax.set_title(f"Method E -- open-loop price response, "
                 f"N={args.n_households}")
    ax.legend()
    vc.finish_figure(fig, args, "price_response.png", __file__)


if __name__ == "__main__":
    main()
