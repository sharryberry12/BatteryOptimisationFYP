"""
two_stage_doe_allocation.py -- Method B: two-stage DOE allocation
=================================================================

Stage 1: the DNSP splits the feeder-level envelope into per-household
         envelopes under one of several allocation rules.
Stage 2: every household solves its existing QP independently with its
         allocated envelope -- exactly the deployed-in-Australia
         architecture (SA Power Networks Flexible Exports trial).

The research signal is the pair (efficiency gap vs the centralised
optimum, fairness across households) per allocation rule -- the AEMO
fairness question (VPP_EXTENSION.md Section 4).

Run from the repo root, e.g.:
    python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save
    python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py \
        --rules equal,maxmin --scenario tight_tou
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.two_stage")

RULES = ["equal", "prorata_pv", "prorata_surplus", "maxmin"]


# ==========================================================
# STAGE 1 -- ALLOCATION RULES
# ==========================================================

def water_fill(budget, caps):
    """
    Max-min fair progressive filling: allocate `budget` among agents with
    per-agent caps, equalising allocations until each saturates.
    """
    caps = np.asarray(caps, dtype=float)
    alloc = np.zeros_like(caps)
    remaining = float(budget)
    active = caps > 1e-12
    while remaining > 1e-9 and active.any():
        share = remaining / active.sum()
        add = np.minimum(np.where(active, share, 0.0), caps - alloc)
        alloc += add
        remaining -= add.sum()
        active &= (caps - alloc) > 1e-12
        if add.sum() < 1e-12:
            break
    return alloc


def export_caps(households):
    """Max physical export per household/interval: pi >= net - P_MAX."""
    return np.vstack([np.maximum(vc.P_MAX - hh.net, 0.0)
                      for hh in households])


def allocate(rule, households, d_min, d_max):
    """
    Split (d_min, d_max) into per-household envelopes.
    Returns (d_min_i, d_max_i), each shaped (N, T), with
    sum_i d_min_i >= d_min so feeder compliance holds by construction.
    """
    N = len(households)
    budget = np.where(np.isfinite(d_min), -d_min, np.inf)  # export kW >= 0
    d_max_i = np.tile(d_max / N, (N, 1))  # import side; inf stays inf

    if rule == "equal":
        alloc = np.tile(budget / N, (N, 1))
    elif rule == "prorata_pv":
        # Proxy for system/connection size: day-peak PV (the CSV's
        # Generator Capacity column is not carried through the pipeline).
        w = np.array([max(hh.pv.max(), 1e-9) for hh in households])
        alloc = np.outer(w / w.sum(), budget)
    elif rule == "prorata_surplus":
        # Per-interval forecast surplus (pv - load)+; equal split in
        # intervals where nobody has surplus.
        w = np.vstack([np.maximum(-hh.net, 0.0) for hh in households])
        col = w.sum(axis=0)
        share = np.where(col > 1e-9, w / np.maximum(col, 1e-9), 1.0 / N)
        # inf budget (unconstrained interval): share*inf gives 0*inf=nan
        alloc = np.where(np.isinf(budget), np.inf,
                         share * np.where(np.isfinite(budget), budget, 0.0))
    elif rule == "maxmin":
        caps = export_caps(households)
        alloc = np.zeros((N, vc.T))
        for k in range(vc.T):
            if np.isfinite(budget[k]):
                alloc[:, k] = water_fill(budget[k], caps[:, k])
            else:
                alloc[:, k] = np.inf
    else:
        raise ValueError(f"unknown rule {rule!r}")

    return -alloc, d_max_i


# ==========================================================
# STAGE 2 -- INDEPENDENT HOUSEHOLD SOLVES
# ==========================================================

def run_rule(rule, households, d_min, d_max):
    """Allocate, then solve every household independently."""
    d_min_i, d_max_i = allocate(rule, households, d_min, d_max)
    B = np.zeros((len(households), vc.T))
    curtail_kwh = 0.0
    n_failed = 0
    for i, hh in enumerate(households):
        solver = vc.HouseholdSolver(hh, d_min=d_min_i[i], d_max=d_max_i[i])
        curtail_kwh += float((solver.doe_relax_kw
                              + solver.import_relax_kw).sum() * vc.DT)
        b, status = solver.solve()
        if "solved" not in status:
            n_failed += 1  # zeros returned; shows up in violation metrics
        elif vc.validate_dispatch(hh, b, d_min_hh=solver.d_min_eff,
                                  d_max_hh=solver.d_max_eff):
            n_failed += 1
            logger.warning("household %s: dispatch violates its allocated "
                           "envelope", hh.name)
        B[i] = b
    return B, curtail_kwh, n_failed


def main():
    parser = vc.standard_argparser(
        "Method B: two-stage DOE allocation (deployed-practice baseline)")
    parser.add_argument("--rules", default=",".join(RULES),
                        help=f"Comma-separated allocation rules "
                             f"from {RULES}")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the centralised ground-truth solve")
    args = parser.parse_args()
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    logger.info("Day %s, scenario %s, rules %s",
                date_iso, args.scenario, rules)

    obj_star = None
    if not args.no_benchmark:
        res = vc.solve_centralised(households, d_min, d_max, soft=True)
        obj_star = res.objective
        logger.info("Centralised benchmark objective: %.4f (%.3f s)",
                    obj_star, res.solve_time)

    hours = vc.hours_axis()
    fig_p, ax_p = plt.subplots(figsize=(10, 5))
    rows = []
    for rule in rules:
        B, curtail_kwh, n_failed = run_rule(rule, households, d_min, d_max)
        obj = vc.objective_surrogate(households, B)
        agg_pi = vc.aggregate_pi(households, B)
        viol = vc.envelope_violation(agg_pi, d_min, d_max)
        savings = vc.savings_vector(households, B, tariff, args.mode)
        gap = (obj - obj_star) / obj_star * 100.0 \
            if obj_star is not None else np.nan
        rows.append((rule, obj, gap, savings.sum(),
                     vc.jain_index(savings), vc.gini(savings),
                     viol["max_kw"], curtail_kwh, n_failed))
        ax_p.plot(hours, agg_pi, label=rule)

    logger.info("=== Method B: allocation rule comparison ===")
    logger.info("  %-16s %10s %8s %9s %6s %6s %8s %8s %6s",
                "rule", "objective", "gap%", "save$/d",
                "Jain", "Gini", "viol kW", "curt kWh", "fail")
    for r in rows:
        logger.info("  %-16s %10.3f %8.2f %9.2f %6.3f %6.3f %8.3f %8.2f %6d",
                    *r)

    if np.isfinite(d_min).any():
        ax_p.plot(hours, d_min, "r--", label="feeder envelope")
    ax_p.axhline(0.0, color="k", lw=0.5)
    ax_p.set_xlabel("hour of day")
    ax_p.set_ylabel("feeder-head power (kW, +import)")
    ax_p.set_title(f"Method B aggregate profiles -- N={args.n_households}")
    ax_p.legend()
    vc.finish_figure(fig_p, args, "two_stage_aggregate.png", __file__)

    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    gaps = [r[2] for r in rows]
    jains = [r[4] for r in rows]
    ax_b.bar(x - 0.2, gaps, 0.4, label="efficiency gap %", color="tab:red")
    ax_b2 = ax_b.twinx()
    ax_b2.bar(x + 0.2, jains, 0.4, label="Jain fairness", color="tab:blue")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([r[0] for r in rows])
    ax_b.set_ylabel("gap vs centralised (%)")
    ax_b2.set_ylabel("Jain index")
    ax_b2.set_ylim(0, 1.05)
    ax_b.set_title("Efficiency-fairness trade-off by allocation rule")
    vc.finish_figure(fig_b, args, "two_stage_tradeoff.png", __file__)


if __name__ == "__main__":
    main()
