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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from paths import FIGURES  # noqa: E402
from vpp import vpp_common as vc  # noqa: E402
from vpp import vpp_export as vexport  # noqa: E402

logger = logging.getLogger("vpp.two_stage")

RULES = ["equal", "prorata_pv", "prorata_surplus", "maxmin"]


# ==========================================================
# STAGE 1 -- ALLOCATION RULES
# ==========================================================

def floor_fill(budget, floors):
    """
    Max-min fair allocation with LOWER bounds (the import-side mirror of
    water_fill): every agent receives the same allowance lambda except
    those whose floor exceeds it, who receive their floor; lambda is set
    so the allocations sum to budget. If the floors alone exceed the
    budget every agent gets its floor and the remainder is the fleet's
    unavoidable shortfall (reported by the solves). An unbounded budget
    passes through unbounded.
    """
    floors = np.asarray(floors, dtype=float)
    if not np.isfinite(budget):
        return np.full_like(floors, np.inf)
    if floors.sum() >= budget:
        return floors.copy()
    pinned = np.zeros(len(floors), dtype=bool)
    while True:
        free = ~pinned
        lam = (budget - floors[pinned].sum()) / free.sum()
        newly = free & (floors > lam + 1e-12)
        if not newly.any():
            return np.where(pinned, floors, lam)
        pinned |= newly


def import_floors(households):
    """Import a household cannot avoid even at full discharge:
    (net - P_MAX)+ per interval, shape (N, T)."""
    return np.vstack([np.maximum(hh.net - vc.P_MAX, 0.0)
                      for hh in households])


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
    Split (d_min, d_max) into per-household envelopes. Returns
    (d_min_i, d_max_i), each shaped (N, T), with sum_i d_min_i >= d_min
    and sum_i d_max_i <= d_max so feeder compliance holds by construction.

    Export side (budget -d_min): equal / prorata_pv / prorata_surplus /
    maxmin against each household's physical export cap.
    Import side (budget d_max): equal for `equal` and `prorata_pv` (PV
    size says nothing about import need); proportional to forecast
    need (net)+ for `prorata_surplus`; max-min fair with floors for
    `maxmin` -- every household gets the same allowance except those
    whose unavoidable import (net - P_MAX)+ exceeds it, which get their
    floor. Unbounded intervals stay unbounded.
    """
    N = len(households)
    budget = np.where(np.isfinite(d_min), -d_min, np.inf)  # export kW >= 0
    d_max = np.asarray(d_max, dtype=float)

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

    if rule == "maxmin":
        floors = import_floors(households)
        d_max_i = np.empty((N, vc.T))
        for k in range(vc.T):
            d_max_i[:, k] = floor_fill(d_max[k], floors[:, k])
    elif rule == "prorata_surplus":
        need = np.vstack([np.maximum(hh.net, 0.0) for hh in households])
        col = need.sum(axis=0)
        share = np.where(col > 1e-9, need / np.maximum(col, 1e-9), 1.0 / N)
        d_max_i = np.where(np.isinf(d_max), np.inf,
                           share * np.where(np.isfinite(d_max), d_max, 0.0))
    else:
        d_max_i = np.tile(d_max / N, (N, 1))     # inf stays inf

    return -alloc, d_max_i


# ==========================================================
# STAGE 2 -- INDEPENDENT HOUSEHOLD SOLVES
# ==========================================================

def load_envelope(run_dir, n_households, date_iso):
    """
    Feeder envelope (d_min, d_max) from a run directory's manifest --
    a static_vs_doe_replay run (d_max_doe_kw) or a pipeline run
    (d_max_kw). The manifest's ensemble must match the one being solved.
    """
    manifest = vexport.load_manifest(run_dir)
    ens = manifest.get("ensemble", {})
    if (ens.get("n_households") != n_households
            or str(ens.get("date")) != date_iso):
        raise ValueError(
            f"{Path(run_dir).name}: manifest ensemble "
            f"(N={ens.get('n_households')}, {ens.get('date')}) does not "
            f"match this run (N={n_households}, {date_iso})")
    env = manifest["envelope"]
    key = "d_max_doe_kw" if "d_max_doe_kw" in env else "d_max_kw"
    return (vexport.envelope_array(env["d_min_kw"]),
            vexport.envelope_array(env[key]))


def run_rule(rule, households, d_min, d_max, soft=False):
    """Allocate, then solve every household independently.

    soft=True gives each household the slack formulation of
    vc.HouseholdSolver(soft=True): an energy-infeasible slice is met
    best-effort and its excess reported, instead of the zero-dispatch
    fallback of the hard solve (which counts in n_failed).

    Returns (B, curtail_kw, shortfall_kw, n_failed). curtail_kw and
    shortfall_kw are per-interval AGGREGATE arrays (T,) of the realised
    out-of-envelope power of the returned dispatch against the original
    (pre-relaxation) allocated envelopes: export excess is PV a deployed
    inverter would spill to stay inside its DOE, import excess is cap
    shortfall the battery cannot cover. They are distinct physical
    quantities -- do not sum them into one "curtailment" number.
    """
    d_min_i, d_max_i = allocate(rule, households, d_min, d_max)
    B = np.zeros((len(households), vc.T))
    curtail_kw = np.zeros(vc.T)
    shortfall_kw = np.zeros(vc.T)
    n_failed = 0
    for i, hh in enumerate(households):
        solver = vc.HouseholdSolver(hh, d_min=d_min_i[i], d_max=d_max_i[i],
                                    soft=soft)
        b, status = solver.solve()
        if "solved" not in status:
            n_failed += 1  # zeros returned; shows up in violation metrics
        elif vc.validate_dispatch(
                hh, b,
                # in soft mode the envelope is met up to the reported
                # slack, so only the local invariants are checked here
                d_min_hh=None if soft else solver.d_min_eff,
                d_max_hh=None if soft else solver.d_max_eff):
            n_failed += 1
            logger.warning("household %s: dispatch violates its allocated "
                           "envelope", hh.name)
        B[i] = b
        pi = hh.net - b
        curtail_kw += np.maximum(d_min_i[i] - pi, 0.0)    # export excess
        shortfall_kw += np.maximum(pi - d_max_i[i], 0.0)  # import excess
    return B, curtail_kw, shortfall_kw, n_failed


def main():
    parser = vc.standard_argparser(
        "Method B: two-stage DOE allocation (deployed-practice baseline)")
    parser.add_argument("--rules", default=",".join(RULES),
                        help=f"Comma-separated allocation rules "
                             f"from {RULES}")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the centralised ground-truth solve")
    parser.add_argument("--soft", action="store_true",
                        help="Soft per-household envelopes: infeasible "
                             "slices report shortfall/curtailment "
                             "instead of falling back to no dispatch")
    parser.add_argument("--envelope-from", default=None,
                        help="Run directory whose manifest.json supplies "
                             "the feeder envelope (d_min_kw and "
                             "d_max_doe_kw or d_max_kw) instead of "
                             "--scenario; N and date must match")
    args = parser.parse_args()
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]

    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    envelope_src = f"scenario {args.scenario!r}"
    if args.envelope_from:
        d_min, d_max = load_envelope(args.envelope_from,
                                     len(households), date_iso)
        envelope_src = f"manifest in {Path(args.envelope_from).name}"
    logger.info("Day %s, envelope from %s, rules %s, %s household "
                "solves", date_iso, envelope_src, rules,
                "soft" if args.soft else "hard")

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
        B, curtail_kw, shortfall_kw, n_failed = run_rule(
            rule, households, d_min, d_max, soft=args.soft)
        obj = vc.objective_surrogate(households, B)
        agg_pi = vc.aggregate_pi(households, B)
        # deliverable aggregate: a deployed inverter spills the export
        # excess, so credit curtail_kw before measuring the residual
        # feeder violation (import shortfall stays visible -- nothing
        # physical removes it)
        agg_delivered = agg_pi + curtail_kw
        viol = vc.envelope_violation(agg_delivered, d_min, d_max)
        savings = vc.savings_vector(households, B, tariff, args.mode)
        gap = (obj - obj_star) / obj_star * 100.0 \
            if obj_star is not None else np.nan
        rows.append((rule, obj, gap, savings.sum(),
                     vc.jain_index(savings), vc.gini(savings),
                     viol["max_kw"],
                     float(curtail_kw.sum() * vc.DT),
                     float(shortfall_kw.sum() * vc.DT), n_failed))
        ax_p.plot(hours, agg_delivered, label=rule)

    logger.info("=== Method B: allocation rule comparison ===")
    logger.info("  %-16s %10s %8s %9s %6s %6s %8s %8s %9s %6s",
                "rule", "objective", "gap%", "save$/d",
                "Jain", "Gini", "viol kW", "curt kWh", "short kWh", "fail")
    for r in rows:
        logger.info("  %-16s %10.3f %8.2f %9.2f %6.3f %6.3f %8.3f %8.2f "
                    "%9.2f %6d", *r)
    if args.save:
        table = pd.DataFrame(rows, columns=[
            "rule", "objective", "gap_pct", "savings_per_day", "jain",
            "gini", "residual_violation_kw", "curtail_kwh",
            "shortfall_kwh", "n_failed"])
        table.insert(0, "solve", "soft" if args.soft else "hard")
        table.insert(0, "date", date_iso)
        outdir = Path(args.output_dir) if args.output_dir \
            else FIGURES / "vpp" / Path(__file__).resolve().parent.name
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "rule_comparison.csv"
        table.to_csv(csv_path, index=False)
        logger.info("rule table written: %s", csv_path)

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
