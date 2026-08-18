"""
vpp_registry.py
===============

Uniform interface over the coupling methods in vpp/ so the end-to-end
pipeline (run_vpp_network.py) can invoke any of them interchangeably
(PIPELINE_DESIGN.md Section 3.1).

Each MethodSpec wraps solve functions that already exist in the method
modules -- nothing in those files changes. run() returns a VPPDispatch
that normalises the heterogeneous return shapes and records HOW the
dispatch was obtained (convergence, iterations, timings, extras).

Extras values that are numpy arrays are written to extras.npz by the
exporter; everything else lands in manifest.json.
"""

import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

VPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = VPP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vpp import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.registry")


class PipelineError(RuntimeError):
    """A pipeline stage failed in a way that makes the run meaningless."""


@dataclass(frozen=True)
class EnsembleContext:
    """Everything Stage 1 produced, bundled for the method adapters."""
    households: list          # list[vc.HouseholdDay]
    d_min: np.ndarray         # feeder envelope, kW (T,)
    d_max: np.ndarray
    tariff: np.ndarray        # $/kWh (T,)
    date_iso: str
    mode: str                 # billing topology: "fit" | "net"


@dataclass
class VPPDispatch:
    """Normalised result of any coupling method."""
    B: np.ndarray             # (N, T) battery dispatch, +ve = discharge
    method: str
    converged: bool           # False = iteration cap / allocation failures
    iterations: "int | None"  # None for one-shot solves
    solve_time: float         # seconds spent inside the method
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    name: str                 # canonical name == vpp/ subfolder
    description: str
    aliases: tuple
    add_args: Callable        # add_args(parser) -> None
    run: Callable             # run(ctx: EnsembleContext, args) -> VPPDispatch


def _module(subdir):
    """Import vpp/<subdir>/<subdir>.py (module name equals folder name)."""
    return importlib.import_module(f"vpp.{subdir}.{subdir}")


def _require_solved(status, what):
    if "solved" not in status:
        raise PipelineError(
            f"{what} failed (OSQP status: {status}). A hard feeder "
            "envelope can be infeasible -- retry with --soft where "
            "available, or a looser --scenario / --export-limit.")


# ==========================================================
# METHOD A -- centralised QP
# ==========================================================

def _args_centralised(p):
    p.add_argument("--soft", action="store_true",
                   help="Soften the feeder envelope with penalised slack "
                        "(always feasible)")
    p.add_argument("--penalty", type=float, default=1e3,
                   help="Linear slack penalty in soft mode")


def _run_centralised(ctx, args):
    res = vc.solve_centralised(ctx.households, ctx.d_min, ctx.d_max,
                               soft=args.soft, penalty=args.penalty)
    _require_solved(res.status, "centralised QP")
    extras = {
        "status": res.status,
        "objective": res.objective,
        "n_variables": res.n_variables,
        "n_constraints": res.n_constraints,
        "soft": bool(args.soft),
    }
    converged = True
    if res.y_couple is not None:
        extras["y_couple"] = np.asarray(res.y_couple)
    if res.slack_up is not None:
        slack_kwh = float((res.slack_up + res.slack_lo).sum() * vc.DT)
        extras["slack_up"] = np.asarray(res.slack_up)
        extras["slack_lo"] = np.asarray(res.slack_lo)
        extras["soft_slack_kwh"] = slack_kwh
        # Soft slack > 0 means the envelope could not be met -- a result,
        # not an error, but flag it as non-converged-to-feasible.
        converged = slack_kwh < 1e-3
    return VPPDispatch(B=res.B, method="centralised_qp", converged=converged,
                       iterations=None, solve_time=res.solve_time,
                       extras=extras)


# ==========================================================
# METHOD B -- two-stage DOE allocation
# ==========================================================

def _args_two_stage(p):
    mod = _module("two_stage_doe_allocation")
    p.add_argument("--rule", default="equal", choices=mod.RULES,
                   help="Stage-1 allocation rule for the per-household "
                        "envelopes")


def _run_two_stage(ctx, args):
    mod = _module("two_stage_doe_allocation")
    t0 = time.perf_counter()
    B, curtail_kwh, n_failed = mod.run_rule(
        args.rule, ctx.households, ctx.d_min, ctx.d_max)
    dt = time.perf_counter() - t0
    return VPPDispatch(
        B=B, method="two_stage_doe_allocation",
        converged=(n_failed == 0), iterations=None, solve_time=dt,
        extras={"rule": args.rule, "curtail_kwh": float(curtail_kwh),
                "n_failed_households": int(n_failed)})


# ==========================================================
# METHOD C -- dual decomposition
# ==========================================================

def _args_dual(p):
    p.add_argument("--iters", type=int, default=300,
                   help="Subgradient iterations")
    p.add_argument("--alpha0", type=float, default=10.0,
                   help="Initial step size (alpha_t = alpha0/sqrt(t)); scale "
                        "with the dual magnitude (~50-120 on the Ausgrid "
                        "instances -- 0.5 under-converges, see the method "
                        "README)")
    p.add_argument("--iterate", choices=["avg", "last"], default="avg",
                   help="Report the ergodic average (converges) or the "
                        "last iterate")
    p.add_argument("--tol-kw", type=float, default=0.05,
                   help="Envelope-violation threshold for the converged "
                        "flag (kW)")


def _run_dual(ctx, args):
    mod = _module("dual_decomposition")
    t0 = time.perf_counter()
    out = mod.run_dual(ctx.households, ctx.d_min, ctx.d_max,
                       args.iters, args.alpha0)
    dt = time.perf_counter() - t0
    B = out["B_avg"] if args.iterate == "avg" else out["B_last"]
    final_viol = vc.envelope_violation(
        vc.aggregate_pi(ctx.households, B), ctx.d_min, ctx.d_max)["max_kw"]
    return VPPDispatch(
        B=np.asarray(B), method="dual_decomposition",
        converged=(final_viol <= args.tol_kw),
        iterations=args.iters, solve_time=dt,
        extras={"iterate": args.iterate, "alpha0": args.alpha0,
                "final_violation_kw": float(final_viol),
                "mu": np.asarray(out["mu"]),
                "viol_last_history": np.asarray(out["hist"]["viol_last"]),
                "viol_avg_history": np.asarray(out["hist"]["viol_avg"])})


# ==========================================================
# METHOD D -- sharing ADMM
# ==========================================================

def _args_admm(p):
    p.add_argument("--rho", type=float, default=50.0,
                   help="ADMM penalty parameter")
    p.add_argument("--iters", type=int, default=200,
                   help="Maximum ADMM iterations")
    p.add_argument("--tol-kw", type=float, default=0.05,
                   help="Primal/dual residual stopping tolerance (kW)")


def _run_admm(ctx, args):
    mod = _module("sharing_admm")
    t0 = time.perf_counter()
    B, hist, n_used = mod.run_admm(ctx.households, ctx.d_min, ctx.d_max,
                                   args.rho, args.iters, args.tol_kw)
    dt = time.perf_counter() - t0
    converged = bool(hist["r"] and hist["r"][-1] < args.tol_kw
                     and hist["s"][-1] < args.tol_kw)
    return VPPDispatch(
        B=np.asarray(B), method="sharing_admm",
        converged=converged, iterations=int(n_used), solve_time=dt,
        extras={"rho": args.rho,
                "final_primal_kw": float(hist["r"][-1]) if hist["r"] else None,
                "final_dual_kw": float(hist["s"][-1]) if hist["s"] else None,
                "primal_history": np.asarray(hist["r"]),
                "dual_history": np.asarray(hist["s"]),
                "viol_history": np.asarray(hist["viol"])})


# ==========================================================
# METHOD E -- one-shot price-based control
# ==========================================================

def _args_price(p):
    p.add_argument("--signal", choices=["none", "tou", "shadow"],
                   default="shadow",
                   help="Broadcast price signal. 'shadow' uses the "
                        "centralised solve's coupling duals (approximately "
                        "reproduces the coupled optimum); 'tou' is the "
                        "naive retail-shaped herding demo; 'none' is the "
                        "uncoupled baseline")
    p.add_argument("--gamma", type=float, default=50.0,
                   help="Scale for the TOU-shaped signal")


def _run_price(ctx, args):
    mod = _module("price_based_control")
    y_star = None
    if args.signal == "shadow":
        bench = vc.solve_centralised(ctx.households, ctx.d_min, ctx.d_max)
        _require_solved(bench.status,
                        "centralised benchmark (needed for --signal shadow)")
        y_star = bench.y_couple
    mu = mod.build_signal(args.signal, args.gamma, ctx.tariff, y_star)
    t0 = time.perf_counter()
    B = mod.respond(ctx.households, mu)
    dt = time.perf_counter() - t0
    return VPPDispatch(
        B=np.asarray(B), method="price_based_control",
        converged=True, iterations=None, solve_time=dt,
        extras={"signal": args.signal, "gamma": args.gamma,
                "mu": np.asarray(mu)})


# ==========================================================
# FCAS co-optimisation
# ==========================================================

def _args_fcas(p):
    p.add_argument("--fcas-price", type=float, default=0.1,
                   help="Raise enablement price, $/kW per interval (flat)")
    p.add_argument("--tau", type=float, default=1.0 / 12.0,
                   help="Required sustain duration in hours "
                        "(default 5 min)")


def _run_fcas(ctx, args):
    mod = _module("fcas_cooptimisation")
    price = args.fcas_price * np.ones(vc.T)
    B, R, dt, status = mod.solve_fcas(ctx.households, ctx.d_min, ctx.d_max,
                                      price, args.tau)
    _require_solved(status, "FCAS co-optimised QP")
    R_agg = R.sum(axis=0)
    return VPPDispatch(
        B=np.asarray(B), method="fcas_cooptimisation",
        converged=True, iterations=None, solve_time=dt,
        extras={"status": status, "fcas_price": args.fcas_price,
                "tau_hours": args.tau,
                "fcas_revenue": float(np.sum(price * R_agg)),
                "raise_total_kwh": float(R_agg.sum() * vc.DT),
                "R": np.asarray(R)})


# ==========================================================
# REGISTRY
# ==========================================================

REGISTRY = {
    "centralised_qp": MethodSpec(
        name="centralised_qp",
        description="Method A: centralised monolithic QP (ground truth)",
        aliases=("centralised",),
        add_args=_args_centralised, run=_run_centralised),
    "two_stage_doe_allocation": MethodSpec(
        name="two_stage_doe_allocation",
        description="Method B: two-stage DOE allocation "
                    "(deployed-practice baseline)",
        aliases=("two_stage", "two-stage"),
        add_args=_args_two_stage, run=_run_two_stage),
    "dual_decomposition": MethodSpec(
        name="dual_decomposition",
        description="Method C: dual decomposition via projected subgradient",
        aliases=("dual",),
        add_args=_args_dual, run=_run_dual),
    "sharing_admm": MethodSpec(
        name="sharing_admm",
        description="Method D: consensus/sharing ADMM "
                    "(best decomposition)",
        aliases=("admm",),
        add_args=_args_admm, run=_run_admm),
    "price_based_control": MethodSpec(
        name="price_based_control",
        description="Method E: one-shot price-based indirect control",
        aliases=("price",),
        add_args=_args_price, run=_run_price),
    "fcas_cooptimisation": MethodSpec(
        name="fcas_cooptimisation",
        description="FCAS contingency-raise co-optimisation under DOEs "
                    "(reserve schedule goes to extras, not the network)",
        aliases=("fcas",),
        add_args=_args_fcas, run=_run_fcas),
}
