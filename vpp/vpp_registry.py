"""
vpp_registry.py
===============

Uniform interface over the two coupling methods in vpp/ -- Method A
(centralised_qp) and Method B (two_stage_doe_allocation) -- so the
end-to-end pipeline (run_vpp_network.py) can invoke either
interchangeably (PIPELINE_DESIGN.md Section 3.1).

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
    B, curtail_kw, shortfall_kw, n_failed = mod.run_rule(
        args.rule, ctx.households, ctx.d_min, ctx.d_max)
    dt = time.perf_counter() - t0
    return VPPDispatch(
        B=B, method="two_stage_doe_allocation",
        converged=(n_failed == 0), iterations=None, solve_time=dt,
        extras={"rule": args.rule,
                "curtail_kwh": float(curtail_kw.sum() * vc.DT),
                "import_shortfall_kwh": float(shortfall_kw.sum() * vc.DT),
                "n_failed_households": int(n_failed)})


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
}
