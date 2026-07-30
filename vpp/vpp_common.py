"""
vpp_common.py
=============

Shared infrastructure for the VPP coupling methods under vpp/.

Everything here reuses the single-household pipeline from osqp_daily.py
(data loading, cleaning, tariff, billing, weight heuristic, constraint
blocks) and adds only what multi-household coupling needs:

- ensemble assembly for a single day (HouseholdDay records)
- feeder-level envelope scenarios (D_min, D_max in kW at the feeder head)
- the coupled ground-truth solve (Method A) that every other method
  benchmarks against
- persistent per-household OSQP workspaces for the iterative methods
- dispatch validation (paper_context.md Section 10 invariants)
- savings and fairness metrics

Formulation note (matches the code, not the paper's stacked [pi; beta]
form): the decision variable is b in R^T per household (battery power,
positive on discharge). Grid power is pi = net - b with net = load - pv,
so the feeder coupling  D_min <= sum_i pi_i <= D_max  becomes

    agg_net - D_max  <=  sum_i b_i  <=  agg_net - D_min

i.e. the coupling rows of any stacked problem are simply [I I ... I].
Because P = 2*diag(h) with h >= 1, the objective is strictly convex in b
and every method below shares the same unique coupled optimum.
"""

import argparse
import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import osqp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import osqp_daily as base  # noqa: E402  (root-level module, path set above)

logger = logging.getLogger("vpp")

T = base.T
DT = base.DT
P_MAX = base.P_MAX
E_MAX_DEFAULT = base.E_MAX_DEFAULT
SOC_INIT_FRAC = 0.5  # must match base.build_constraints default

DATA_CSV = REPO_ROOT / "data.csv"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

OSQP_SETTINGS = dict(
    verbose=False, eps_abs=1e-6, eps_rel=1e-6, polish=True, warm_start=True,
)


# ==========================================================
# ENSEMBLE ASSEMBLY
# ==========================================================

@dataclass(frozen=True)
class HouseholdDay:
    """One household's data and uncoupled reference dispatch for one day."""
    name: str            # "cust" or "cust#rep" when profiles are replicated
    customer: int
    date: str            # ISO yyyy-mm-dd
    load: np.ndarray     # kW, length T
    pv: np.ndarray       # kW, length T
    net: np.ndarray      # load - pv, kW
    h: np.ndarray        # heuristic weights (fixed for all coupled solves)
    e_max: float         # kWh
    b_uncoupled: np.ndarray   # envelope-free dispatch (today's behaviour)
    savings_uncoupled: float  # $/day vs no-battery baseline


def load_day_arrays(data_path=DATA_CSV, use_cache=True):
    """
    Load + clean the Ausgrid CSV via osqp_daily and collapse to day arrays.
    The cleaning pass over the full CSV takes ~a minute, so the result is
    pickled into vpp/cache/ keyed on the CSV's mtime and size.

    Pickle safety: the cache is written and read only by this module on
    the local machine (never downloaded / untrusted input). Delete
    vpp/cache/ to force a rebuild.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. The Ausgrid CSV is not "
            "redistributed with the repo -- see README.md.")
    stat = data_path.stat()
    cache_file = CACHE_DIR / f"day_arrays_{stat.st_mtime_ns}_{stat.st_size}.pkl"
    if use_cache and cache_file.is_file():
        logger.info("Loading cached day arrays: %s", cache_file.name)
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    df = base.clean_dataset(base.load_dataset(str(data_path)))
    day_arrays = base.extract_day_arrays(df)

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(day_arrays, fh)
    logger.info("Cached day arrays to %s", cache_file.name)
    return day_arrays


def _to_iso(date_raw):
    """Ausgrid dates are d-MMM-yy strings; normalise to ISO."""
    try:
        ts = pd.to_datetime(date_raw, format="%d-%b-%y")
    except (ValueError, TypeError):
        ts = pd.to_datetime(date_raw, dayfirst=True)
    return ts.strftime("%Y-%m-%d")


def pick_date(day_arrays, date_iso=None):
    """
    Choose the simulation day. If date_iso is given, validate it exists.
    Otherwise pick, among dates covered by every customer, the one with
    the highest ensemble PV total (a high-export summer day -- the
    interesting regime for export envelopes). Deterministic.
    """
    coverage = {}
    pv_total = {}
    for days in day_arrays.values():
        for date_raw, _load, pv in days:
            iso = _to_iso(date_raw)
            coverage[iso] = coverage.get(iso, 0) + 1
            pv_total[iso] = pv_total.get(iso, 0.0) + float(pv.sum())

    if date_iso is not None:
        if date_iso not in coverage:
            raise ValueError(f"No customer has data for {date_iso}")
        return date_iso

    full = max(coverage.values())
    candidates = [d for d, c in coverage.items() if c == full]
    return max(candidates, key=lambda d: pv_total[d])


def assemble_ensemble(day_arrays, n_households, date_iso=None,
                      mode="fit", e_max=E_MAX_DEFAULT):
    """
    Build the household ensemble for one day.

    Weights h_i come from the uncoupled per-household heuristic
    (base.optimise_H, envelope-free). They are FROZEN for every coupled
    method so the coupled objective is identical across methods and
    optimality gaps are well defined. The heuristic's dispatch is kept as
    b_uncoupled -- the "today's behaviour" baseline.

    If n_households exceeds the clean ensemble, profiles are replicated
    cyclically (name gets a '#rep' suffix), mirroring the replication
    already used to spread 145 customers over the feeder loads.
    """
    if n_households < 1:
        raise ValueError("n_households must be >= 1")
    date_iso = pick_date(day_arrays, date_iso)
    tariff = base.build_tariff()

    pool = []
    for cust in sorted(day_arrays):
        for date_raw, load, pv in day_arrays[cust]:
            if _to_iso(date_raw) == date_iso:
                pool.append((cust, load, pv))
                break
    if not pool:
        raise ValueError(f"No customers cover {date_iso}")

    households = []
    for i in range(n_households):
        cust, load, pv = pool[i % len(pool)]
        rep = i // len(pool)
        name = str(cust) if rep == 0 else f"{cust}#{rep}"
        h, b_unc, sav = base.optimise_H(load, pv, tariff, e_max, mode)
        households.append(HouseholdDay(
            name=name, customer=cust, date=date_iso,
            load=load, pv=pv, net=load - pv,
            h=h, e_max=e_max,
            b_uncoupled=b_unc, savings_uncoupled=sav,
        ))
    logger.info("Ensemble: %d households on %s (%d unique customers)",
                n_households, date_iso, min(n_households, len(pool)))
    return households, date_iso, tariff


# ==========================================================
# FEEDER-LEVEL ENVELOPE SCENARIOS
# ==========================================================

def feeder_envelope(scenario, n_households,
                    export_limit_kw=1.5, import_limit_kw=np.inf):
    """
    Feeder-head envelope (D_min, D_max) in kW, sign convention as in the
    papers: D_min <= 0 caps aggregate export, D_max >= 0 caps import.
    export_limit_kw / import_limit_kw are PER-HOUSEHOLD bases, scaled by N.

    Scenarios:
      none          no constraint
      static        flat export cap (SAPN-style fixed limit)
      tight_tou     TOU-shaped: with export_limit_kw=1.5 this reproduces
                    the per-household 'tight' envelope of
                    osqp_daily_with_DOE.py (1.5 / 0.9 / 0.45 kW) times N
      dynamic_solar midday-relaxed dynamic envelope (roughly the same
                    daily average as 'static' but tracking solar hosting
                    capacity -- generous 10:00-17:00, tight in the evening)
    """
    ones = np.ones(T)
    if scenario == "none":
        return -np.inf * ones, np.inf * ones

    L = export_limit_kw * n_households
    d_max = (import_limit_kw * n_households) * ones \
        if np.isfinite(import_limit_kw) else np.inf * ones

    if scenario == "static":
        frac = ones.copy()
    elif scenario == "tight_tou":
        frac = np.empty(T)
        frac[0:14] = 1.0   # 00:00-07:00 off-peak
        frac[44:48] = 1.0  # 22:00-24:00
        frac[14:28] = 0.6  # 07:00-14:00 shoulder
        frac[40:44] = 0.6  # 20:00-22:00
        frac[28:40] = 0.3  # 14:00-20:00 peak
    elif scenario == "dynamic_solar":
        frac = np.full(T, 0.3)
        frac[14:20] = 1.2  # 07:00-10:00 morning ramp
        frac[20:34] = 2.2  # 10:00-17:00 solar soak
        frac[34:40] = 1.2  # 17:00-20:00 evening ramp
    else:
        raise ValueError(f"Unknown envelope scenario: {scenario!r}")

    return -L * frac, d_max


# ==========================================================
# PER-HOUSEHOLD LOCAL SUBPROBLEM (persistent workspace)
# ==========================================================

class HouseholdSolver:
    """
    Persistent OSQP workspace for one household's local QP in b-space.

    Objective: sum_k h_k (net_k - b_k)^2 plus whatever linear/proximal
    terms a method injects per-iteration through solve(q_extra=...) and
    the rho shift on the P diagonal. The constraint matrix (including
    optional per-household DOE identity rows) is fixed at setup,
    honouring the warm-start invariant: only q ever changes.
    """

    def __init__(self, hh, rho=0.0, d_min=None, d_max=None, settings=None):
        self.hh = hh
        A, l, u = base.build_constraints(hh.e_max)
        self.doe_relax_kw = np.zeros(T)
        self.import_relax_kw = np.zeros(T)
        self.d_min_eff = None
        self.d_max_eff = None
        if d_min is not None or d_max is not None:
            dmin = -np.inf * np.ones(T) if d_min is None else np.asarray(d_min)
            dmax = np.inf * np.ones(T) if d_max is None else np.asarray(d_max)
            l_doe = hh.net - dmax          # b >= net - D_max  (import cap)
            u_doe = hh.net - dmin          # b <= net - D_min  (export cap)
            # An envelope that demands more export than the battery can
            # physically provide (u_doe < -P_MAX) would be infeasible; the
            # physical meaning is PV curtailment. Relax to the battery
            # limit and record the shortfall so callers can report it.
            self.doe_relax_kw = np.maximum(-P_MAX - u_doe, 0.0)
            self.import_relax_kw = np.maximum(l_doe - P_MAX, 0.0)
            u_doe = np.maximum(u_doe, -P_MAX)
            l_doe = np.minimum(l_doe, P_MAX)
            # effective (post-relaxation) envelope in pi space, so callers
            # can validate dispatches against what was actually enforced
            self.d_min_eff = hh.net - u_doe
            self.d_max_eff = hh.net - l_doe
            A = sp.vstack([A, sp.eye(T, format="csc")]).tocsc()
            l = np.concatenate([l, l_doe])
            u = np.concatenate([u, u_doe])

        self.q0 = -2.0 * hh.h * hh.net
        P = sp.diags(2.0 * hh.h + rho, format="csc")
        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=self.q0.copy(), A=A, l=l, u=u,
                        **(settings or OSQP_SETTINGS))

    def solve(self, q_extra=None):
        """Solve with q = q0 + q_extra. Returns (b, status_string)."""
        q = self.q0 if q_extra is None else self.q0 + q_extra
        self.prob.update(q=np.asarray(q, dtype=np.float64))
        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            logger.warning("household %s: OSQP status %s",
                           self.hh.name, res.info.status)
            return np.zeros(T), res.info.status
        return res.x.copy(), res.info.status


# ==========================================================
# METHOD A -- CENTRALISED GROUND TRUTH (shared benchmark)
# ==========================================================

@dataclass
class CentralisedResult:
    B: np.ndarray            # (N, T) battery dispatch
    status: str
    solve_time: float        # seconds, solve only
    objective: float         # surrogate objective sum_i sum_k h (net-b)^2
    y_couple: np.ndarray     # duals of the coupling rows (hard mode), else None
    slack_up: np.ndarray     # import-side slack (soft mode), else None
    slack_lo: np.ndarray     # export-side slack (soft mode), else None
    n_variables: int
    n_constraints: int


def solve_centralised(households, d_min, d_max, soft=False, penalty=1e3,
                      settings=None):
    """
    Method A: one stacked OSQP problem over all households.

    Hard mode enforces the feeder envelope exactly; if the envelope is
    infeasible OSQP reports primal infeasibility. Soft mode adds
    slack >= 0 on each side of the coupling with a linear penalty, so the
    solve always succeeds and the slack values identify when and by how
    much the envelope is violated (a result, not an error).
    """
    N = len(households)
    agg_net = np.sum([hh.net for hh in households], axis=0)

    blocks, ls, us = [], [], []
    for hh in households:
        A_i, l_i, u_i = base.build_constraints(hh.e_max)
        blocks.append(A_i)
        ls.append(l_i)
        us.append(u_i)
    A_local = sp.block_diag(blocks, format="csc")
    l_local = np.concatenate(ls)
    u_local = np.concatenate(us)
    couple = sp.hstack([sp.eye(T, format="csc")] * N, format="csc")

    P_diag = np.concatenate([2.0 * hh.h for hh in households])
    q = np.concatenate([-2.0 * hh.h * hh.net for hh in households])

    if not soft:
        A = sp.vstack([A_local, couple], format="csc")
        l = np.concatenate([l_local, agg_net - d_max])
        u = np.concatenate([u_local, agg_net - d_min])
    else:
        # x = [b_1..b_N, s_up, s_lo]; sum(b) + s_up >= agg_net - d_max
        #                             sum(b) - s_lo <= agg_net - d_min
        n_loc = A_local.shape[0]
        Z_loc = sp.csc_matrix((n_loc, 2 * T))
        I_T = sp.eye(T, format="csc")
        Z_T = sp.csc_matrix((T, T))
        A = sp.vstack([
            sp.hstack([A_local, Z_loc]),
            sp.hstack([couple, I_T, Z_T]),
            sp.hstack([couple, Z_T, -I_T]),
            sp.hstack([sp.csc_matrix((2 * T, N * T)), sp.eye(2 * T)]),
        ], format="csc")
        l = np.concatenate([l_local, agg_net - d_max,
                            -np.inf * np.ones(T), np.zeros(2 * T)])
        u = np.concatenate([u_local, np.inf * np.ones(T),
                            agg_net - d_min, np.inf * np.ones(2 * T)])
        P_diag = np.concatenate([P_diag, np.zeros(2 * T)])
        q = np.concatenate([q, penalty * np.ones(2 * T)])

    P = sp.diags(P_diag, format="csc")
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, **(settings or OSQP_SETTINGS))
    t0 = time.perf_counter()
    res = prob.solve()
    solve_time = time.perf_counter() - t0

    if res.info.status_val not in (1, 2):
        logger.warning("centralised OSQP status: %s", res.info.status)
        return CentralisedResult(
            B=np.zeros((N, T)), status=res.info.status,
            solve_time=solve_time, objective=np.nan,
            y_couple=None, slack_up=None, slack_lo=None,
            n_variables=A.shape[1], n_constraints=A.shape[0])

    B = res.x[:N * T].reshape(N, T)
    slack_up = res.x[N * T:N * T + T] if soft else None
    slack_lo = res.x[N * T + T:] if soft else None
    n_loc = A_local.shape[0]
    y_couple = None if soft else np.asarray(res.y[n_loc:n_loc + T])

    return CentralisedResult(
        B=B, status=res.info.status, solve_time=solve_time,
        objective=objective_surrogate(households, B),
        y_couple=y_couple, slack_up=slack_up, slack_lo=slack_lo,
        n_variables=A.shape[1], n_constraints=A.shape[0])


# ==========================================================
# METRICS AND VALIDATION
# ==========================================================

def objective_surrogate(households, B):
    """The coupled objective sum_i sum_k h_ik (net_ik - b_ik)^2."""
    return float(sum(np.sum(hh.h * (hh.net - b) ** 2)
                     for hh, b in zip(households, B)))


def aggregate_pi(households, B):
    """Aggregate feeder-head power sum_i pi_i = sum_i net_i - sum_i b_i."""
    agg_net = np.sum([hh.net for hh in households], axis=0)
    return agg_net - np.sum(B, axis=0)


def envelope_violation(agg_pi, d_min, d_max):
    """Max instantaneous exceedance (kW) and violated energy (kWh)."""
    over = np.maximum(agg_pi - d_max, 0.0)
    under = np.maximum(d_min - agg_pi, 0.0)
    exceed = over + under
    return {
        "max_kw": float(exceed.max(initial=0.0)),
        "energy_kwh": float(exceed.sum() * DT),
        "n_intervals": int((exceed > 1e-6).sum()),
    }


def validate_dispatch(hh, b, tol=1e-4, d_min_hh=None, d_max_hh=None):
    """
    paper_context.md Section 10 invariants for one household. Power
    balance holds by construction (pi := net - b). Returns a list of
    violation strings; empty means the dispatch is valid.

    tol is 1e-4 kW / kWh (0.1 Wh): OSQP's eps=1e-6 bounds residuals of
    the stacked problem, not per-constraint satisfaction, so boundary
    touches (e.g. SOC pinned at e_max) can overshoot 1e-5.
    """
    problems = []
    if abs(float(np.sum(b))) > tol * T:
        problems.append(f"SOC neutrality: |sum b| = {abs(np.sum(b)):.2e}")
    soc = SOC_INIT_FRAC * hh.e_max - DT * np.cumsum(b)
    if soc.min() < -tol or soc.max() > hh.e_max + tol:
        problems.append(
            f"SOC out of [0,{hh.e_max}]: [{soc.min():.4f},{soc.max():.4f}]")
    if np.abs(b).max() > P_MAX + tol:
        problems.append(f"rate limit: max|b| = {np.abs(b).max():.4f}")
    pi = hh.net - b
    if d_max_hh is not None and (pi - np.asarray(d_max_hh)).max() > tol:
        problems.append("household DOE import bound violated")
    if d_min_hh is not None and (np.asarray(d_min_hh) - pi).max() > tol:
        problems.append("household DOE export bound violated")
    return problems


def validate_ensemble(households, B, tol=1e-4):
    """Run validate_dispatch on every household; log and count failures."""
    n_bad = 0
    for hh, b in zip(households, B):
        problems = validate_dispatch(hh, b, tol=tol)
        if problems:
            n_bad += 1
            logger.warning("household %s invalid: %s",
                           hh.name, "; ".join(problems))
    if n_bad == 0:
        logger.info("validate_ensemble: all %d dispatches valid",
                    len(households))
    return n_bad


def day_savings(hh, b, tariff, mode):
    """$ saved this day vs the no-battery baseline (positive = benefit)."""
    no_batt = base.bill(hh.load, hh.pv, np.zeros(T), tariff, mode)
    return float(no_batt - base.bill(hh.load, hh.pv, b, tariff, mode))


def savings_vector(households, B, tariff, mode):
    return np.array([day_savings(hh, b, tariff, mode)
                     for hh, b in zip(households, B)])


def jain_index(x):
    """
    Jain's fairness index (sum x)^2 / (n sum x^2), range 1/n..1. Only
    meaningful for non-negative allocations; negative entries are
    clipped to zero first (a household made worse off contributes as
    zero benefit).
    """
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    denom = len(x) * float(np.sum(x ** 2))
    return float(np.sum(x) ** 2 / denom) if denom > 0 else 1.0


def gini(x):
    """Gini coefficient; nan when mean benefit is not positive."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    mean = x.mean() if n else 0.0
    if n == 0 or mean <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * n * mean))


# ==========================================================
# SHARED CLI / EXPERIMENT PLUMBING
# ==========================================================

def standard_argparser(description):
    """CLI flags shared by every approach script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-households", type=int, default=20,
                   help="Ensemble size N (profiles replicate beyond 145)")
    p.add_argument("--date", default=None,
                   help="ISO day to simulate (default: max-PV common day)")
    p.add_argument("--mode", choices=["fit", "net"], default="fit",
                   help="Billing topology for savings reporting")
    p.add_argument("--e-max", type=float, default=E_MAX_DEFAULT,
                   help="Battery capacity per household (kWh)")
    p.add_argument("--data", default=str(DATA_CSV),
                   help="Path to the Ausgrid CSV")
    p.add_argument("--scenario", default="static",
                   choices=["none", "static", "tight_tou", "dynamic_solar"],
                   help="Feeder envelope scenario")
    p.add_argument("--export-limit", type=float, default=1.5,
                   help="Per-household export base for the envelope (kW)")
    p.add_argument("--import-limit", type=float, default=np.inf,
                   help="Per-household import base for the envelope (kW)")
    p.add_argument("--save", action="store_true",
                   help="Save figures instead of showing them")
    p.add_argument("--output-dir", default=None,
                   help="Figure directory (default: <script dir>/figures)")
    return p


def setup_ensemble(args):
    """Standard experiment setup shared by every approach script."""
    day_arrays = load_day_arrays(args.data)
    households, date_iso, tariff = assemble_ensemble(
        day_arrays, args.n_households, args.date,
        mode=args.mode, e_max=args.e_max)
    d_min, d_max = feeder_envelope(
        args.scenario, args.n_households,
        export_limit_kw=args.export_limit,
        import_limit_kw=args.import_limit)
    return households, date_iso, tariff, d_min, d_max


def uncoupled_baseline(households):
    """(B, agg_pi) for today's envelope-free household behaviour."""
    B = np.vstack([hh.b_uncoupled for hh in households])
    return B, aggregate_pi(households, B)


def finish_figure(fig, args, name, script_file):
    """Save under <script dir>/figures (or --output-dir), else show."""
    import matplotlib.pyplot as plt
    if args.save:
        outdir = Path(args.output_dir) if args.output_dir \
            else Path(script_file).resolve().parent / "figures"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("figure saved: %s", path)
        plt.close(fig)
    else:
        plt.show()


def hours_axis():
    """Hour-of-day x axis for interval index 0..T-1 (interval 1 = 00:30)."""
    return (np.arange(T) + 1) * DT
