"""
osqp_daily_with_DOE.py
======================

Extension of osqp_daily.py (Ratnam et al. 2015) to account for Dynamic 
Operating Envelopes (DOEs) as broadcast by the DNO.

Key modifications from the base script:
--------------------------------------

1. New function: generate_doe_envelope()
   - Creates synthetic DOE bounds for testing (time-varying export limits)
   - In practice, these would be received from the DNO/AMI in real-time

2. Modified function: build_constraints()
   - Now accepts optional doe_min, doe_max vectors
   - Adds DOE constraints as [p_k >= doe_min_k, p_k <= doe_max_k] 
     via the A matrix in OSQP format

3. Updated: solve_battery()
   - Passes DOE bounds through to the solver

4. Extended: simulate_day()
   - Now accepts doe_envelope parameter
   - Returns DOE_slack (tracking if envelope was violated)

5. New metrics:
   - doe_slack_usage (per-day slack violations)
   - doe_compliant (binary: did day respect envelope?)
   - savings_without_doe vs savings_with_doe (impact analysis)

This script maintains backward compatibility with the baseline (doe_envelope=None).

CLI (2026-08-16):  --mode {fit,net}  --scenarios ...  --export-limit kW [kW ...]
                   --no-compare  --data
Outputs outputs/profiles/<mode>_doe_<scenario>[_cap<kW>].csv in the long format the
network scripts read (elermorevale_openDSS.py --profiles ...).

Formulation (2026-08-19): decision vector x = [b | c | s]
    b  battery kW (+ discharge)                    -- osqp_daily's variable
    c  curtailed PV kW, 0 <= c <= pv               -- export-side relief
    s  import shortfall kW, >= 0                   -- import-side slack
grid flow p = load - pv + c - b, and
    minimise  sum_k h_k p_k^2 + PENALTY_KW * sum_k h_k (c_k + s_k)
    s.t.  battery rate / SOC / daily neutrality on b,
          doe_min <= p            (export cap, HARD -- always feasible via c),
          p <= doe_max + s        (import cap, SOFT -- load cannot be shed).
The penalties exceed any flattening benefit, so c and s are used only when
the battery cannot meet the envelope; the bill charges curtailment through
pv - c. Without an envelope c and s are pinned to zero and the result equals
osqp_daily.solve_battery exactly. --import-limit adds a flat import cap to
any scenario (label suffix _imp<kW>). Profiles gain curtail_kw and
import_shortfall_kw columns.

Constraint mechanics (2026-08-16 fix): all rows live in the persistent OSQP
workspace from setup and are switched per day by updating bounds (and P/q).
Before that fix the DOE rows were passed to solver.update(A=...), which osqp
silently ignores, so no DOE result generated earlier actually had the
envelope enforced. Regression tests: tests/test_doe_constraints.py.
"""

import logging
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp

# repo root on sys.path so `paths` imports from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_CSV, OUTPUTS, PROFILES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONSTANTS
# ==========================================================

DT = 0.5            # hours per interval
T = 48              # intervals per day
P_MAX = 5.0         # kW charge/discharge limit
E_MAX_DEFAULT = 10.0  # kWh
FIT_RATE = 0.40     # $/kWh
H_BAR = 1000.0      # cap on heuristic weights
HEURISTIC_MAX_ITERS = 20

# Cleaning thresholds (unchanged from base script)
GC_MAX_THRESHOLD     = 0.006
GG_CAT1_MAX          = 0.06
GG_CAT2_MAX          = 0.101
GG_CAT2_SUM          = 0.65
GG_CAT3_EARLY_SUM    = 0.04
GG_CAT3_EARLY_LIMIT  = 10


# ==========================================================
# NEW: DOE GENERATION AND UTILITY FUNCTIONS
# ==========================================================

def generate_doe_envelope(scenario="conservative", base_export_limit=3.0,
                          base_import_limit=np.inf):
    """
    Generate a time-varying DOE envelope for testing.

    Parameters:
    -----------
    scenario : str   (shapes the EXPORT side)
        'none'         -> no export constraint (p_min = -inf)
        'conservative' -> moderate export limits (80% of baseline)
        'tight'        -> strict limits (30% of baseline), worst during peak
        'rolling'      -> realistic day-ahead forecast + ±10% uncertainty

    base_export_limit : float
        Maximum export (negative p) the feeder can accept (kW)
        E.g., 3.0 means we won't send more than 3 kW back to grid

    base_import_limit : float
        Flat cap on IMPORT (positive p), kW; inf = unconstrained. This is
        the import-side envelope: it targets the synchronised off-peak
        charging that the price-only QP produces (NETWORK_AWARE_DISPATCH.md
        section 5, item 1). Load cannot be shed, so an import cap the
        battery cannot honour is met best-effort and the shortfall is
        reported (see solve_battery).

    Returns:
    --------
    doe_min, doe_max : np.ndarray of shape (T,)
        Lower and upper bounds on p_k for each interval k
        doe_min_k is the most negative p can go (export limit)
        doe_max_k is the most positive p can go (import limit)
    """
    if scenario == "none":
        # Baseline: no DOE constraint
        doe_min = -np.inf * np.ones(T)
        doe_max = np.inf * np.ones(T)
    
    elif scenario == "conservative":
        # Moderate export constraint: 80% of baseline across all hours
        doe_min = -0.8 * base_export_limit * np.ones(T)
        doe_max = np.inf * np.ones(T)
    
    elif scenario == "tight":
        # Tight constraint: varies by time of day
        # Peak hours (2pm-8pm, intervals 28-40) get tightest limits
        doe_max = np.inf * np.ones(T)
        doe_min = np.zeros(T)
        
        # Off-peak (10pm-7am): moderate limit
        doe_min[:14] = -0.5 * base_export_limit     # 00:00-07:00
        doe_min[44:] = -0.5 * base_export_limit     # 22:00-24:00
        
        # Shoulder (7am-2pm, 8pm-10pm): tighter
        doe_min[14:28] = -0.3 * base_export_limit   # 07:00-14:00
        doe_min[40:44] = -0.3 * base_export_limit   # 20:00-22:00
        
        # Peak (2pm-8pm): very tight
        doe_min[28:40] = -0.15 * base_export_limit  # 14:00-20:00
    
    elif scenario == "rolling":
        # Realistic: day-ahead forecast ± 10% uncertainty
        # Simulate a "nominal" envelope that varies by hour
        nominal = np.array([
            -2.0, -2.0, -1.5, -1.5, -1.5, -1.5, -1.0,  # off-peak
            -1.0, -1.0, -1.0, -0.8, -0.8, -0.8,        # shoulder
            -0.8, -0.6, -0.5, -0.5, -0.4, -0.4, -0.4,  # peak
            -0.5, -0.6, -0.8, -1.0, -1.2, -1.5,        # evening
            -1.5, -1.5, -1.5, -2.0, -2.0, -2.0,        # night
            -2.5, -2.5, -2.5, -2.0, -2.0, -2.0,        # early morning
            -1.5, -1.5, -1.2, -1.2, -1.0, -1.0,        # before 7am
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.5,        # 6am-8am
        ])
        assert len(nominal) == T
        # Add ±10% uncertainty band
        doe_min = nominal * 1.1  # looser in uncertainty direction (allow more export)
        doe_max = np.inf * np.ones(T)
    
    else:
        raise ValueError(f"Unknown DOE scenario: {scenario}")

    if np.isfinite(base_import_limit):
        doe_max = float(base_import_limit) * np.ones(T)

    return doe_min, doe_max


def load_doe_from_csv(path, scenario_col="scenario", doe_min_col="p_min_kw", 
                      doe_max_col="p_max_kw"):
    """
    Load DOE envelopes from a CSV file (one row per half-hour interval).
    Expected columns: scenario, p_min_kw, p_max_kw (or custom).
    
    Returns:
    --------
    dict {date: (doe_min, doe_max) arrays}
    """
    df = pd.read_csv(path)
    out = {}
    for date, group in df.groupby("date"):
        if len(group) != T:
            logger.warning("Date %s has %d intervals, expected %d, skipping",
                          date, len(group), T)
            continue
        doe_min = group[doe_min_col].values.astype(np.float64)
        doe_max = group[doe_max_col].values.astype(np.float64)
        out[date] = (doe_min, doe_max)
    return out


# ==========================================================
# MODIFIED: BUILD CONSTRAINTS WITH DOE BOUNDS
# ==========================================================

# Decision vector x = [b | c | s], each of length T:
#   b  battery power (kW, > 0 discharge, < 0 charge)          -- as in osqp_daily
#   c  curtailed PV (kW, 0 <= c <= pv)                          -- export-side relief
#   s  import shortfall (kW, >= 0)                              -- import-side slack
# Grid flow  p = load - pv + c - b = net + c - b.
IDX_B = slice(0, T)
IDX_C = slice(T, 2 * T)
IDX_S = slice(2 * T, 3 * T)
N_VAR = 3 * T
N_BASE_ROWS = 2 * T + 1                    # [b bounds][SOC][sum=0]
N_ROWS = N_BASE_ROWS + 4 * T               # + [c box][s box][export][import]

# Linear penalties on the relief variables, PER UNIT WEIGHT h_k: curtailing or
# falling short by 1 kW in interval k costs PENALTY_KW * h_k in the objective.
# The marginal benefit a kW of relief can ever bring to the flattening term is
# |d(h p^2)/dp| = 2 h_k |p_k|, so with PENALTY_KW = 100 relief is never worth
# it below |p| = 50 kW -- i.e. curtailment and import shortfall are used ONLY
# when the battery cannot meet the envelope (feasibility), never to flatten.
# Scaling by h_k (rather than one huge constant) keeps the QP well
# conditioned whatever the heuristic does to the weights. The bill still sees
# the true cost of curtailment (lost FiT / self-consumption) through pv - c.
PENALTY_KW = 100.0


@dataclass
class DispatchResult:
    """One day's DOE-constrained dispatch."""
    b: np.ndarray               # battery kW (+ discharge)
    curtail: np.ndarray         # curtailed PV kW (>= 0); non-zero only under a finite export cap
    import_slack: np.ndarray    # import-cap shortfall kW (>= 0); load cannot be shed
    status: str                 # OSQP status string

    @property
    def doe_feasible(self):
        """Envelope met exactly? Export is always met (curtailment); import
        is met iff no shortfall was needed."""
        return bool(np.max(self.import_slack, initial=0.0) <= 1e-4)


def build_constraints(e_max, soc_init_frac=0.5, p_max=P_MAX,
                      doe_min=None, doe_max=None):
    """
    Build the stacked (A, l, u) constraint block for OSQP over x = [b, c, s].

    Rows (fixed sparsity; only bounds change per day):
      1. |b_k| <= p_max                          rate limit            (T)
      2. 0 <= SOC_k <= e_max                     via cumulative sum    (T)
      3. sum(b) = 0                              daily neutrality      (1)
      4. 0 <= c_k <= pv_k                        curtail only real PV  (T)
      5. s_k >= 0                                shortfall slack       (T)
      6. c_k - b_k >= doe_min_k - net_k          export cap, HARD      (T)
         (p = net + c - b >= doe_min; always feasible: curtail c = pv,
          b = 0 gives p = load >= doe_min since doe_min <= 0)
      7. c_k - b_k - s_k <= doe_max_k - net_k    import cap, SOFT      (T)
         (p <= doe_max + s; s > 0 only when the battery cannot supply
          load - doe_max, e.g. load - doe_max > P_MAX or the SOC runs out)

    The persistent OSQP workspace keeps ONE sparsity pattern for the whole
    run (paper_context.md section 4 invariant); with no envelope rows 6-7
    are inactive (+-inf) and row 4 pins c = 0, so the problem is exactly
    osqp_daily's. Passing a differently-shaped A to solver.update() is not
    supported (osqp 1.x silently ignores an `A=` keyword) -- that is how the
    DOE constraint was a no-op before 2026-08-16.
    """
    soc_init = soc_init_frac * e_max
    I_T = sp.eye(T, format="csc")
    Z_T = sp.csc_matrix((T, T))
    A_soc = sp.csc_matrix(np.tril(np.ones((T, T))) * DT)
    ones_row = sp.csc_matrix(np.ones((1, T)))
    Z_row = sp.csc_matrix((1, T))

    A = sp.vstack([
        sp.hstack([I_T, Z_T, Z_T]),          # 1 rate
        sp.hstack([-A_soc, Z_T, Z_T]),       # 2 SOC
        sp.hstack([ones_row, Z_row, Z_row]),  # 3 neutrality
        sp.hstack([Z_T, I_T, Z_T]),          # 4 curtail box
        sp.hstack([Z_T, Z_T, I_T]),          # 5 slack box
        sp.hstack([-I_T, I_T, Z_T]),         # 6 export: c - b
        sp.hstack([-I_T, I_T, -I_T]),        # 7 import: c - b - s
    ]).tocsc()

    inf = np.inf * np.ones(T)
    l = np.hstack([-p_max * np.ones(T), -soc_init * np.ones(T), [0.0],
                   np.zeros(T), np.zeros(T), -inf, -inf])
    u = np.hstack([p_max * np.ones(T), (e_max - soc_init) * np.ones(T), [0.0],
                   np.zeros(T), inf, inf, inf])

    doe_info = {"doe_min": doe_min, "doe_max": doe_max}
    return A, l, u, doe_info


def envelope_row_bounds(net, pv, doe_min=None, doe_max=None):
    """
    Per-day bounds for rows 4-7 (curtail box, slack box, export, import).

    With no envelope: c pinned to 0, export/import rows inactive. With an
    envelope: c may take up to pv where an export cap is finite; export row
    l = doe_min - net; import row u = doe_max - net.
    """
    inf = np.inf * np.ones(T)
    if doe_min is None or doe_max is None:
        return (np.zeros(T), np.zeros(T),      # c box
                np.zeros(T), inf,              # s box
                -inf, inf,                     # export
                -inf, inf)                     # import
    doe_min = np.asarray(doe_min, dtype=float)
    doe_max = np.asarray(doe_max, dtype=float)
    c_upper = np.where(np.isfinite(doe_min), np.maximum(pv, 0.0), 0.0)
    return (np.zeros(T), c_upper,
            np.zeros(T), inf,
            doe_min - net, inf,
            -inf, doe_max - net)


# ==========================================================
# MODIFIED: PERSISTENT OSQP SOLVER WITH DOE
# ==========================================================

_SOLVER_CACHE = {"solver": None, "e_max": None}


# Tiny quadratic regularisation on the relief variables (relative to h):
# without it P is singular along the (b, c) direction wherever the envelope
# pins p exactly (e.g. a zero-export cap), and OSQP's ADMM crawls like on an
# LP. 1e-3 keeps P strictly convex and shifts the answer by < 0.1 %.
RELIEF_REG = 1e-3


def build_P(h_diag):
    """
    Upper-triangular P for  sum_k h_k (net_k + c_k - b_k)^2  (+ linear
    penalties, which live in q). Expanding the square:
        P_bb = 2h,  P_cc = 2h (1 + reg),  P_bc = -2h,  P_ss = 2h reg.
    Returned as CSC upper triangle so `.data` has a fixed order for
    solver.update(Px=...).
    """
    h = np.asarray(h_diag, dtype=float)
    D = sp.diags(2.0 * h, format="csc")
    Dc = sp.diags(2.0 * h * (1.0 + RELIEF_REG), format="csc")
    Ds = sp.diags(2.0 * h * RELIEF_REG, format="csc")
    Z = sp.csc_matrix((T, T))
    P = sp.bmat([[D, -D, Z],
                 [None, Dc, Z],
                 [None, None, Ds]], format="csc")
    return sp.triu(P, format="csc")


def _get_solver(e_max, doe_info=None):
    """
    Build the OSQP workspace once per e_max; the day-specific numbers enter
    through solver.update(Px=, q=, l=, u=) with a fixed sparsity pattern.
    """
    cache = _SOLVER_CACHE
    if cache["solver"] is not None and cache["e_max"] == e_max:
        return cache["solver"]

    A, l, u, _ = build_constraints(e_max, doe_min=None, doe_max=None)
    P0 = build_P(np.ones(T))
    q0 = np.zeros(N_VAR)

    solver = osqp.OSQP()
    solver.setup(
        P=P0, q=q0, A=A, l=l, u=u,
        verbose=False,
        eps_abs=1e-6, eps_rel=1e-6,
        polish=True, warm_start=True,
        # a zero-export cap pins p = 0 over the PV window, where the
        # flattening term is flat in (b, c) and OSQP's ADMM converges slowly;
        # give it room rather than accept an unpolished iterate
        max_iter=40000,
    )
    cache["solver"] = solver
    cache["e_max"] = e_max
    cache["P_nnz"] = P0.nnz
    return solver


def solve_battery(load, pv, h_diag, e_max, doe_min=None, doe_max=None):
    """
    Solve the QP for a given day with optional DOE constraints on the grid
    flow p = load - pv + c - b:

        doe_min_k <= p_k             hard  (curtailment c makes it feasible)
        p_k <= doe_max_k + s_k       soft  (shortfall s reported, penalised)

    minimise  sum_k h_k p_k^2 + PENALTY_KW * sum_k h_k (c_k + s_k)

    Returns a DispatchResult (b, curtail, import_slack, status). Without an
    envelope c and s are pinned to zero and the result equals
    osqp_daily.solve_battery.
    """
    solver = _get_solver(e_max)
    net = load - pv                       # p when b = c = 0
    h = np.asarray(h_diag, dtype=float)

    q = np.concatenate([-2.0 * h * net,                          # b
                        2.0 * h * net + PENALTY_KW * h,           # c
                        PENALTY_KW * h])                          # s

    _, l_base, u_base, _ = build_constraints(e_max)
    l_base, u_base = l_base[:N_BASE_ROWS], u_base[:N_BASE_ROWS]
    lc, uc, ls, us, lx, ux, li, ui = envelope_row_bounds(net, pv, doe_min, doe_max)

    solver.update(Px=build_P(h).data, q=q,
                  l=np.hstack([l_base, lc, ls, lx, li]),
                  u=np.hstack([u_base, uc, us, ux, ui]))
    res = solver.solve()
    if res.info.status_val not in (1, 2):
        logger.warning("OSQP status: %s", res.info.status)

    x = np.asarray(res.x, dtype=float)
    return DispatchResult(b=x[IDX_B].copy(),
                         curtail=np.maximum(x[IDX_C], 0.0),
                         import_slack=np.maximum(x[IDX_S], 0.0),
                         status=res.info.status)


# ==========================================================
# TARIFF (unchanged)
# ==========================================================

def build_tariff():
    """Paper's TOU: off-peak 0.03, shoulder 0.06, peak 0.30 $/kWh."""
    tariff = np.zeros(T)
    tariff[0:14] = 0.03
    tariff[44:48] = 0.03
    tariff[14:28] = 0.06
    tariff[40:44] = 0.06
    tariff[28:40] = 0.30
    return tariff


# ==========================================================
# BILLING (unchanged)
# ==========================================================

def bill_topology1(load, pv, b, tariff, fit=FIT_RATE):
    flow_m2 = np.maximum(load - b, 0.0)
    return np.sum(flow_m2 * tariff * DT) - np.sum(pv * fit * DT)


def bill_topology2(load, pv, b, tariff, net_credit=FIT_RATE):
    p = load - pv - b
    imp = np.maximum(p, 0.0)
    exp = np.maximum(-p, 0.0)
    return np.sum(imp * tariff * DT) - np.sum(exp * net_credit * DT)


def bill(load, pv, b, tariff, mode):
    if mode == "fit":
        return bill_topology1(load, pv, b, tariff)
    if mode == "net":
        return bill_topology2(load, pv, b, tariff)
    raise ValueError(f"unknown mode {mode!r}")


# ==========================================================
# HEURISTIC H (unchanged)
# ==========================================================

def build_H0_diag(tariff):
    h_tilde = tariff.copy()
    h_plus = np.min(h_tilde[h_tilde > 0]) if np.any(h_tilde > 0) else 1.0
    h0 = h_tilde / h_plus
    return np.clip(h0, 1.0, H_BAR)


def optimise_H(load, pv, tariff, e_max, mode, doe_min=None, doe_max=None):
    """
    Paper heuristic with DOE support. The weights H are optimized
    to maximize savings subject to DOE constraints.
    """
    h = build_H0_diag(tariff)
    base_cost = bill(load, pv, np.zeros(T), tariff, mode)

    def savings_for(h_vec):
        res = solve_battery(load, pv, h_vec, e_max, doe_min, doe_max)
        # curtailed PV is not generated: the FiT / self-consumption it
        # would have earned is lost, so the bill sees pv - c
        return base_cost - bill(load, pv - res.curtail, res.b, tariff, mode), res

    best_s, best_res = savings_for(h)
    best_h = h.copy()

    unique_levels = np.unique(h)[::-1]
    tiers = [np.where(h == lvl)[0] for lvl in unique_levels]

    current = h.copy()
    for _ in range(HEURISTIC_MAX_ITERS):
        improved_this_round = False
        for idx in tiers:
            trial = current.copy()
            trial[idx] = np.minimum(trial[idx] * 2.0, H_BAR)
            if np.allclose(trial, current):
                continue
            s, res = savings_for(trial)
            if s > best_s + 1e-12:
                best_s, best_res, best_h = s, res, trial.copy()
                current = trial
                improved_this_round = True
        if not improved_this_round:
            break

    return best_h, best_res, best_s


# ==========================================================
# MODIFIED: SINGLE-DAY SIMULATION WITH DOE
# ==========================================================

@dataclass
class DayResult:
    """simulate_day() output for one customer-day."""
    savings: float              # $ vs the no-battery baseline (curtailment costed)
    b: np.ndarray               # battery kW (+ discharge)
    p: np.ndarray               # grid flow kW = load - pv + curtail - b
    h: np.ndarray               # heuristic weights used
    doe_compliant: bool         # p inside [doe_min, doe_max] everywhere
    doe_slack: np.ndarray       # post-hoc breach of the envelope, kW (>= 0)
    curtail: np.ndarray         # curtailed PV kW
    import_slack: np.ndarray    # import-cap shortfall kW (solver's slack)


def simulate_day(load, pv, tariff, mode, e_max=E_MAX_DEFAULT,
                 doe_min=None, doe_max=None):
    """
    Simulate a single day with optional DOE constraints. Returns a DayResult.

    The export side is always met (curtailment absorbs what the battery
    cannot); the import side is met unless load - doe_max exceeds what the
    battery can supply, in which case doe_compliant is False and doe_slack /
    import_slack carry the shortfall.
    """
    h, res, savings = optimise_H(load, pv, tariff, e_max, mode, doe_min, doe_max)
    p = load - pv + res.curtail - res.b

    doe_compliant = True
    doe_slack = np.zeros(T)
    if doe_min is not None and doe_max is not None:
        tol = 1e-4
        below_min = p < (doe_min - tol)
        above_max = p > (doe_max + tol)
        doe_slack[below_min] = doe_min[below_min] - p[below_min]
        doe_slack[above_max] = p[above_max] - doe_max[above_max]
        if np.any(below_min) or np.any(above_max):
            doe_compliant = False

    return DayResult(savings=savings, b=res.b, p=p, h=h,
                     doe_compliant=doe_compliant, doe_slack=doe_slack,
                     curtail=res.curtail, import_slack=res.import_slack)


# ==========================================================
# MODIFIED: PER-CUSTOMER WORKER WITH DOE TRACKING
# ==========================================================

def _worker(args):
    """
    Worker function for multiprocessing. Now tracks DOE compliance.
    """
    (customer, days, tariff, mode, e_max,
     doe_scenario, base_export_limit, base_import_limit) = args

    total = 0.0
    day_profiles = []
    soc_init = 0.5 * e_max

    # Generate DOE envelope (same for all days of this customer)
    doe_min, doe_max = generate_doe_envelope(doe_scenario, base_export_limit,
                                            base_import_limit)

    for date, load, pv in days:
        r = simulate_day(load, pv, tariff, mode, e_max, doe_min, doe_max)

        total += r.savings
        soc = soc_init - np.cumsum(r.b) * DT

        day_profiles.append({
            "date": date,
            "load": load,
            "pv": pv,
            "battery": r.b,
            "grid": r.p,
            "soc": soc,
            "savings": r.savings,
            "doe_compliant": r.doe_compliant,
            "doe_slack": r.doe_slack,
            "doe_slack_total": float(np.sum(r.doe_slack)),
            "curtail": r.curtail,
            "curtail_kwh": float(np.sum(r.curtail) * DT),
            "import_shortfall": r.import_slack,
            "import_shortfall_kwh": float(np.sum(r.import_slack) * DT),
        })

    return customer, total, day_profiles


def run_all(day_arrays, mode, e_max=E_MAX_DEFAULT, doe_scenario="none",
            base_export_limit=3.0, base_import_limit=np.inf):
    """
    Run simulation for every customer with DOE support.

    Parameters:
    -----------
    doe_scenario : str
        'none', 'conservative', 'tight', 'rolling'   (export-side shape)
    base_export_limit : float
        Feeder export headroom (kW) the scenario scales -- see
        generate_doe_envelope(). Sweep it to ask "how tight must the
        envelope be before the network sees zero over-voltage?"
    base_import_limit : float
        Flat import cap (kW), inf = none. Targets the synchronised off-peak
        charging; shortfall (load the battery cannot cover) is reported.
    """
    tariff = build_tariff()
    jobs = [(cust, days, tariff, mode, e_max, doe_scenario,
             base_export_limit, base_import_limit)
            for cust, days in day_arrays.items()]
    n_proc = min(cpu_count(), len(jobs)) or 1

    logger.info(
        "Running %s simulations on %d cores (E_max=%.1f kWh, DOE=%s, "
        "base export limit=%.2f kW, import limit=%s kW)",
        mode, n_proc, e_max, doe_scenario, base_export_limit,
        f"{base_import_limit:.2f}" if np.isfinite(base_import_limit) else "inf")

    customers, savings = [], []
    all_profiles = {}
    doe_stats = {"compliant_days": 0, "violating_days": 0,
                 "curtail_kwh": 0.0, "import_shortfall_kwh": 0.0,
                 "days_with_curtail": 0, "days_with_shortfall": 0}

    with Pool(processes=n_proc) as pool:
        for cust, total, profiles in pool.imap_unordered(_worker, jobs, chunksize=1):
            compliant = sum(1 for p in profiles if p["doe_compliant"])
            violating = len(profiles) - compliant
            doe_stats["compliant_days"] += compliant
            doe_stats["violating_days"] += violating
            doe_stats["curtail_kwh"] += sum(p["curtail_kwh"] for p in profiles)
            doe_stats["import_shortfall_kwh"] += sum(p["import_shortfall_kwh"] for p in profiles)
            doe_stats["days_with_curtail"] += sum(1 for p in profiles if p["curtail_kwh"] > 1e-3)
            doe_stats["days_with_shortfall"] += sum(1 for p in profiles if p["import_shortfall_kwh"] > 1e-3)

            logger.info("Customer %s: $%.2f/yr (%d/%d days DOE-compliant)",
                        cust, total, compliant, len(profiles))
            customers.append(cust)
            savings.append(total)
            all_profiles[cust] = profiles

    order = np.argsort(customers)
    n_days = doe_stats["compliant_days"] + doe_stats["violating_days"]

    logger.info("=== DOE Compliance Summary ===")
    logger.info("  Total days simulated: %d", n_days)
    logger.info("  DOE-compliant days:   %d (%.1f%%)", doe_stats["compliant_days"],
                100.0 * doe_stats["compliant_days"] / (n_days + 1e-6))
    logger.info("  DOE-violating days:   %d (import shortfall only -- export is "
                "always met via curtailment)", doe_stats["violating_days"])
    logger.info("  PV curtailed:         %.1f kWh over %d customer-days",
                doe_stats["curtail_kwh"], doe_stats["days_with_curtail"])
    logger.info("  Import shortfall:     %.1f kWh over %d customer-days",
                doe_stats["import_shortfall_kwh"], doe_stats["days_with_shortfall"])

    return (np.array(customers)[order],
            np.array(savings)[order],
            all_profiles)


# ==========================================================
# NEW: COMPARISON FUNCTION
# ==========================================================

def compare_scenarios(day_arrays, mode, e_max=E_MAX_DEFAULT):
    """
    Run side-by-side comparison of different DOE scenarios.
    Returns a DataFrame comparing annual savings and DOE impact.
    """
    scenarios = ["none", "conservative", "tight"]
    results = {}
    
    for scenario in scenarios:
        logger.info("\n>>> Running scenario: %s", scenario.upper())
        customers, savings, profiles = run_all(day_arrays, mode, e_max, scenario)
        results[scenario] = {
            "customers": customers,
            "savings": savings,
            "profiles": profiles,
        }
    
    # Build comparison table
    df_comparison = pd.DataFrame({
        "customer": results["none"]["customers"],
        "savings_no_doe": results["none"]["savings"],
        "savings_conservative": results["conservative"]["savings"],
        "savings_tight": results["tight"]["savings"],
    })
    
    df_comparison["impact_conservative"] = (
        df_comparison["savings_conservative"] - 
        df_comparison["savings_no_doe"])
    df_comparison["impact_tight"] = (
        df_comparison["savings_tight"] - 
        df_comparison["savings_no_doe"])
    
    logger.info("\n=== SCENARIO COMPARISON ===")
    logger.info("Mean savings, no DOE:      $%.2f/yr",
               df_comparison["savings_no_doe"].mean())
    logger.info("Mean savings, conservative: $%.2f/yr",
               df_comparison["savings_conservative"].mean())
    logger.info("Mean savings, tight:        $%.2f/yr",
               df_comparison["savings_tight"].mean())
    logger.info("\nMean impact on savings:")
    logger.info("  Conservative DOE: $%.2f/yr (%.1f%%)",
               df_comparison["impact_conservative"].mean(),
               100.0 * df_comparison["impact_conservative"].mean() / 
               (df_comparison["savings_no_doe"].mean() + 1e-6))
    logger.info("  Tight DOE:        $%.2f/yr (%.1f%%)",
               df_comparison["impact_tight"].mean(),
               100.0 * df_comparison["impact_tight"].mean() / 
               (df_comparison["savings_no_doe"].mean() + 1e-6))
    
    return df_comparison


# ==========================================================
# SAVE PROFILES (modified for DOE)
# ==========================================================

def save_profiles(all_profiles, mode, doe_scenario="none", out_dir=str(PROFILES)):
    """
    Save half-hourly simulation results, including DOE compliance flags.
    """
    import os
    
    os.makedirs(out_dir, exist_ok=True)
    per_day_dir = os.path.join(out_dir, f"{mode}_doe_{doe_scenario}")
    os.makedirs(per_day_dir, exist_ok=True)
    
    intervals = np.arange(1, T + 1)
    hours = np.arange(T) * DT
    
    rows = []
    for cust in sorted(all_profiles.keys()):
        cust_dir = os.path.join(per_day_dir, f"cust_{cust}")
        os.makedirs(cust_dir, exist_ok=True)
        
        for day_prof in all_profiles[cust]:
            date = day_prof["date"]
            load = day_prof["load"]
            pv = day_prof["pv"]
            batt = day_prof["battery"]
            grid = day_prof["grid"]
            soc = day_prof["soc"]
            sav = day_prof["savings"]
            doe_compliant = day_prof["doe_compliant"]
            doe_slack = day_prof["doe_slack"]
            curtail = day_prof.get("curtail", np.zeros(T))
            shortfall = day_prof.get("import_shortfall", np.zeros(T))

            fname = os.path.join(cust_dir, f"{date}.csv")
            np.savetxt(fname, grid, fmt="%.6f", delimiter=",")

            for k in range(T):
                rows.append({
                    "customer": int(cust),
                    "date": date,
                    "interval": int(intervals[k]),
                    "hour": float(hours[k]),
                    "load_kw": float(load[k]),
                    "pv_kw": float(pv[k]),          # raw PV; curtailment is separate
                    "battery_kw": float(batt[k]),
                    "grid_kw": float(grid[k]),      # = load - pv + curtail - battery
                    "soc_kwh": float(soc[k]),
                    "daily_savings": float(sav),
                    "doe_compliant": int(doe_compliant),
                    "doe_slack_kw": float(doe_slack[k]),
                    "curtail_kw": float(curtail[k]),
                    "import_shortfall_kw": float(shortfall[k]),
                })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{mode}_doe_{doe_scenario}.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)
    return csv_path


# ==========================================================
# EXAMPLE: PLOT DOE IMPACT
# ==========================================================

def plot_doe_impact(day_arrays, customer_id, date_str, mode="fit",
                    e_max=E_MAX_DEFAULT):
    """
    Plot a single day's results across multiple DOE scenarios.
    """
    tariff = build_tariff()
    
    # Find the day
    found_load, found_pv = None, None
    for days in day_arrays.values():
        for date, load, pv in days:
            if str(date) == date_str:
                found_load, found_pv = load, pv
                break
    
    if found_load is None:
        logger.warning("Day not found: customer %s, %s", customer_id, date_str)
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    hours = np.arange(T) * DT
    
    scenarios = ["none", "conservative", "tight"]
    
    for idx, scenario in enumerate(scenarios):
        ax_p = axes[idx // 2, idx % 2]
        
        doe_min, doe_max = generate_doe_envelope(scenario)
        r = simulate_day(found_load, found_pv, tariff, mode, e_max,
                         doe_min, doe_max)

        ax_p.fill_between(hours, doe_min, doe_max, alpha=0.2,
                          color="green", label="DOE envelope")
        ax_p.plot(hours, r.p, color="steelblue", marker="o",
                  label=f"grid flow (${r.savings:.2f}/day)", linewidth=2)
        if r.curtail.max() > 1e-6:
            ax_p.plot(hours, r.curtail, color="orange", linestyle=":",
                      label="curtailed PV")
        ax_p.axhline(0, color="black", lw=0.5, linestyle="--")
        ax_p.set_ylabel("Power (kW)")
        ax_p.set_title(
            f"Scenario: {scenario.upper()} | Compliant: {r.doe_compliant} | "
            f"Curtailed: {np.sum(r.curtail) * DT:.2f} kWh | "
            f"Shortfall: {np.sum(r.import_slack) * DT:.2f} kWh")
        ax_p.grid(alpha=0.3)
        ax_p.legend()
    
    # Fourth panel: load and PV
    ax_lg = axes[1, 1]
    ax_lg.plot(hours, found_load, label="load", color="black", linewidth=2)
    ax_lg.plot(hours, found_pv, label="PV", color="orange", linewidth=2)
    ax_lg.set_xlabel("Hour of day")
    ax_lg.set_ylabel("Power (kW)")
    ax_lg.set_title(f"Load & PV | Customer {customer_id}, {date_str}")
    ax_lg.grid(alpha=0.3)
    ax_lg.legend()
    
    plt.tight_layout()
    plt.show()


# ==========================================================
# DATA LOADING (duplicated from osqp_daily.py -- keep in sync)
# ==========================================================
# This script is a copy-extension of osqp_daily.py, not an import (see
# CLAUDE.md): the loader, cleaning rules and constants below mirror the base
# script and any fix there must be applied here as well.

def load_dataset(path):
    """Read the Ausgrid CSV and return a long-format kW DataFrame
    (mirror of osqp_daily.load_dataset)."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        first_line = fh.readline()
    has_title_row = "Customer" not in first_line.split(",")[0]
    df = pd.read_csv(path, skiprows=1 if has_title_row else 0)
    
    time_cols = list(df.columns[5:])
    if len(time_cols) != T:
        raise ValueError(
            f"Expected {T} time columns, found {len(time_cols)}: {time_cols}")
    interval_map = {label: i + 1 for i, label in enumerate(time_cols)}
    
    df_long = df.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode",
                 "Consumption Category", "date"],
        value_vars=time_cols,
        var_name="time_label",
        value_name="energy_kwh",
    )
    df_long["interval"] = df_long["time_label"].map(interval_map)
    df_long["energy_kwh"] = pd.to_numeric(
        df_long["energy_kwh"], errors="coerce").fillna(0.0)
    df_long["power_kw"] = df_long["energy_kwh"] / DT
    
    pivot = df_long.pivot_table(
        index=["Customer", "date", "interval"],
        columns="Consumption Category",
        values="power_kw",
    ).reset_index()
    pivot.columns.name = None
    for col in ("GC", "CL", "GG"):
        if col not in pivot.columns:
            pivot[col] = 0.0
        pivot[col] = pivot[col].fillna(0.0)
    
    pivot["date_parsed"] = pd.to_datetime(pivot["date"], format="%d-%b-%y")
    pivot = pivot.sort_values(["Customer", "date_parsed", "interval"])
    pivot["load"] = pivot["GC"] + pivot["CL"]
    pivot["pv"] = pivot["GG"]
    
    logger.info(
        "Loaded %d rows | %d customers | %d unique dates (%s..%s)",
        len(pivot),
        pivot["Customer"].nunique(),
        pivot["date_parsed"].nunique(),
        pivot["date_parsed"].min().strftime("%Y-%m-%d"),
        pivot["date_parsed"].max().strftime("%Y-%m-%d"),
    )
    return pivot


def identify_clean_customers(df):
    """[COPY FROM ORIGINAL]"""
    daily = df.groupby(["Customer", "date_parsed"], sort=False).agg(
        gc_max=("GC", "max"),
        gg_max=("GG", "max"),
        gg_sum=("GG", "sum"),
        n_intervals=("interval", "count"),
    )
    
    early = (
        df[df["interval"] <= GG_CAT3_EARLY_LIMIT]
        .groupby(["Customer", "date_parsed"], sort=False)["GG"]
        .sum()
        .rename("gg_early_sum")
    )
    daily = daily.join(early)
    daily["gg_early_sum"] = daily["gg_early_sum"].fillna(0.0)
    
    daily["gc_anom"] = daily["gc_max"] < GC_MAX_THRESHOLD
    daily["gg_cat1"] = daily["gg_max"] < GG_CAT1_MAX
    daily["gg_cat2"] = (daily["gg_max"] < GG_CAT2_MAX) & \
                       (daily["gg_sum"] <= GG_CAT2_SUM)
    daily["gg_cat3"] = daily["gg_early_sum"] > GG_CAT3_EARLY_SUM
    
    per_cust = daily.groupby(level="Customer").agg(
        any_gc=("gc_anom", "any"),
        any_c1=("gg_cat1", "any"),
        any_c2=("gg_cat2", "any"),
        any_c3=("gg_cat3", "any"),
    )
    bad = (per_cust["any_gc"] | per_cust["any_c1"]
           | per_cust["any_c2"] | per_cust["any_c3"])
    
    n_total = len(per_cust)
    logger.info("Cleaning report (Ratnam et al. 2017, Section 3):")
    logger.info("  Total customers in dataset             : %4d", n_total)
    logger.info("  Removed by GC rule (load <6W any day)  : %4d",
                int(per_cust["any_gc"].sum()))
    logger.info("  Removed by GG Cat 1 (peak <60W any day): %4d",
                int(per_cust["any_c1"].sum()))
    logger.info("  Removed by GG Cat 2 (daily PV <=0.325kWh): %4d",
                int(per_cust["any_c2"].sum()))
    logger.info("  Removed by GG Cat 3 (PV before 5 am)   : %4d",
                int(per_cust["any_c3"].sum()))
    logger.info("  Removed by ANY rule (with overlap)     : %4d",
                int(bad.sum()))
    logger.info("  Clean customers retained               : %4d",
                int((~bad).sum()))
    
    return sorted(per_cust.index[~bad].astype(int).tolist())


def clean_dataset(df):
    """[COPY FROM ORIGINAL]"""
    clean_ids = identify_clean_customers(df)
    df_clean = df[df["Customer"].isin(clean_ids)]
    
    good_days = []
    incomplete = 0
    for _, day in df_clean.groupby(["Customer", "date_parsed"], sort=False):
        if len(day) == T:
            good_days.append(day)
        else:
            incomplete += 1
    
    if not good_days:
        logger.warning("No complete days survived cleaning!")
        return pd.DataFrame(columns=df.columns)
    
    out = pd.concat(good_days, ignore_index=True)
    logger.info("Customers after cleaning + completeness : %d",
                out["Customer"].nunique())
    logger.info("Incomplete days dropped (DST etc.)      : %d", incomplete)
    return out


def extract_day_arrays(df):
    """[COPY FROM ORIGINAL]"""
    out = {}
    df = df.sort_values(["Customer", "date_parsed", "interval"])
    for cust, cust_df in df.groupby("Customer", sort=True):
        days = []
        for date_parsed, day in cust_df.groupby("date_parsed", sort=True):
            if len(day) != T:
                continue
            date_str = day["date"].iloc[0]
            days.append((date_str,
                         day["load"].to_numpy(dtype=np.float64),
                         day["pv"].to_numpy(dtype=np.float64)))
        out[int(cust)] = days
    return out


# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Load data, run the requested DOE scenarios, save network-ready profile
    CSVs (outputs/profiles/<mode>_doe_<label>.csv, the same long format
    elermorevale_openDSS.py --profiles reads), and optionally the
    scenario-comparison table.

    Defaults reproduce the original workflow: fit mode, scenarios
    none / conservative / tight at a 3 kW feeder export headroom, followed
    by the comparison table.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="DOE-constrained QP battery dispatch (extension of "
                    "osqp_daily.py). Writes outputs/profiles/<mode>_doe_<label>.csv "
                    "for each scenario, consumable by the network scripts.")
    parser.add_argument("--data", default=str(DATA_CSV),
                        help="Ausgrid CSV (default: data/data.csv)")
    parser.add_argument("--mode", choices=["fit", "net"], default="fit",
                        help="Tariff mode passed to the QP (default: fit)")
    parser.add_argument("--scenarios", nargs="+",
                        default=["none", "conservative", "tight"],
                        help="DOE scenarios from generate_doe_envelope() "
                             "(default: none conservative tight)")
    parser.add_argument("--export-limit", type=float, nargs="+", default=[3.0],
                        help="base_export_limit values (kW) to run each "
                             "scenario at; more than one value labels the "
                             "outputs <scenario>_cap<value> (default: 3.0)")
    parser.add_argument("--import-limit", type=float, default=np.inf,
                        help="Flat per-household IMPORT cap (kW) applied on "
                             "top of every scenario; the label gains "
                             "_imp<kW>. Targets the synchronised off-peak "
                             "charging (default: none)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the none/conservative/tight comparison "
                             "table (which re-runs all three scenarios)")
    args = parser.parse_args()

    df_raw = load_dataset(args.data)
    df_clean = clean_dataset(df_raw)
    day_arrays = extract_day_arrays(df_clean)
    logger.info("Extracted %d customers", len(day_arrays))

    label_with_cap = len(args.export_limit) > 1
    imp_suffix = (f"_imp{args.import_limit:g}"
                  if np.isfinite(args.import_limit) else "")
    baseline_savings = None
    for cap in args.export_limit:
        for scenario in args.scenarios:
            label = (f"{scenario}_cap{cap:g}" if label_with_cap else scenario) + imp_suffix
            logger.info("\n" + "=" * 60)
            logger.info("SCENARIO %s (mode=%s, base export limit %.2f kW, "
                        "import limit %s)", label.upper(), args.mode, cap,
                        f"{args.import_limit:g} kW" if np.isfinite(args.import_limit) else "none")
            logger.info("=" * 60)
            _, savings, profiles = run_all(
                day_arrays, mode=args.mode, doe_scenario=scenario,
                base_export_limit=cap, base_import_limit=args.import_limit)
            logger.info("Mean annual savings (%s): $%.2f", label,
                        np.mean(savings))
            if scenario == "none":
                baseline_savings = savings
            elif baseline_savings is not None:
                delta = np.mean(savings - baseline_savings)
                logger.info("Impact vs no-DOE: $%.2f/yr (%.1f%%)", delta,
                            100.0 * delta / (np.mean(baseline_savings) + 1e-6))
            save_profiles(profiles, mode=args.mode, doe_scenario=label)

    if not args.no_compare:
        logger.info("\n" + "=" * 60)
        logger.info("SCENARIO COMPARISON SUMMARY")
        logger.info("=" * 60)
        df_comp = compare_scenarios(day_arrays, mode=args.mode)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(OUTPUTS / "doe_scenario_comparison.csv", index=False)


if __name__ == "__main__":
    main()
