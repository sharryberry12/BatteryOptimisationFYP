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
Outputs profiles/<mode>_doe_<scenario>[_cap<kW>].csv in the long format the
network scripts read (elermorevale_openDSS.py --profiles ...).

Constraint mechanics (2026-08-16 fix): the DOE rows are an identity block that
is part of the persistent OSQP workspace from setup and are switched on per
day by updating their bounds. Before this fix the rows were passed to
solver.update(A=...), which osqp silently ignores, so no DOE result generated
earlier actually had the envelope enforced (all scenarios gave identical
dispatch). Days whose envelope is infeasible for the battery fall back to
the unconstrained dispatch and are flagged (doe_compliant=False, doe_slack>0)
-- PV curtailment is NOT modelled. Regression tests: tests/test_doe_constraints.py.
"""

import logging
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp

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

def generate_doe_envelope(scenario="conservative", base_export_limit=3.0):
    """
    Generate a time-varying DOE envelope for testing.
    
    Parameters:
    -----------
    scenario : str
        'none'         -> no constraint (p_min = -inf, p_max = +inf)
        'conservative' -> moderate export limits (80% of baseline)
        'tight'        -> strict limits (30% of baseline), worst during peak
        'rolling'      -> realistic day-ahead forecast + ±10% uncertainty
    
    base_export_limit : float
        Maximum export (negative p) the feeder can accept (kW)
        E.g., 3.0 means we won't send more than 3 kW back to grid
    
    Returns:
    --------
    doe_min, doe_max : np.ndarray of shape (T,)
        Lower and upper bounds on p_k for each interval k
        doe_min_k is the most negative p can go (export limit)
        doe_max_k is the most positive p can go (import limit, usually unbound)
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

def build_constraints(e_max, soc_init_frac=0.5, p_max=P_MAX,
                      doe_min=None, doe_max=None):
    """
    Build the stacked (A, l, u) constraint block for OSQP.
    
    Decision variable: b in R^T (battery discharge, >0 discharge, <0 charge)
    Grid flow is p = l - g - b.
    
    Constraints:
    1. |b_k| <= p_max                              (charge/discharge rate limit)
    2. 0 <= SOC_k <= e_max                         (state of charge bounds)
    3. sum(b) = 0                                  (energy conservation: start/end SOC equal)
    4. [NEW] doe_min_k <= p_k <= doe_max_k        (Dynamic Operating Envelope)
    
    For constraint 4, we express p_k = l_k - g_k - b_k, so:
        doe_min_k <= l_k - g_k - b_k <= doe_max_k
        doe_min_k - l_k + g_k <= -b_k <= doe_max_k - l_k + g_k
        l_k - g_k - doe_max_k <= b_k <= l_k - g_k - doe_min_k
    
    The DOE rows are an identity block that is ALWAYS part of A, so the
    persistent OSQP workspace keeps one sparsity pattern for the whole run
    (paper_context.md §4 invariant): with no envelope they are inactive
    (l = -inf, u = +inf); with an envelope only the bounds change, per day,
    via solver.update(l=, u=). Passing a differently-shaped A to
    solver.update() is NOT supported by osqp -- osqp 1.x silently ignores an
    `A=` keyword -- and that is exactly how the DOE constraint was a no-op
    in every result generated before 2026-08-16 (see NETWORK_AWARE_DISPATCH.md).
    """
    soc_init = soc_init_frac * e_max
    A_soc = np.tril(np.ones((T, T))) * DT
    I_T = sp.eye(T, format="csc")
    A_soc_sp = sp.csc_matrix(A_soc)
    A_eq = sp.csc_matrix(np.ones((1, T)))

    # Stack: [b bounds] [SOC bounds] [sum=0] [DOE rows on b]
    A = sp.vstack([I_T, -A_soc_sp, A_eq, I_T]).tocsc()

    l = np.hstack([
        -p_max * np.ones(T),
        -soc_init * np.ones(T),
        np.array([0.0]),
        -np.inf * np.ones(T),             # DOE rows: inactive until set
    ])
    u = np.hstack([
        p_max * np.ones(T),
        (e_max - soc_init) * np.ones(T),
        np.array([0.0]),
        np.inf * np.ones(T),
    ])

    doe_info = {
        "doe_min": doe_min,
        "doe_max": doe_max,
    }

    return A, l, u, doe_info


N_BASE_ROWS = 2 * T + 1                    # [b bounds][SOC][sum=0]


def doe_row_bounds(net, doe_min=None, doe_max=None):
    """
    Bounds for the DOE identity rows on b for one day.

    doe_min_k <= p_k <= doe_max_k with p = net - b  <=>
        net_k - doe_max_k <= b_k <= net_k - doe_min_k
    With no envelope the rows are inactive (-inf, +inf).
    """
    if doe_min is None or doe_max is None:
        return -np.inf * np.ones(T), np.inf * np.ones(T)
    return net - np.asarray(doe_max, dtype=float), \
        net - np.asarray(doe_min, dtype=float)


# ==========================================================
# MODIFIED: PERSISTENT OSQP SOLVER WITH DOE
# ==========================================================

_SOLVER_CACHE = {"solver": None, "e_max": None}


def _get_solver(e_max, doe_info=None):
    """
    Build OSQP solver. doe_info is stored but constraints are updated dynamically
    in solve_battery since DOE depends on the day.
    """
    cache = _SOLVER_CACHE
    if cache["solver"] is not None and cache["e_max"] == e_max:
        return cache["solver"]
    
    A, l, u, _ = build_constraints(e_max, doe_min=None, doe_max=None)
    P0 = sp.diags(2.0 * np.ones(T), format="csc")
    q0 = np.zeros(T)
    
    solver = osqp.OSQP()
    solver.setup(
        P=P0, q=q0, A=A, l=l, u=u,
        verbose=False,
        eps_abs=1e-6, eps_rel=1e-6,
        polish=True, warm_start=True,
    )
    cache["solver"] = solver
    cache["e_max"] = e_max
    cache["P_nnz"] = P0.nnz
    return solver


def solve_battery(load, pv, h_diag, e_max, doe_min=None, doe_max=None):
    """
    Solve the QP for a given day with optional DOE constraints.
    
    If doe_min/doe_max are provided, they constrain the grid flow p_k:
        doe_min_k <= p_k <= doe_max_k
    
    Since p_k = l_k - g_k - b_k, this becomes:
        l_k - g_k - doe_max_k <= b_k <= l_k - g_k - doe_min_k
    
    The DOE rows already exist in the workspace (build_constraints); only
    their bounds are updated here, so the sparsity pattern never changes.

    If the envelope is infeasible for this day (the battery cannot absorb /
    supply enough -- typically a sunny day whose forced charging exceeds the
    SOC headroom), OSQP reports primal infeasibility and its iterate is not
    a valid dispatch. We then re-solve WITHOUT the envelope and return that
    (SOC- and rate-feasible) dispatch; simulate_day() records the resulting
    envelope breach in doe_slack / doe_compliant. Returns (b, doe_feasible).
    """
    solver = _get_solver(e_max)
    net = load - pv                       # p when b = 0

    P_data = 2.0 * h_diag
    q = -2.0 * h_diag * net

    _, l_base, u_base, _ = build_constraints(e_max)
    l_base, u_base = l_base[:N_BASE_ROWS], u_base[:N_BASE_ROWS]

    def _solve(l_doe, u_doe):
        solver.update(Px=P_data, q=q,
                      l=np.hstack([l_base, l_doe]),
                      u=np.hstack([u_base, u_doe]))
        return solver.solve()

    have_doe = doe_min is not None and doe_max is not None
    res = _solve(*doe_row_bounds(net, doe_min, doe_max))
    doe_feasible = True
    if res.info.status_val not in (1, 2):
        if have_doe:
            # envelope cannot be met today: fall back to the unconstrained
            # dispatch and let the caller measure the breach
            doe_feasible = False
            res = _solve(*doe_row_bounds(net))
        if res.info.status_val not in (1, 2):
            logger.warning("OSQP status: %s", res.info.status)

    return res.x, doe_feasible


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
        b, _ = solve_battery(load, pv, h_vec, e_max, doe_min, doe_max)
        return base_cost - bill(load, pv, b, tariff, mode), b
    
    best_s, best_b = savings_for(h)
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
            s, b = savings_for(trial)
            if s > best_s + 1e-12:
                best_s, best_b, best_h = s, b, trial.copy()
                current = trial
                improved_this_round = True
        if not improved_this_round:
            break
    
    return best_h, best_b, best_s


# ==========================================================
# MODIFIED: SINGLE-DAY SIMULATION WITH DOE
# ==========================================================

def simulate_day(load, pv, tariff, mode, e_max=E_MAX_DEFAULT, 
                 doe_min=None, doe_max=None):
    """
    Simulate a single day with optional DOE constraints.
    
    Returns:
    --------
    savings : float
        Daily operational savings ($)
    b : ndarray
        Battery discharge profile (kW)
    p : ndarray
        Grid flow profile (kW)
    h : ndarray
        Optimized weight vector
    doe_compliant : bool
        True if solution respects all DOE bounds
    doe_slack : ndarray
        Positive values where constraint is violated (for debugging)
    """
    h, b, savings = optimise_H(load, pv, tariff, e_max, mode, doe_min, doe_max)
    p = load - pv - b
    
    # Check DOE compliance
    doe_compliant = True
    doe_slack = np.zeros(T)
    if doe_min is not None and doe_max is not None:
        # Check if p_k is within bounds (within numerical tolerance)
        tol = 1e-5
        below_min = p < (doe_min - tol)
        above_max = p > (doe_max + tol)
        
        doe_slack[below_min] = doe_min[below_min] - p[below_min]
        doe_slack[above_max] = p[above_max] - doe_max[above_max]
        
        if np.any(below_min) or np.any(above_max):
            doe_compliant = False
    
    return savings, b, p, h, doe_compliant, doe_slack


# ==========================================================
# MODIFIED: PER-CUSTOMER WORKER WITH DOE TRACKING
# ==========================================================

def _worker(args):
    """
    Worker function for multiprocessing. Now tracks DOE compliance.
    """
    (customer, days, tariff, mode, e_max,
     doe_scenario, base_export_limit) = args

    total = 0.0
    day_profiles = []
    soc_init = 0.5 * e_max

    # Generate DOE envelope (same for all days of this customer)
    doe_min, doe_max = generate_doe_envelope(doe_scenario, base_export_limit)
    
    for date, load, pv in days:
        s, b, p, _, doe_compliant, doe_slack = simulate_day(
            load, pv, tariff, mode, e_max, doe_min, doe_max)
        
        total += s
        soc = soc_init - np.cumsum(b) * DT
        
        day_profiles.append({
            "date": date,
            "load": load,
            "pv": pv,
            "battery": b,
            "grid": p,
            "soc": soc,
            "savings": s,
            "doe_compliant": doe_compliant,
            "doe_slack": doe_slack,
            "doe_slack_total": np.sum(doe_slack),
        })
    
    return customer, total, day_profiles


def run_all(day_arrays, mode, e_max=E_MAX_DEFAULT, doe_scenario="none",
            base_export_limit=3.0):
    """
    Run simulation for every customer with DOE support.

    Parameters:
    -----------
    doe_scenario : str
        'none', 'conservative', 'tight', 'rolling'
    base_export_limit : float
        Feeder export headroom (kW) the scenario scales -- see
        generate_doe_envelope(). Sweep it to ask "how tight must the
        envelope be before the network sees zero over-voltage?"
    """
    tariff = build_tariff()
    jobs = [(cust, days, tariff, mode, e_max, doe_scenario, base_export_limit)
            for cust, days in day_arrays.items()]
    n_proc = min(cpu_count(), len(jobs)) or 1

    logger.info(
        "Running %s simulations on %d cores (E_max=%.1f kWh, DOE=%s, "
        "base export limit=%.2f kW)",
        mode, n_proc, e_max, doe_scenario, base_export_limit)
    
    customers, savings = [], []
    all_profiles = {}
    doe_stats = {"compliant_days": 0, "violating_days": 0}
    
    with Pool(processes=n_proc) as pool:
        for cust, total, profiles in pool.imap_unordered(_worker, jobs, chunksize=1):
            compliant = sum(1 for p in profiles if p["doe_compliant"])
            violating = len(profiles) - compliant
            doe_stats["compliant_days"] += compliant
            doe_stats["violating_days"] += violating
            
            logger.info("Customer %s: $%.2f/yr (%d/%d days DOE-compliant)",
                       cust, total, compliant, len(profiles))
            customers.append(cust)
            savings.append(total)
            all_profiles[cust] = profiles
    
    order = np.argsort(customers)
    
    logger.info("=== DOE Compliance Summary ===")
    logger.info("  Total days simulated: %d", 
               doe_stats["compliant_days"] + doe_stats["violating_days"])
    logger.info("  DOE-compliant days:   %d (%.1f%%)",
               doe_stats["compliant_days"],
               100.0 * doe_stats["compliant_days"] / 
               (doe_stats["compliant_days"] + doe_stats["violating_days"] + 1e-6))
    logger.info("  DOE-violating days:   %d", doe_stats["violating_days"])
    
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

def save_profiles(all_profiles, mode, doe_scenario="none", out_dir="profiles"):
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
            
            fname = os.path.join(cust_dir, f"{date}.csv")
            np.savetxt(fname, grid, fmt="%.6f", delimiter=",")
            
            for k in range(T):
                rows.append({
                    "customer": int(cust),
                    "date": date,
                    "interval": int(intervals[k]),
                    "hour": float(hours[k]),
                    "load_kw": float(load[k]),
                    "pv_kw": float(pv[k]),
                    "battery_kw": float(batt[k]),
                    "grid_kw": float(grid[k]),
                    "soc_kwh": float(soc[k]),
                    "daily_savings": float(sav),
                    "doe_compliant": int(doe_compliant),
                    "doe_slack_kw": float(doe_slack[k]),
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
        s, b, p, _, compliant, slack = simulate_day(
            found_load, found_pv, tariff, mode, e_max, doe_min, doe_max)
        
        ax_p.fill_between(hours, doe_min, doe_max, alpha=0.2, 
                          color="green", label="DOE envelope")
        ax_p.plot(hours, p, color="steelblue", marker="o", 
                 label=f"grid flow (${s:.2f}/day)", linewidth=2)
        ax_p.axhline(0, color="black", lw=0.5, linestyle="--")
        ax_p.set_ylabel("Power (kW)")
        ax_p.set_title(
            f"Scenario: {scenario.upper()} | Compliant: {compliant} | "
            f"Slack: {np.sum(slack):.3f} kWh")
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
    CSVs (profiles/<mode>_doe_<label>.csv, the same long format
    elermorevale_openDSS.py --profiles reads), and optionally the
    scenario-comparison table.

    Defaults reproduce the original workflow: fit mode, scenarios
    none / conservative / tight at a 3 kW feeder export headroom, followed
    by the comparison table.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="DOE-constrained QP battery dispatch (extension of "
                    "osqp_daily.py). Writes profiles/<mode>_doe_<label>.csv "
                    "for each scenario, consumable by the network scripts.")
    parser.add_argument("--data", default="data.csv",
                        help="Ausgrid CSV (default: data.csv)")
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
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the none/conservative/tight comparison "
                             "table (which re-runs all three scenarios)")
    args = parser.parse_args()

    df_raw = load_dataset(args.data)
    df_clean = clean_dataset(df_raw)
    day_arrays = extract_day_arrays(df_clean)
    logger.info("Extracted %d customers", len(day_arrays))

    label_with_cap = len(args.export_limit) > 1
    baseline_savings = None
    for cap in args.export_limit:
        for scenario in args.scenarios:
            label = f"{scenario}_cap{cap:g}" if label_with_cap else scenario
            logger.info("\n" + "=" * 60)
            logger.info("SCENARIO %s (mode=%s, base export limit %.2f kW)",
                        label.upper(), args.mode, cap)
            logger.info("=" * 60)
            _, savings, profiles = run_all(
                day_arrays, mode=args.mode, doe_scenario=scenario,
                base_export_limit=cap)
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
        df_comp.to_csv("doe_scenario_comparison.csv", index=False)


if __name__ == "__main__":
    main()
