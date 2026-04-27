"""
osqp_daily_openDSS.py
================

Faithful re-implementation of the QP-based residential battery scheduling
algorithm of Ratnam, Weller & Kellett (Renewable Energy 75, 2015), applied
to the Ausgrid "Solar home electricity data" (2010-2013).

Relative to osqp_daily.py this file:

  * Matches the paper's heuristic for the weighting matrix H (Section 5)
    more faithfully: base-line H0 is built from the saturated ratio
    h_tilde_k / h_plus, and the greedy search doubles the currently-largest
    tier of weights (not just the max-tariff indices) until no further
    improvement is obtained, with weights capped at h_bar = 1000.

  * Implements both paper metering topologies correctly:
        - Topology 1 (gross FiT): separate meters for PV generation and
          for load+battery; PV credited at the FiT rate on the raw generation
          profile g, load billed at TOU on (l - b).
        - Topology 2 (net metering): a single bi-directional meter on p;
          imports billed at TOU, exports credited at the net-metering rate.

  * Uses a single persistent OSQP workspace per worker process. Because
    the constraint matrices A_batt, l_batt, u_batt are problem-invariant
    and the Hessian P = 2H is diagonal, we set up once and then use
    osqp.update(Px=..., q=...) on every subsequent day. This is roughly
    an order of magnitude faster than tearing down and re-creating the
    solver each call.

  * Pre-extracts each (customer, day) slice into plain numpy arrays once,
    outside the hot loop, so the optimisation never touches pandas.

  * Reproduces the figures actually shown in the paper: Fig. 2 (example
    load/PV/battery/grid profiles), Fig. 5 (SOC trajectories for customers
    75 and 200 on the representative days), Fig. 6 (daily savings
    distribution), Fig. 7 (annual savings histogram by topology) and
    Fig. 8 (annual savings vs battery capacity sweep).

Differences from osqp_daily.py worth noting in review comments:
  - osqp_daily.py uses FEED_IN = 0.08 for its "net" mode; the paper uses
    0.40 $/kWh for both the gross FiT (topology 1) and the net-metering
    export credit (topology 2). v2 uses the paper's values by default.
  - osqp_daily.py's FiT billing credits pv + battery_export at the FiT
    rate. The paper's topology 1 has separate gross meters so PV is
    credited exactly once, on the raw generation g; any battery export
    through M2 is treated as load billing at TOU (and cannot be negative
    since M2 is gross). v2 implements this.
  - osqp_daily.py calls figure7 with arguments in an order that swaps the
    Net and FiT subplot labels. Fixed here.
  - osqp_daily.py's build_H0 returns the tariff directly. The paper
    normalises by h_plus and caps at h_bar. Fixed here.
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
# CONSTANTS (match paper Section 6.1 where applicable)
# ==========================================================

DT = 0.5            # hours per interval
T = 48              # intervals per day
P_MAX = 5.0         # kW charge/discharge limit (paper uses 5 kW)
E_MAX_DEFAULT = 10.0  # kWh (paper uses 10 kWh nominal)

FIT_RATE = 0.40     # $/kWh   (paper: hc(M1) = hc(M3) = 0.40)
H_BAR = 1000.0      # cap on heuristic weights (paper Section 5)
HEURISTIC_MAX_ITERS = 20  # safety cap on the greedy loop


# ==========================================================
# DATA LOADING + CLEANING (same clean-customer list as v1,
# which follows the paper's companion dataset paper)
# ==========================================================

CLEAN_CUSTOMER_IDS = [
    2, 13, 14, 20, 33, 35, 38, 39, 56, 69, 73, 74, 75, 82, 87, 88,
    101, 104, 106, 109, 110, 119, 124, 130, 137, 141, 144, 152, 153,
    157, 161, 169, 176, 184, 188, 189, 193, 200, 201, 202, 204, 206,
    207, 210, 211, 212, 214, 218, 244, 246, 253, 256, 273, 276, 297,
]


def load_dataset(path):
    df = pd.read_csv(path, skiprows=1)
    time_cols = df.columns[5:]
    df_long = df.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode",
                 "Consumption Category", "date"],
        value_vars=time_cols,
        var_name="time",
        value_name="energy",
    )
    df_long["time"] = pd.to_datetime(df_long["time"], format="%H:%M")
    pivot = df_long.pivot_table(
        index=["Customer", "date", "time"],
        columns="Consumption Category",
        values="energy",
    ).reset_index()
    pivot = pivot.sort_values(["Customer", "date", "time"])
    for col in ("GC", "CL", "GG"):
        if col not in pivot.columns:
            pivot[col] = 0.0
        pivot[col] = pivot[col].fillna(0.0)
    pivot["load"] = pivot["GC"] + pivot["CL"]
    pivot["pv"] = pivot["GG"]
    return pivot


def clean_dataset(df):
    df_clean = df[df["Customer"].isin(CLEAN_CUSTOMER_IDS)]
    good_days = []
    removed = 0
    for _, day in df_clean.groupby(["Customer", "date"], sort=False):
        if len(day) == T:
            good_days.append(day)
        else:
            removed += 1
    if not good_days:
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(good_days, ignore_index=True)
    logger.info("Customers after cleaning: %d", out["Customer"].nunique())
    logger.info("Days removed (incomplete): %d", removed)
    return out


def extract_day_arrays(df):
    """
    Collapse the long-format dataframe into a dict
        customer_id -> list of (date, load_array, pv_array)
    where each array has length T. Avoids pandas overhead in the hot loop.
    """
    out = {}
    df = df.sort_values(["Customer", "date", "time"])
    for cust, cust_df in df.groupby("Customer", sort=True):
        days = []
        for date, day in cust_df.groupby("date", sort=True):
            if len(day) != T:
                continue
            days.append((date,
                         day["load"].to_numpy(dtype=np.float64),
                         day["pv"].to_numpy(dtype=np.float64)))
        out[int(cust)] = days
    return out


# ==========================================================
# BATTERY CONSTRAINT MATRICES (built once per E_MAX)
# ==========================================================

def build_constraints(e_max, soc_init_frac=0.5, p_max=P_MAX):
    """
    Build the stacked (A, l, u) constraint block for OSQP with
    decision variable b in R^T (paper: beta). Sign convention:
    b > 0 discharging, b < 0 charging, matching paper eq. (1)
    via p = l - g - b.
    """
    soc_init = soc_init_frac * e_max
    A_soc = np.tril(np.ones((T, T))) * DT  # integrates b forward
    I_T = sp.eye(T, format="csc")
    A_soc_sp = sp.csc_matrix(A_soc)
    A_eq = sp.csc_matrix(np.ones((1, T)))

    A = sp.vstack([I_T, -A_soc_sp, A_eq]).tocsc()
    # b in [-p_max, p_max]
    # -A_soc*b in [-soc_init, e_max - soc_init]  <=> SOC in [0, e_max]
    # 1^T b = 0
    l = np.hstack([
        -p_max * np.ones(T),
        -soc_init * np.ones(T),
        np.array([0.0]),
    ])
    u = np.hstack([
        p_max * np.ones(T),
        (e_max - soc_init) * np.ones(T),
        np.array([0.0]),
    ])
    # Lower power-limit duplication removed: I_T with symmetric bounds
    # already does both sides in one block, so we only need one copy.
    return A, l, u


# ==========================================================
# PERSISTENT PER-WORKER OSQP SOLVER
# ==========================================================

_SOLVER_CACHE = {"solver": None, "e_max": None}


def _get_solver(e_max):
    cache = _SOLVER_CACHE
    if cache["solver"] is not None and cache["e_max"] == e_max:
        return cache["solver"]

    A, l, u = build_constraints(e_max)
    # Placeholder P and q; will be updated on every solve.
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


def solve_battery(load, pv, h_diag, e_max):
    """
    Solve the QP for a given day. h_diag is the length-T vector of
    diagonal entries of H. Uses osqp.update() for speed.
    """
    solver = _get_solver(e_max)
    net = load - pv                       # p when b = 0
    # Objective: sum_k h_k * (net_k - b_k)^2
    #   = b^T diag(h) b - 2 (h*net)^T b + const
    # OSQP minimises 0.5 x^T P x + q^T x, so P = 2 diag(h), q = -2 h*net.
    P_data = 2.0 * h_diag
    q = -2.0 * h_diag * net
    solver.update(Px=P_data, q=q)
    res = solver.solve()
    if res.info.status_val not in (1, 2):  # solved / solved_inaccurate
        logger.warning("OSQP status: %s", res.info.status)
    return res.x


# ==========================================================
# TARIFF (paper Section 6.1)
# ==========================================================

def build_tariff():
    """Paper's TOU: off-peak 0.03, shoulder 0.06, peak 0.30 $/kWh."""
    tariff = np.zeros(T)
    tariff[0:14] = 0.03   # 00:00 - 07:00
    tariff[44:48] = 0.03  # 22:00 - 24:00
    tariff[14:28] = 0.06  # 07:00 - 14:00
    tariff[40:44] = 0.06  # 20:00 - 22:00
    tariff[28:40] = 0.30  # 14:00 - 20:00
    return tariff


# ==========================================================
# BILLING: two metering topologies from the paper
# ==========================================================

def bill_topology1(load, pv, b, tariff, fit=FIT_RATE):
    """
    Gross FiT. Paper eq. (21) with hb(M1)=0, hc(M2)=0.
      - PV credited on raw g at the flat FiT rate.
      - Load+battery billed on max(l - b, 0) at TOU (M2 is gross).
    """
    flow_m2 = np.maximum(load - b, 0.0)  # M2 is unidirectional
    return np.sum(flow_m2 * tariff * DT) - np.sum(pv * fit * DT)


def bill_topology2(load, pv, b, tariff, net_credit=FIT_RATE):
    """
    Net metering. Paper eq. (25).
      - One bi-directional meter on p = l - g - b.
      - Imports (p>0) billed at TOU, exports (p<0) credited at net_credit.
    """
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
# HEURISTIC H (paper Section 5, Algorithm 1)
# ==========================================================

def build_H0_diag(tariff):
    """
    Base-line H0: h_tilde_k / h_plus, saturated to [1, H_BAR].
    Paper eqs. (36)-(38).
    """
    h_tilde = tariff.copy()
    h_plus = np.min(h_tilde[h_tilde > 0]) if np.any(h_tilde > 0) else 1.0
    h0 = h_tilde / h_plus
    return np.clip(h0, 1.0, H_BAR)


def optimise_H(load, pv, tariff, e_max, mode):
    """
    Paper heuristic: repeatedly double the currently-largest tier of
    weights in H0 while daily savings keep increasing. Weights are
    saturated at H_BAR.
    """
    h = build_H0_diag(tariff)
    base_cost = bill(load, pv, np.zeros(T), tariff, mode)

    def savings_for(h_vec):
        b = solve_battery(load, pv, h_vec, e_max)
        return base_cost - bill(load, pv, b, tariff, mode), b

    best_s, best_b = savings_for(h)
    best_h = h.copy()

    # Tiers of indices, largest weight first. Indices are frozen at the
    # initial H0 grouping, matching the paper's "doubles weights in H0
    # progressively, from the largest to the smallest element".
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
# SINGLE-DAY SIMULATION
# ==========================================================

def simulate_day(load, pv, tariff, mode, e_max=E_MAX_DEFAULT):
    h, b, savings = optimise_H(load, pv, tariff, e_max, mode)
    p = load - pv - b
    return savings, b, p, h


# ==========================================================
# PER-CUSTOMER ANNUAL SAVINGS (multiprocessing worker)
# ==========================================================

def _worker(args):
    customer, days, tariff, mode, e_max = args
    total = 0.0
    day_profiles = []
    soc_init = 0.5 * e_max
    for date, load, pv in days:
        s, b, p, _ = simulate_day(load, pv, tariff, mode, e_max)
        total += s
        soc = soc_init - np.cumsum(b) * DT
        day_profiles.append({
            "date": date,
            "load": load,          # l_k  (kW)
            "pv": pv,              # g_k  (kW)
            "battery": b,          # b_k  (kW, +discharge/−charge)
            "grid": p,             # p_k  (kW, +import/−export)
            "soc": soc,            # SOC at end of each interval (kWh)
            "savings": s,          # daily savings ($)
        })
    return customer, total, day_profiles


def run_all(day_arrays, mode, e_max=E_MAX_DEFAULT):
    """
    Run simulation for every customer. Returns:
        customers   – sorted array of customer IDs
        savings     – corresponding annual savings ($)
        all_profiles – dict  {customer_id: [day_profile_dict, ...]}
            Each day_profile_dict has keys:
                date, load, pv, battery, grid, soc, savings
            All arrays are length-48 numpy float64.
    """
    tariff = build_tariff()
    jobs = [(cust, days, tariff, mode, e_max)
            for cust, days in day_arrays.items()]
    n_proc = min(cpu_count(), len(jobs)) or 1
    logger.info("Running %s simulations on %d cores (E_max=%.1f kWh)",
                mode, n_proc, e_max)

    customers, savings = [], []
    all_profiles = {}
    with Pool(processes=n_proc) as pool:
        for cust, total, profiles in pool.imap_unordered(
                _worker, jobs, chunksize=1):
            logger.info("Customer %s: $%.2f/yr", cust, total)
            customers.append(cust)
            savings.append(total)
            all_profiles[cust] = profiles
    order = np.argsort(customers)
    return (np.array(customers)[order],
            np.array(savings)[order],
            all_profiles)


def save_profiles(all_profiles, mode, out_dir="profiles"):
    """
    Save half-hourly simulation results to disk.

    Produces two outputs:
      1. A single long-format CSV:  profiles/{mode}_profiles.csv
         with columns: customer, date, interval, hour, load_kw, pv_kw,
                       battery_kw, grid_kw, soc_kwh, daily_savings
      2. Per-customer CSVs in profiles/{mode}/cust_{id}/:
         one file per day, containing only the 48-row grid profile p_k.
         These are directly consumable as OpenDSS LoadShape mult files.

    The long CSV is useful for analysis; the per-customer files are
    useful for driving OpenDSS time-series simulations.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    per_day_dir = os.path.join(out_dir, mode)
    os.makedirs(per_day_dir, exist_ok=True)

    intervals = np.arange(1, T + 1)
    hours = np.arange(T) * DT  # 0.0, 0.5, ... 23.5

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

            # Per-day OpenDSS shape file: just the grid profile
            fname = os.path.join(cust_dir, f"{date}.csv")
            np.savetxt(fname, grid, fmt="%.6f", delimiter=",")

            # Accumulate rows for the long CSV
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
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{mode}_profiles.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)
    logger.info("Per-day grid shapes in %s/", per_day_dir)
    return csv_path


# ==========================================================
# FIGURES
# ==========================================================

def _find_day(day_arrays, customer, target_date_str):
    """Return (load, pv) for a (customer, date) pair, or None."""
    for date, load, pv in day_arrays.get(customer, []):
        if str(date) == target_date_str or str(date).startswith(target_date_str):
            return load, pv
    return None


def _soc_from_b(b, e_max, soc_init_frac=0.5):
    c0 = soc_init_frac * e_max
    return c0 - np.cumsum(b) * DT


def figure2_example_day(day_arrays, customer=75, date_str="2011-01-09",
                        e_max=E_MAX_DEFAULT):
    """Paper Fig. 2: load/PV/battery/grid profiles for a single day."""
    found = _find_day(day_arrays, customer, date_str)
    if found is None:
        logger.warning("Fig 2: day not found (cust=%s, date=%s)",
                       customer, date_str)
        return
    load, pv = found
    tariff = build_tariff()
    _, b, p, _ = simulate_day(load, pv, tariff, mode="fit", e_max=e_max)

    hours = np.arange(T) * DT
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(hours, load, label="load l", color="black")
    axes[0].plot(hours, pv, label="PV g", color="orange")
    axes[0].set_ylabel("kW")
    axes[0].set_title(
        f"Fig. 2(a): load & PV, customer {customer}, {date_str}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hours, p, label="grid p", color="steelblue")
    axes[1].plot(hours, b, label="battery b", color="crimson")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("kW")
    axes[1].set_title(f"Fig. 2(c)/(d): grid & battery, E={e_max} kWh")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def figure5_soc(day_arrays, e_max=E_MAX_DEFAULT):
    """Paper Fig. 5: SOC for customer 75 on 2011-01-09 and
    customer 200 on 2010-07-05."""
    tariff = build_tariff()
    specs = [(75, "2011-01-09"), (200, "2010-07-05")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hours = np.arange(T) * DT
    for ax, (cust, date_str) in zip(axes, specs):
        found = _find_day(day_arrays, cust, date_str)
        if found is None:
            ax.set_title(f"Customer {cust}: day {date_str} not found")
            continue
        load, pv = found
        _, b, _, _ = simulate_day(load, pv, tariff, mode="fit", e_max=e_max)
        soc = _soc_from_b(b, e_max)
        ax.plot(hours, soc, color="steelblue")
        ax.set_title(f"Fig. 5: Customer {cust} SOC ({date_str})")
        ax.set_xlabel("Hour")
        ax.set_ylabel("SOC (kWh)")
        ax.set_ylim(0, e_max)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def figure6_daily_savings(day_arrays, customers=(75, 200), mode="fit",
                          e_max=E_MAX_DEFAULT, bin_width=0.25):
    tariff = build_tariff()
    fig, axes = plt.subplots(len(customers), 1,
                             figsize=(8, 3.5 * len(customers)), sharex=True)
    if len(customers) == 1:
        axes = [axes]
    for ax, cust in zip(axes, customers):
        daily = []
        for _, load, pv in day_arrays.get(cust, []):
            s, _, _, _ = simulate_day(load, pv, tariff, mode, e_max)
            daily.append(s)
        if not daily:
            ax.set_title(f"Customer {cust}: no data")
            continue
        daily = np.array(daily)
        lo = np.floor(daily.min() / bin_width) * bin_width
        hi = np.ceil(daily.max() / bin_width) * bin_width
        edges = np.arange(lo, hi + bin_width, bin_width)
        ax.hist(daily, bins=edges, color="gray", edgecolor="black")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_ylabel("Days")
        ax.set_title(
            f"Fig. 6: Customer {cust} daily savings ({mode.upper()}), "
            f"total ${daily.sum():.0f}/yr")
        ax.grid(axis="y", alpha=0.3)
        xlim = max(abs(edges[0]), abs(edges[-1]))
        ax.set_xlim(-xlim, xlim)
    axes[-1].set_xlabel("Daily savings ($)")
    plt.tight_layout()
    plt.show()


def figure7_annual_savings(customers_fit, savings_fit,
                           customers_net, savings_net, bin_width=50):
    """Paper Fig. 7: two-panel histogram, topology 1 (FiT) on top."""
    x_min = min(savings_fit.min(), savings_net.min())
    x_max = max(savings_fit.max(), savings_net.max())
    x_min = np.floor(x_min / bin_width) * bin_width
    x_max = np.ceil(x_max / bin_width) * bin_width
    edges = np.arange(x_min, x_max + bin_width, bin_width)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].hist(savings_fit, bins=edges, color="gray", edgecolor="black")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Fig. 7(a): Annual savings, metering topology 1 (FiT)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(savings_net, bins=edges, color="gray", edgecolor="black")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Fig. 7(b): Annual savings, metering topology 2 (Net)")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(bin_width))
        ax.set_xlim(x_min, x_max)
        ax.axvline(0, color="black", lw=0.8)
    axes[-1].set_xlabel("$/yr")
    plt.tight_layout()
    plt.show()


def figure8_capacity_sweep(day_arrays, customers=(75, 200), mode="fit",
                           capacities=(0.1, 1, 2, 4, 6, 8, 10, 15, 20, 30)):
    """Paper Fig. 8: annual savings vs battery capacity."""
    tariff = build_tariff()
    fig, ax = plt.subplots(figsize=(8, 5))
    for cust in customers:
        totals = []
        for cap in capacities:
            # Reset per-capacity solver
            _SOLVER_CACHE["solver"] = None
            _SOLVER_CACHE["e_max"] = None
            total = 0.0
            for _, load, pv in day_arrays.get(cust, []):
                s, _, _, _ = simulate_day(load, pv, tariff, mode, cap)
                total += s
            totals.append(total)
        ax.plot(capacities, totals, marker="o", label=f"Customer {cust}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel("Annual savings ($/yr)")
    ax.set_title(f"Fig. 8: annual savings vs capacity ({mode.upper()})")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():
    df_raw = load_dataset("data.csv")
    df_clean = clean_dataset(df_raw)
    day_arrays = extract_day_arrays(df_clean)
    logger.info("Extracted %d customers", len(day_arrays))

    # Annual savings for both topologies
    customers_fit, savings_fit, profiles_fit = run_all(day_arrays, mode="fit")
    customers_net, savings_net, profiles_net = run_all(day_arrays, mode="net")

    logger.info("Mean annual savings (FiT, topology 1): $%.2f",
                np.mean(savings_fit))
    logger.info("Mean annual savings (Net, topology 2): $%.2f",
                np.mean(savings_net))

    # Save half-hourly profiles to disk
    save_profiles(profiles_fit, mode="fit")
    save_profiles(profiles_net, mode="net")

    # Paper figures
    figure2_example_day(day_arrays, customer=75, date_str="2011-01-09")
    figure5_soc(day_arrays)
    figure6_daily_savings(day_arrays, customers=(75, 200), mode="fit")
    figure7_annual_savings(customers_fit, savings_fit,
                           customers_net, savings_net)
    figure8_capacity_sweep(day_arrays, customers=(75, 200), mode="fit")


if __name__ == "__main__":
    main()