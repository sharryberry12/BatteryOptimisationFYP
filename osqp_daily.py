import logging
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

"""
Battery dispatch and savings simulation for household load and PV data.

Now supports:
- Net metering
- Feed-in tariff (gross metering)

Default behaviour remains unchanged (net metering).
"""

# ==========================================================
# CONSTANTS
# ==========================================================

DT = 0.5
T = 48

P_MAX = 5
E_MAX = 13.5

SOC_INIT = 0.5 * E_MAX

FEED_IN = 0.08

# ==========================================================
# DATA LOADING + CLEANING
# ==========================================================

def load_dataset(path):
    df = pd.read_csv(path, skiprows=1)

    time_cols = df.columns[5:]
    df_long = df.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode",
                 "Consumption Category", "date"],
        value_vars=time_cols,
        var_name="time",
        value_name="energy"
    )

    df_long["time"] = pd.to_datetime(df_long["time"], format="%H:%M")

    pivot = df_long.pivot_table(
        index=["Customer", "date", "time"],
        columns="Consumption Category",
        values="energy"
    ).reset_index()

    pivot = pivot.sort_values(["Customer", "date", "time"])
    pivot["GC"] = pivot["GC"].fillna(0)
    pivot["CL"] = pivot["CL"].fillna(0)
    pivot["GG"] = pivot["GG"].fillna(0)

    pivot["load"] = pivot["GC"] + pivot["CL"]
    pivot["pv"] = pivot["GG"]

    return pivot


def clean_dataset(df):
    clean_customer_ids = [
        2, 13, 14, 20, 33, 35, 38, 39, 56, 69, 73, 74, 75, 82, 87, 88,
        101, 104, 106, 109, 110, 119, 124, 130, 137, 141, 144, 152, 153,
        157, 161, 169, 176, 184, 188, 189, 193, 200, 201, 202, 204, 206,
        207, 210, 211, 212, 214, 218, 244, 246, 253, 256, 273, 276, 297
    ]

    df_clean = df[df["Customer"].isin(clean_customer_ids)]

    cleaned_days = []
    removed_days = 0

    for (cust, date), day in df_clean.groupby(["Customer", "date"]):
        if len(day) != 48:
            removed_days += 1
            continue
        cleaned_days.append(day)

    if len(cleaned_days) == 0:
        return pd.DataFrame(columns=df.columns)

    df_clean = pd.concat(cleaned_days)

    logger.info("Customers before cleaning: %s", df["Customer"].nunique())
    logger.info("Customers after cleaning: %s", df_clean["Customer"].nunique())
    logger.info("Days removed (incomplete): %s", removed_days)

    return df_clean

# ==========================================================
# SOC MATRIX
# ==========================================================

def build_soc_matrix():
    A = np.tril(np.ones((T, T))) * DT
    return sp.csc_matrix(A)

A_soc = build_soc_matrix()

I_T = sp.eye(T, format='csc')
A_eq = sp.csc_matrix(np.ones((1, T)))

A_batt = sp.vstack([
    I_T,
    -I_T,
    -A_soc,
    A_soc,
    A_eq
]).tocsc()

l_batt = np.hstack([
    -P_MAX * np.ones(T),
    -P_MAX * np.ones(T),
    -SOC_INIT * np.ones(T),
    -(E_MAX - SOC_INIT) * np.ones(T),
    np.array([0.0])
])

u_batt = np.hstack([
    P_MAX * np.ones(T),
    P_MAX * np.ones(T),
    (E_MAX - SOC_INIT) * np.ones(T),
    SOC_INIT * np.ones(T),
    np.array([0.0])
])

_cached_solver = None
_cached_H_diag = None

# ==========================================================
# BILLING (NEW: supports both modes)
# ==========================================================

def bill(p, load, pv, b, tariff, mode="net"):

    if mode == "net":
        import_e = np.maximum(p, 0)
        export_e = np.maximum(-p, 0)
        return np.sum(import_e * tariff * DT) - np.sum(export_e * FEED_IN * DT)

    elif mode == "fit":
        grid_import = np.maximum(load - b, 0)
        grid_export = pv + np.maximum(-(load - b), 0)
        return np.sum(grid_import * tariff * DT) - np.sum(grid_export * FEED_IN * DT)

    else:
        raise ValueError("Unknown metering mode")

# ==========================================================
# BATTERY OPTIMISATION
# ==========================================================

def solve_battery(load, pv, H):

    global _cached_solver, _cached_H_diag

    net = load - pv
    q = -2 * H.dot(net)

    if _cached_solver is None or not np.array_equal(_cached_H_diag, np.diag(H)):
        P = sp.csc_matrix(2 * H)
        _cached_solver = osqp.OSQP()
        _cached_solver.setup(P, q, A_batt, l_batt, u_batt, verbose=False)
        _cached_H_diag = np.diag(H).copy()
    else:
        _cached_solver.update(q=q)

    res = _cached_solver.solve()
    return res.x

# ==========================================================
# HEURISTIC H
# ==========================================================

def build_H0(tariff):
    return tariff.copy()

def optimise_H(load, pv, tariff):

    weights = build_H0(tariff)
    best_savings = -np.inf
    best_H = np.diag(weights)

    for _ in range(10):
        H = np.diag(weights)
        b = solve_battery(load, pv, H)
        p = load - pv - b

        s = bill(load-pv, load, pv, np.zeros_like(b), tariff) - \
            bill(p, load, pv, b, tariff)

        if s > best_savings:
            best_savings = s
            best_H = H.copy()
            weights[tariff == np.max(tariff)] *= 2
        else:
            break

    return best_H

# ==========================================================
# SIMULATION (UPDATED, backward compatible)
# ==========================================================

def simulate_day(load, pv, tariff, mode="net"):

    H = optimise_H(load, pv, tariff)
    b = solve_battery(load, pv, H)
    p = load - pv - b

    base = bill(load-pv, load, pv, np.zeros_like(b), tariff, mode)
    new = bill(p, load, pv, b, tariff, mode)

    savings = base - new

    return savings, b, p

# ==========================================================
# FIGURE 6 DAILY COSTS 
# ==========================================================

def figure6_daily_costs(df, customers=[75, 200], mode="fit", bin_width=0.25, bins=20):
    if mode not in {"net", "fit"}:
        raise ValueError("mode must be 'net' or 'fit'")

    tariff = build_tariff()

    fig, axes = plt.subplots(len(customers), 1, figsize=(8, 4 * len(customers)), sharex=True)
    if len(customers) == 1:
        axes = [axes]

    for ax, customer in zip(axes, customers):
        data = df[df["Customer"] == customer]
        daily_savings = []

        for d in data["date"].unique():
            day = data[data["date"] == d]
            if len(day) != 48:
                continue

            load = day["load"].values
            pv = day["pv"].values
            s, _, _ = simulate_day(load, pv, tariff, mode)
            daily_savings.append(s)

        if len(daily_savings) == 0:
            counts, edges = np.histogram([], bins=bins)
        elif bin_width is not None:
            low = np.floor(min(daily_savings) / bin_width) * bin_width
            high = np.ceil(max(daily_savings) / bin_width) * bin_width
            edges = np.arange(low, high + bin_width, bin_width)
            counts, edges = np.histogram(daily_savings, bins=edges)
        else:
            counts, edges = np.histogram(daily_savings, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax.bar(centers, counts, width=edges[1] - edges[0], color="gray", edgecolor="black")
        ax.set_ylabel("Days")
        ax.set_title(f"Distribution of Daily Savings for Customer {customer} ({mode.upper()})")
        ax.grid(axis="y", alpha=0.3)

        x_limit = max(abs(edges[0]), abs(edges[-1]))
        ax.set_xlim(-x_limit, x_limit)

        y_max = int(np.ceil(counts.max() / 50.0) * 50) if len(counts) > 0 else 50
        ax.set_ylim(0, max(y_max, 50))
        ax.set_yticks(np.arange(0, max(y_max, 50) + 1, 50))
        ax.axvline(0, color="black", linewidth=0.8)

    axes[-1].set_xlabel("Daily savings ($)")
    plt.tight_layout()
    plt.show()

# ==========================================================
# NEW: FIGURE COMPARISON BETWEEN MODES
# ==========================================================

def figure_compare_modes(df, customer=75):

    tariff = build_tariff()

    plt.figure()

    for mode in ["net", "fit"]:

        data = df[df["Customer"] == customer]
        savings = []

        for d in data["date"].unique():
            day = data[data["date"] == d]
            if len(day) != 48:
                continue

            load = day["load"].values
            pv = day["pv"].values

            s, _, _ = simulate_day(load, pv, tariff, mode)
            savings.append(s)

        plt.hist(savings, bins=20, alpha=0.5, label=mode)

    plt.legend()
    plt.title(f"Customer {customer}: Net vs FiT")
    plt.xlabel("Daily savings ($)")
    plt.ylabel("Days")
    plt.show()

# ==========================================================
# FIGURE 7: ANNUAL SAVINGS BY METERING TOPOLOGY
# ==========================================================

def figure7_annual_savings(customers_fit, savings_fit, customers_net, savings_net, bin_width=50):
    # Compare annual savings distributions for FiT (topology 1) vs Net (topology 2).
    # Use customer lists to determine the y-axis range and keep both histograms on the same bin edges.

    max_customers = max(len(customers_fit), len(customers_net))

    x_min = min(np.min(savings_fit) if len(savings_fit) else 0, np.min(savings_net) if len(savings_net) else 0)
    x_max = max(np.max(savings_fit) if len(savings_fit) else 0, np.max(savings_net) if len(savings_net) else 0)
    x_min = np.floor(x_min / bin_width) * bin_width
    x_max = np.ceil(x_max / bin_width) * bin_width
    edges = np.arange(x_min, x_max + bin_width, bin_width)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].hist(savings_fit, bins=edges, color="gray", edgecolor="black")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Annual savings given metering topology 1")
    axes[0].set_ylim(0, 20)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(savings_net, bins=edges, color="gray", edgecolor="black")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Annual savings given metering topology 2")
    axes[1].set_ylim(0, 20)
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(bin_width))
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel("$ / yr")
    plt.tight_layout()
    plt.show()

# ==========================================================
# TARIFF
# ==========================================================

def build_tariff():

    tariff = np.zeros(T)

    tariff[0:14] = 0.03
    tariff[44:48] = 0.03

    tariff[14:28] = 0.06
    tariff[40:44] = 0.06

    tariff[28:40] = 0.30

    return tariff

# ==========================================================
# RUN ALL CUSTOMERS (UPDATED)
# ==========================================================

def _customer_annual_savings(args):
    customer, data, tariff, mode = args
    total = 0

    for d in data["date"].unique():
        day = data[data["date"] == d]
        if len(day) != 48:
            continue

        load = day["load"].values
        pv = day["pv"].values

        s, _, _ = simulate_day(load, pv, tariff, mode)
        total += s

    return customer, total

def run_all(df, mode="fit"):

    tariff = build_tariff()

    df = df.sort_values(["Customer", "date"])
    groups = [(customer, group, tariff, mode)
              for customer, group in df.groupby("Customer", sort=True)]

    processes = min(cpu_count(), len(groups)) or 1
    logger.info("Running (%s) simulations on %s cores", mode, processes)

    customers = []
    savings = []

    with Pool(processes=processes) as pool:
        for customer, total in pool.imap_unordered(_customer_annual_savings, groups, chunksize=1):
            logger.info("Customer %s completed: annual savings $%.2f", customer, total)
            customers.append(customer)
            savings.append(total)

    return np.array(customers), np.array(savings)

# ==========================================================
# MAIN
# ==========================================================

def main():

    df_raw = load_dataset("data.csv")
    df_clean = clean_dataset(df_raw)

    customers_net, savings_net = run_all(df_clean, mode="net")
    customers_fit, savings_fit = run_all(df_clean, mode="fit")

    logger.info("Avg savings (Net): $%.2f", np.mean(savings_net))
    logger.info("Avg savings (FiT): $%.2f", np.mean(savings_fit))

    figure6_daily_costs(df_clean, customers=[75, 200])
    figure7_annual_savings(customers_fit, savings_fit, customers_net, savings_net, bin_width=50)

    # Optional comparison plot
    figure_compare_modes(df_clean, customer=75)


if __name__ == "__main__":
    main()