import logging
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==========================================================
# CONSTANTS
# ==========================================================

DT = 0.5  # hours per interval
T = 48  # intervals per day (48 half-hour steps)

P_MAX = 5  # maximum charge/discharge power (kW)
E_MAX = 13.5  # battery capacity (kWh)

SOC_INIT = 0.5 * E_MAX  # initial SOC (kWh)

FEED_IN = 0.08  # export tariff ($/kWh)


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(path):

    df = pd.read_csv(path, skiprows=1)

    time_cols = df.columns[5:]

    # Reshape from wide to long format
    df_long = df.melt(
        id_vars=["Customer","Generator Capacity","Postcode",
                 "Consumption Category","date"],
        value_vars=time_cols,
        var_name="time",
        value_name="energy"
    )

    # Convert time correctly
    df_long["time"] = pd.to_datetime(df_long["time"], format="%H:%M")

    pivot = df_long.pivot_table(
        index=["Customer","date","time"],
        columns="Consumption Category",
        values="energy"
    ).reset_index()

    # Ensure correct ordering 
    pivot = pivot.sort_values(["Customer","date","time"])

    # Fill missing values with zeros (if any)
    pivot["GC"] = pivot["GC"].fillna(0)
    pivot["CL"] = pivot["CL"].fillna(0)
    pivot["GG"] = pivot["GG"].fillna(0)

    # Create load and pv columns
    pivot["load"] = pivot["GC"] + pivot["CL"]
    pivot["pv"] = pivot["GG"]

    return pivot


# ==========================================================
# DATA CLEANING (EXACT MATCH TO PAPER: CUSTOMER-LEVEL)
# ==========================================================

def clean_dataset(df):
    # Clean customer IDs from the paper (Table 4): 
    # These are the 54 customers with no anomalies over the full 3-year dataset
    clean_customer_ids = [
        2, 13, 14, 20, 33, 35, 38, 39, 56, 69, 73, 74, 75, 82, 87, 88,
        101, 104, 106, 109, 110, 119, 124, 130, 137, 141, 144, 152, 153,
        157, 161, 169, 176, 184, 188, 189, 193, 201, 202, 204, 206, 207,
        210, 211, 212, 214, 218, 244, 246, 253, 256, 273, 276, 297
    ]

    # Filter to only these clean customers
    df_clean = df[df["Customer"].isin(clean_customer_ids)]

    # Additional check for completeness (though paper's clean set should have full days)
    removed_days = 0
    cleaned_days = []

    # Group by customer and date, ensuring we have exactly 48 time intervals per day
    for (cust, date), day in df_clean.groupby(["Customer", "date"]):
        day = day.sort_values("time")

        if len(day) != 48:
            removed_days += 1
            continue

        cleaned_days.append(day)

    df_clean = pd.concat(cleaned_days)

    logger.info("Customers before cleaning: %s", df["Customer"].nunique())
    logger.info("Customers after cleaning: %s", df_clean["Customer"].nunique())
    logger.info("Days removed (incomplete): %s", removed_days)

    return df_clean


# ==========================================================
# TARIFF STRUCTURE
# ==========================================================

def build_tariff():

    # Tariff structure based on the paper's description (Figure 3):
    # - Off-peak (0.03 $/kWh): 00:00-07:00 and 22:00-24:00
    # - Shoulder (0.06 $/kWh): 07:00-14:00 and 20:00-22:00
    # - Peak (0.30 $/kWh): 14:00-20:00

    tariff = np.zeros(T)

    # Off-peak
    tariff[0:14] = 0.03
    tariff[44:48] = 0.03

    # Shoulder
    tariff[14:28] = 0.06
    tariff[40:44] = 0.06

    # Peak
    tariff[28:40] = 0.30

    return tariff


# ==========================================================
# FIGURE HELPERS
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6,4),
    "figure.dpi": 150
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=600)
    plt.close()


def figure3_tariff():
    tariff = build_tariff()
    t = np.arange(T) * DT

    plt.step(t, tariff, where="post")
    plt.xlabel("Time of day (hours)")
    plt.ylabel("Electricity price ($/kWh)")
    plt.title("Time-of-use tariff structure")
    plt.xlim(0, 24)

    save_fig("figure3_tariff")


def figure4_load_pv(df, customer=1):
    data = df[df["Customer"] == customer]
    day = data[data["date"] == data["date"].iloc[0]]

    load = day["load"].values
    pv = day["pv"].values
    t = np.arange(len(load)) * DT

    plt.plot(t, load, label="Load")
    plt.plot(t, pv, label="PV generation")
    plt.xlabel("Time of day (hours)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.title("Daily household load and PV generation")

    save_fig("figure4_load_pv")


def figure5_soc(schedule, soc_init=SOC_INIT):
    soc = soc_init + np.cumsum(schedule) * DT
    t = np.arange(len(soc)) * DT

    plt.plot(t, soc)
    plt.xlabel("Time of day (hours)")
    plt.ylabel("State of Charge (kWh)")
    plt.title("Battery state of charge trajectory")

    save_fig("figure5_soc")


def figure6_savings_histogram(savings):
    plt.hist(savings, bins=30)
    plt.xlabel("Annual savings ($)")
    plt.ylabel("Number of households")
    plt.title("Distribution of annual battery savings")

    save_fig("figure6_savings_histogram")


def figure8_capacity_sweep(capacities, savings):
    plt.plot(capacities, savings, marker='o')
    plt.xlabel("Battery capacity (kWh)")
    plt.ylabel("Annual savings ($)")
    plt.title("Battery savings vs capacity")

    save_fig("figure8_capacity_sweep")

# ==========================================================
# BILL CALCULATION
# ==========================================================

def bill_net(p, tariff):

    # Calculate imports and exports
    import_e = np.maximum(p,0)
    export_e = np.maximum(-p,0)

    # Net metering bill: cost of imports minus credit for exports
    return np.sum(import_e * tariff * DT) - np.sum(export_e * FEED_IN * DT)


# ==========================================================
# SOC MATRIX
# ==========================================================

def build_soc_matrix():

    # integration matrix to convert power schedule into SOC (kWh)
    A = np.tril(np.ones((T,T))) * DT
    return sp.csc_matrix(A)

A_soc = build_soc_matrix()


# ==========================================================
# BATTERY OPTIMISATION
# ==========================================================

def solve_battery(load, pv, H):

    # net household load before battery dispatch (kW)
    net = load - pv

    # quadratic cost terms for deviation from target profile
    P = 2 * H
    q = -2 * H.dot(net)

    # Constraints for charge/discharge limits and SOC limits
    P = sp.csc_matrix(P)

    # Build constraint matrices for charge/discharge limits and SOC limits
    A = sp.vstack([
        sp.eye(T),
        -sp.eye(T),
        A_soc,
        -A_soc
    ]).tocsc()

    # Lower bounds for charge/discharge and SOC
    l = np.hstack([
        -P_MAX*np.ones(T),
        -P_MAX*np.ones(T),
        -SOC_INIT*np.ones(T),
        -(E_MAX - SOC_INIT)*np.ones(T)
    ])

    # Upper bounds for charge/discharge and SOC
    u = np.hstack([
        P_MAX*np.ones(T),
        P_MAX*np.ones(T),
        (E_MAX - SOC_INIT)*np.ones(T),
        SOC_INIT*np.ones(T)
    ])

    # Set up and solve the quadratic program using OSQP
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False)

    res = prob.solve()

    return res.x


# ==========================================================
# H HEURISTIC
# ==========================================================

def optimise_H(load, pv, tariff):

    weights = tariff.copy()

    best = -1
    best_H = np.diag(weights)

    # Heuristic optimization of H by iteratively adjusting weights based on savings
    for _ in range(10):

        H = np.diag(weights)

        b = solve_battery(load, pv, H)

        p = load - pv - b

        s = bill_net(load-pv,tariff) - bill_net(p,tariff)

        if s > best:

            best = s
            best_H = H.copy()

            # Increase weights for time intervals with highest tariff to encourage battery usage during those times
            weights[tariff == np.max(tariff)] *= 2

        else:
            break

    return best_H


# ==========================================================
# SIMULATE ONE DAY
# ==========================================================

def simulate_day(load, pv, tariff):

    H = optimise_H(load, pv, tariff)

    b = solve_battery(load, pv, H)

    p = load - pv - b

    return bill_net(load-pv,tariff) - bill_net(p,tariff)


# ==========================================================
# RUN ALL CUSTOMERS
# ==========================================================

def _customer_annual_savings(args):
    customer, data, tariff = args
    total = 0

    for d in data["date"].unique():
        day = data[data["date"] == d]
        if len(day) != 48:
            continue

        load = day["load"].values
        pv = day["pv"].values
        total += simulate_day(load, pv, tariff)

    return customer, total


def run_all(df):
    # Build tariff structure
    tariff = build_tariff()

    df = df.sort_values(["Customer", "date"])
    groups = [(customer, group, tariff) for customer, group in df.groupby("Customer", sort=True)]

    if len(groups) == 0:
        return np.array([])

    processes = min(cpu_count(), len(groups)) or 1
    logger.info("Running customer simulations on %s CPU cores", processes)

    savings = []
    with Pool(processes=processes) as pool:
        for customer, total in pool.imap_unordered(_customer_annual_savings, groups, chunksize=1):
            logger.info("Completed customer %s: Annual savings $%.2f", customer, total)
            savings.append(total)

    return np.array(savings)


# ==========================================================
# MAIN
# ==========================================================

def capacity_sweep(load, pv, tariff):

    capacities = np.linspace(2,20,10)

    savings_vs_capacity = []

    global E_MAX

    original_capacity = E_MAX

    for cap in capacities:

        E_MAX = cap

        s = simulate_day(load,pv,tariff)

        savings_vs_capacity.append(s)

    E_MAX = original_capacity

    return capacities, np.array(savings_vs_capacity)


def generate_figures(df, savings, tariff):
    example_customer = df["Customer"].iloc[0]
    example_data = df[df["Customer"] == example_customer].sort_values(["date", "time"])
    example_day = example_data[example_data["date"] == example_data["date"].iloc[0]]
    load_ex = example_day["load"].values
    pv_ex = example_day["pv"].values

    figure3_tariff()
    figure4_load_pv(df, customer=example_customer)
    figure6_savings_histogram(savings)

    H_ex = optimise_H(load_ex, pv_ex, tariff)
    b_ex = solve_battery(load_ex, pv_ex, H_ex)
    figure5_soc(b_ex)

    capacities, savings_vs_capacity = capacity_sweep(load_ex, pv_ex, tariff)
    figure8_capacity_sweep(capacities, savings_vs_capacity)


def main():
    df = load_dataset("data.csv")
    df_clean = clean_dataset(df)
    df_clean.to_csv("cleaned_data.csv", index=False)

    logger.info("Cleaned dataset exported")

    if len(df_clean) == 0:
        logger.error("Dataset empty after cleaning.")
        exit()

    savings = run_all(df_clean)
    logger.info("Average annual savings: $%.2f", np.mean(savings))

    logger.info("Generating figures")
    generate_figures(df_clean, savings, build_tariff())


if __name__ == "__main__":
    main()