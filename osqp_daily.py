import pandas as pd
import numpy as np
import osqp
import scipy.sparse as sp
import matplotlib.pyplot as plt
from generate_figures import *

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
# LOAD DATASET
# ==========================================================

def load_dataset(path):

    df = pd.read_csv(path, skiprows=1)

    time_cols = df.columns[5:]

    df_long = df.melt(
        id_vars=["Customer","Generator Capacity","Postcode",
                 "Consumption Category","date"],
        value_vars=time_cols,
        var_name="time",
        value_name="energy"
    )

    # Convert time correctly (CRITICAL FIX)
    df_long["time"] = pd.to_datetime(df_long["time"], format="%H:%M")

    pivot = df_long.pivot_table(
        index=["Customer","date","time"],
        columns="Consumption Category",
        values="energy"
    ).reset_index()

    pivot = pivot.sort_values(["Customer","date","time"])

    pivot["GC"] = pivot["GC"].fillna(0)
    pivot["CL"] = pivot["CL"].fillna(0)
    pivot["GG"] = pivot["GG"].fillna(0)

    pivot["load"] = pivot["GC"] + pivot["CL"]
    pivot["pv"] = pivot["GG"]

    return pivot


# ==========================================================
# DATA CLEANING (EXACT MATCH TO PAPER: CUSTOMER-LEVEL)
# ==========================================================

def clean_dataset(df):
    # Clean customer IDs from the paper (Table 4): These are the 54 customers with no anomalies over the full 3-year dataset
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

    for (cust, date), day in df_clean.groupby(["Customer", "date"]):
        day = day.sort_values("time")

        if len(day) != 48:
            removed_days += 1
            continue

        cleaned_days.append(day)

    df_clean = pd.concat(cleaned_days)

    print("Customers before cleaning:", df["Customer"].nunique())
    print("Customers after cleaning:", df_clean["Customer"].nunique())
    print("Days removed (incomplete):", removed_days)

    return df_clean


# ==========================================================
# TARIFF STRUCTURE
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
# BILL CALCULATION
# ==========================================================

def bill_net(p, tariff):

    import_e = np.maximum(p,0)
    export_e = np.maximum(-p,0)

    return np.sum(import_e * tariff * DT) - np.sum(export_e * FEED_IN * DT)


# ==========================================================
# SOC MATRIX
# ==========================================================

def build_soc_matrix():

    A = np.tril(np.ones((T,T))) * DT
    return sp.csc_matrix(A)

A_soc = build_soc_matrix()


# ==========================================================
# BATTERY OPTIMISATION
# ==========================================================

def solve_battery(load, pv, H):

    net = load - pv

    P = 2 * H
    q = -2 * H.dot(net)

    P = sp.csc_matrix(P)

    A = sp.vstack([
        sp.eye(T),
        -sp.eye(T),
        A_soc,
        -A_soc
    ]).tocsc()

    l = np.hstack([
        -P_MAX*np.ones(T),
        -P_MAX*np.ones(T),
        -SOC_INIT*np.ones(T),
        -(E_MAX - SOC_INIT)*np.ones(T)
    ])

    u = np.hstack([
        P_MAX*np.ones(T),
        P_MAX*np.ones(T),
        (E_MAX - SOC_INIT)*np.ones(T),
        SOC_INIT*np.ones(T)
    ])

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

    for _ in range(10):

        H = np.diag(weights)

        b = solve_battery(load, pv, H)

        p = load - pv - b

        s = bill_net(load-pv,tariff) - bill_net(p,tariff)

        if s > best:

            best = s
            best_H = H.copy()

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

def run_all(df):

    tariff = build_tariff()

    customers = df["Customer"].unique()

    savings = []

    for c in customers:

        data = df[df["Customer"] == c]

        total = 0

        for d in data["date"].unique():

            day = data[data["date"] == d]

            if len(day) != 48:
                continue

            load = day["load"].values
            pv = day["pv"].values

            total += simulate_day(load,pv,tariff)

        savings.append(total)

        print("customer",c,"annual savings",total)

    return np.array(savings)


# ==========================================================
# MAIN
# ==========================================================

df = load_dataset("data.csv")

df_clean = clean_dataset(df)

df_clean.to_csv("cleaned_data.csv", index=False)

print("Cleaned dataset exported")

if len(df_clean) == 0:

    print("Dataset empty after cleaning.")
    exit()

savings = run_all(df_clean)

print("average annual savings", np.mean(savings))


# ==========================================================
# PLOTS
# ==========================================================

plt.hist(savings, bins=30)
plt.title("Annual Savings Distribution")
plt.xlabel("Savings ($)")
plt.ylabel("Customers")
plt.show()

# =========================================================
# FIGURES
# =========================================================

tariff = build_tariff()

# ==========================================================
# Example household/day for Figures 4,5,8
# ==========================================================

example_customer = df["Customer"].iloc[0]

example_data = df[df["Customer"] == example_customer].sort_values(["date","time"])

example_day = example_data[example_data["date"] == example_data["date"].iloc[0]]

load_ex = example_day["load"].values
pv_ex = example_day["pv"].values


# ==========================================================
# Example battery optimisation (for SOC figure)
# ==========================================================

H_ex = optimise_H(load_ex, pv_ex, tariff)

b_ex = solve_battery(load_ex, pv_ex, H_ex)


# ==========================================================
# Capacity sweep (Figure 8)
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


capacities, savings_vs_capacity = capacity_sweep(load_ex, pv_ex, tariff)


# ==========================================================
# Generate figures
# ==========================================================

figure3_tariff()

figure4_load_pv(df)

soc = SOC_INIT - np.cumsum(b_ex) * DT
figure5_soc(b_ex)

figure6_savings_histogram(savings)

figure8_capacity_sweep(capacities, savings_vs_capacity)