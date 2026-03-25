import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL PARAMETERS
# ==========================================================

DT = 0.5
T = 48

P_MAX = 5
E_MAX = 13.5
ETA = 0.95

FEED_IN = 0.08

# ==========================================================
# DATASET LOADER
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

    pivot = df_long.pivot_table(
        index=["Customer","date","time"],
        columns="Consumption Category",
        values="energy"
    ).reset_index()

    pivot["load"] = pivot["GC"].fillna(0) + pivot["CL"].fillna(0)
    pivot["pv"] = pivot["GG"].fillna(0)

    return pivot


# ==========================================================
# TARIFF MODELS
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
# BILLING MODELS
# ==========================================================

def bill_net_metering(p, tariff):

    import_e = np.maximum(p,0)
    export_e = np.maximum(-p,0)

    bill = np.sum(import_e * tariff * DT) \
         - np.sum(export_e * FEED_IN * DT)

    return bill


def bill_gross(load, pv, tariff):

    consumption_cost = np.sum(load * tariff * DT)
    generation_credit = np.sum(pv * FEED_IN * DT)

    return consumption_cost - generation_credit


# ==========================================================
# BATTERY OPTIMISATION MODEL
# ==========================================================

def solve_battery(load, pv, H):

    b_charge = cp.Variable(T)
    b_discharge = cp.Variable(T)

    p = load - pv + b_charge - b_discharge

    soc = cp.cumsum(
        ETA * b_charge * DT - (1/ETA) * b_discharge * DT
    )

    objective = cp.Minimize(cp.quad_form(p, H))

    constraints = [

        b_charge >= 0,
        b_discharge >= 0,

        b_charge <= P_MAX,
        b_discharge <= P_MAX,

        soc >= 0,
        soc <= E_MAX

    ]

    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    b = b_discharge.value - b_charge.value

    return b


# ==========================================================
# H HEURISTIC OPTIMISATION
# ==========================================================

def optimise_H(load, pv, tariff):

    weights = tariff.copy()

    best_savings = -1
    best_H = np.diag(weights)

    for _ in range(15):

        H = np.diag(weights)

        b = solve_battery(load, pv, H)

        p = load - pv - b

        bill0 = bill_net_metering(load-pv, tariff)
        bill1 = bill_net_metering(p, tariff)

        savings = bill0 - bill1

        if savings > best_savings:

            best_savings = savings
            best_H = H.copy()

            # increase weights during peak price
            weights[tariff == np.max(tariff)] *= 2

        else:
            break

    return best_H


# ==========================================================
# DAILY SIMULATION
# ==========================================================

def simulate_day(load, pv, tariff):

    H = optimise_H(load, pv, tariff)

    b = solve_battery(load, pv, H)

    p = load - pv - b

    soc = np.cumsum(b * DT)

    bill_no = bill_net_metering(load-pv, tariff)
    bill_yes = bill_net_metering(p, tariff)

    savings = bill_no - bill_yes

    return savings


# ==========================================================
# CUSTOMER ANNUAL SIMULATION
# ==========================================================

def simulate_customer(df, customer, tariff):

    data = df[df["Customer"] == customer]

    savings_total = 0

    for date in data["date"].unique():

        day = data[data["date"] == date]

        load = day["load"].values
        pv = day["pv"].values

        if len(load) != 48:
            continue

        savings = simulate_day(load, pv, tariff)

        savings_total += savings

    return savings_total


# ==========================================================
# ALL CUSTOMERS SIMULATION
# ==========================================================

def run_all_customers(df):

    tariff = build_tariff()

    customers = df["Customer"].unique()

    results = []

    for c in customers:

        print("Simulating customer", c)

        annual = simulate_customer(df, c, tariff)

        results.append(annual)

    return np.array(results)


# ==========================================================
# FIGURE GENERATION
# ==========================================================

def plot_histogram(savings):

    plt.figure()

    plt.hist(savings, bins=30)

    plt.title("Annual Savings Distribution")
    plt.xlabel("Savings ($)")
    plt.ylabel("Number of customers")

    plt.show()


# ==========================================================
# BATTERY CAPACITY SWEEP
# ==========================================================

def capacity_sweep(load, pv, tariff):

    capacities = [0,2,4,6,8,10,13,20]

    savings = []

    global E_MAX

    for cap in capacities:

        E_MAX = cap

        s = simulate_day(load, pv, tariff)

        savings.append(s)

    plt.figure()

    plt.plot(capacities, savings, marker="o")

    plt.title("Savings vs Battery Capacity")
    plt.xlabel("Battery capacity (kWh)")
    plt.ylabel("Savings ($)")

    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    df = load_dataset("data.csv")

    tariff = build_tariff()

    print("Running full simulation across customers...")

    savings = run_all_customers(df)

    print("Average annual savings:", np.mean(savings))

    plot_histogram(savings)

    # Example capacity sweep for first household
    sample = df[df["Customer"] == 1]
    day = sample[sample["date"] == sample["date"].iloc[0]]

    load = day["load"].values
    pv = day["pv"].values

    capacity_sweep(load, pv, tariff)


if __name__ == "__main__":
    main()