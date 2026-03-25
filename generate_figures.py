import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PUBLICATION PLOT STYLE
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

DT = 0.5


def save_fig(name):

    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/{name}.pdf")
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=600)

    plt.close()


# ==========================================================
# FIGURE 3 — TOU TARIFF
# ==========================================================

def figure3_tariff():

    tariff = np.zeros(48)

    tariff[0:14] = 0.03
    tariff[14:28] = 0.06
    tariff[28:40] = 0.30
    tariff[40:44] = 0.06
    tariff[44:48] = 0.03

    t = np.arange(48) * 0.5

    plt.step(t, tariff, where="post")

    plt.xlabel("Time of day (hours)")
    plt.ylabel("Electricity price ($/kWh)")
    plt.title("Time-of-use tariff structure")

    plt.xlim(0,24)

    save_fig("figure3_tariff")


# ==========================================================
# FIGURE 4 — LOAD VS PV EXAMPLE
# ==========================================================

def figure4_load_pv(df, customer=1):

    data = df[df["Customer"] == customer]

    day = data[data["date"] == data["date"].iloc[0]]

    load = day["load"].values
    pv = day["pv"].values

    t = np.arange(len(load))*0.5

    plt.plot(t, load, label="Load")
    plt.plot(t, pv, label="PV generation")

    plt.xlabel("Time of day (hours)")
    plt.ylabel("Power (kW)")

    plt.legend()

    plt.title("Daily household load and PV generation")

    save_fig("figure4_load_pv_example")


# ==========================================================
# FIGURE 5 — SOC TRAJECTORY
# ==========================================================

def figure5_soc(b):

    soc = np.cumsum(b) * DT

    t = np.arange(len(soc))*DT

    plt.plot(t, soc)

    plt.xlabel("Time of day (hours)")
    plt.ylabel("State of Charge (kWh)")

    plt.title("Battery state of charge trajectory")

    save_fig("figure5_soc")


# ==========================================================
# FIGURE 6/7 — SAVINGS HISTOGRAM
# ==========================================================

def figure6_savings_histogram(savings):

    plt.hist(savings, bins=30)

    plt.xlabel("Annual savings ($)")
    plt.ylabel("Number of households")

    plt.title("Distribution of annual battery savings")

    save_fig("figure6_savings_histogram")


# ==========================================================
# FIGURE 8 — BATTERY CAPACITY SWEEP
# ==========================================================

def figure8_capacity_sweep(capacities, savings):

    plt.plot(capacities, savings, marker='o')

    plt.xlabel("Battery capacity (kWh)")
    plt.ylabel("Annual savings ($)")

    plt.title("Battery savings vs capacity")

    save_fig("figure8_capacity_sweep")