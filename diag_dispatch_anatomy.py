"""Anatomy of one QP dispatch day: why the profiles look 'flat'."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import osqp_daily as base
from peak_duty_analysis import C_BLUE, C_ORANGE, C_AQUA, INK_2, MUTED

DT, T = base.DT, base.T
tariff = base.build_tariff()

df = pd.read_csv("profiles/fit_profiles.csv", nrows=48 * 400)
g = df[(df["customer"] == 1) & (df["date"] == "22-Nov-10")]
g = g.sort_values("interval")
load, pv = g["load_kw"].to_numpy(), g["pv_kw"].to_numpy()
b, grid = g["battery_kw"].to_numpy(), g["grid_kw"].to_numpy()
net = load - pv
hours = np.arange(T) * DT

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax in (ax1, ax2):
    ax.axvspan(14, 20, color="#f0efec", zorder=0)
    ax.axvspan(7, 14, color="#f9f9f7", zorder=0)
    ax.axvspan(20, 22, color="#f9f9f7", zorder=0)
    ax.grid(axis="x", visible=False)
ax1.annotate("peak tariff", xy=(17, 0.95), xycoords=("data", "axes fraction"),
             ha="center", fontsize=8, color=INK_2)

ax1.plot(hours, net, color=C_ORANGE, lw=2, label="net load (load − PV)")
ax1.plot(hours, grid, color=C_BLUE, lw=2, label="grid power after battery")
ax1.axhline(0, color=MUTED, lw=0.8)
ax1.set_ylabel("kW")
ax1.set_title("Customer 1, 2010-11-22 (FiT mode) — the QP flattens GRID "
              "power, stepwise by tariff tier", loc="left",
              fontweight="bold", fontsize=10)
ax1.legend(fontsize=9)

ax2.plot(hours, b, color=C_AQUA, lw=2, label="battery (+discharge)")
ax2.plot(hours, net, color=C_ORANGE, lw=1, alpha=0.5,
         label="net load (for comparison)")
ax2.axhline(0, color=MUTED, lw=0.8)
ax2.set_ylabel("kW")
ax2.set_xlabel("hour of day")
ax2.set_xlim(0, 24)
ax2.set_xticks(range(0, 25, 3))
ax2.set_title("… so the BATTERY mirrors every wiggle of net load, offset "
              "by the flat grid level", loc="left", fontweight="bold",
              fontsize=10)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig("figures/qp_dispatch_anatomy.png", dpi=150,
            bbox_inches="tight")
print("saved figures/qp_dispatch_anatomy.png")
