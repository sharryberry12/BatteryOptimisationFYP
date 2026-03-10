import pandas as pd
import numpy as np
import cvxpy as cp

# -------------------------
# 1 Load and clean dataset
# -------------------------

df = pd.read_csv("data.csv", skiprows=1)

# reshape 48 half-hour columns
time_cols = df.columns[5:]

df_long = df.melt(
    id_vars=["Customer","Generator Capacity","Postcode",
             "Consumption Category","date"],
    value_vars=time_cols,
    var_name="time",
    value_name="energy"
)

# pivot categories
pivot = df_long.pivot_table(
    index=["Customer","date","time"],
    columns="Consumption Category",
    values="energy"
).reset_index()

# compute load and PV
pivot["load"] = pivot["GC"].fillna(0) + pivot["CL"].fillna(0)
pivot["pv"] = pivot["GG"].fillna(0)

# pick one household
house = pivot[pivot["Customer"]==1]

# select one day
day = house[house["date"]=="1-Jul-10"]

l = day["load"].values
g = day["pv"].values

T = len(l)

# -------------------------
# 2 Electricity prices
# -------------------------

price_import = 0.30
price_export = 0.08

# price signal
s = np.where(l-g > 0, price_import, price_export)

# weighting matrix
H = np.diag(s)

# -------------------------
# 3 Battery parameters
# -------------------------

E_max = 13.5   # kWh
P_max = 5      # kW
eta = 0.95

# decision variable
b = cp.Variable(T)

# grid power
p = l - g - b

# objective
objective = cp.Minimize(cp.quad_form(p, H))

# SOC dynamics
soc = cp.cumsum(b)

constraints = [
    b <= P_max,
    b >= -P_max,
    soc <= E_max,
    soc >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve()

b_opt = b.value
p_opt = l - g - b_opt

# -------------------------
# 4 Cost comparison
# -------------------------

cost_no_battery = np.sum((l-g) * s)
cost_battery = np.sum(p_opt * s)

print("Cost without battery:", cost_no_battery)
print("Cost with battery:", cost_battery)
print("Savings:", cost_no_battery - cost_battery)