"""
opendss_daily.py
================

Network-level validation of the Ratnam et al. (2015) QP battery schedules
using OpenDSS, driven entirely from Python via dss-python.

Builds a synthetic but realistic Australian LV distribution feeder from
scratch — no external .dss files required. Maps the 54 clean-dataset
customers from osqp_daily_v2 onto buses, injects their half-hourly grid
profiles as LoadShapes, runs 48-step daily power flow, and records:

    * Per-node voltage magnitudes (p.u.)
    * Substation (transformer secondary) active power flow (kW)
    * Total circuit losses (kWh)
    * Transformer loading (% of rated kVA)

Two scenarios are compared on each simulated day:

    Baseline  — no battery (p = l − g, the raw net load)
    QP        — battery dispatched (p = l − g − b, from osqp_daily_v2)

Usage:
    python opendss_daily.py                          # run representative days
    python opendss_daily.py --full                   # run every day in dataset

Prerequisites:
    pip install dss-python numpy pandas matplotlib

Australian LV feeder parameters:
    * 11 kV / 433 V, 200 kVA transformer (Dyn11)
    * 300 m underground backbone, 95 mm² Al XLPE
    * 10 backbone nodes at 30 m spacing, 3-phase 4-wire
    * 54 customers on single-phase service drops, ~18 per phase
    * AS 60038: 230 V nominal, +10 % / −6 % statutory limits
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONSTANTS
# ==========================================================

T = 48
DT = 0.5  # hours

# Voltage limits (AS 60038)
V_NOM = 230.0       # V phase-to-neutral
V_UPPER_PU = 1.10   # +10 %
V_LOWER_PU = 0.94   # −6 %

# Transformer
TX_KVA = 200.0
TX_PRIMARY_KV = 11.0
TX_SECONDARY_KV = 0.433  # line-to-line
TX_R_PCT = 1.5
TX_X_PCT = 4.0

# LV backbone cable: 95 mm² Al XLPE underground (typical Ausgrid)
CABLE_R_PER_KM = 0.32   # Ω/km
CABLE_X_PER_KM = 0.073  # Ω/km
CABLE_NORM_AMPS = 260.0  # A

# Backbone geometry
N_BACKBONE_NODES = 10
SEGMENT_LENGTH_M = 30.0  # metres between nodes
TOTAL_FEEDER_M = N_BACKBONE_NODES * SEGMENT_LENGTH_M  # 300 m

# Service drop (from backbone to customer meter)
SERVICE_LENGTH_M = 15.0
SERVICE_R_PER_KM = 0.64   # smaller cable, higher R
SERVICE_X_PER_KM = 0.08


# ==========================================================
# DSS ENGINE INITIALISATION
# ==========================================================

def get_dss():
    """Import and return the DSS engine singleton."""
    try:
        from dss import DSS
        return DSS
    except ImportError:
        logger.error(
            "dss-python not installed. Run: pip install dss-python")
        sys.exit(1)


# ==========================================================
# CUSTOMER-TO-BUS MAPPING
# ==========================================================

def assign_customers_to_buses(customer_ids, n_nodes=N_BACKBONE_NODES):
    """
    Distribute customers round-robin across 3 phases and evenly
    along the backbone. Returns:
        mapping: dict  {customer_id: (backbone_node, phase)}
    where backbone_node ∈ [1, n_nodes] and phase ∈ [1, 2, 3].
    """
    ids = sorted(customer_ids)
    mapping = {}
    for i, cid in enumerate(ids):
        phase = (i % 3) + 1
        node = (i // 3) % n_nodes + 1
        mapping[cid] = (node, phase)
    return mapping


# ==========================================================
# NETWORK BUILDER
# ==========================================================

def build_network(dss, customer_bus_map):
    """
    Define the full LV feeder circuit in OpenDSS.
    All elements are created via dss.Text.Command strings.
    """
    cmd = dss.Text

    # --- Clear and create circuit ---
    cmd.Command = "Clear"
    cmd.Command = (
        f"New Circuit.AusLV "
        f"basekv={TX_PRIMARY_KV} "
        f"pu=1.0 "
        f"phases=3 "
        f"bus1=sourcebus "
        f"Isc3=10000 Isc1=10500"
    )

    # --- Line code for backbone cable ---
    seg_km = SEGMENT_LENGTH_M / 1000.0
    cmd.Command = (
        f"New Linecode.backbone nphases=3 "
        f"r1={CABLE_R_PER_KM} x1={CABLE_X_PER_KM} "
        f"r0={CABLE_R_PER_KM * 3} x0={CABLE_X_PER_KM * 3} "
        f"units=km normamps={CABLE_NORM_AMPS}"
    )

    # --- Line code for service drops (single-phase) ---
    svc_km = SERVICE_LENGTH_M / 1000.0
    cmd.Command = (
        f"New Linecode.service nphases=1 "
        f"r1={SERVICE_R_PER_KM} x1={SERVICE_X_PER_KM} "
        f"r0={SERVICE_R_PER_KM * 3} x0={SERVICE_X_PER_KM * 3} "
        f"units=km normamps=80"
    )

    # --- MV/LV transformer ---
    cmd.Command = (
        f"New Transformer.MV_LV phases=3 windings=2 "
        f"buses=[sourcebus, lv_node_0] "
        f"conns=[delta, wye] "
        f"kvs=[{TX_PRIMARY_KV}, {TX_SECONDARY_KV}] "
        f"kvas=[{TX_KVA}, {TX_KVA}] "
        f"%Rs=[{TX_R_PCT / 2}, {TX_R_PCT / 2}] "
        f"xhl={TX_X_PCT}"
    )

    # --- Backbone line segments: lv_node_0 → lv_node_1 → ... → lv_node_N ---
    for i in range(N_BACKBONE_NODES):
        cmd.Command = (
            f"New Line.seg_{i} "
            f"bus1=lv_node_{i} bus2=lv_node_{i + 1} "
            f"linecode=backbone length={seg_km} units=km"
        )

    # --- Service drops and customer loads ---
    for cid, (node, phase) in customer_bus_map.items():
        bus_bb = f"lv_node_{node}.{phase}"
        bus_cust = f"cust_{cid}.{phase}"

        # Service line
        cmd.Command = (
            f"New Line.svc_{cid} "
            f"bus1={bus_bb} bus2={bus_cust} "
            f"linecode=service length={svc_km} units=km"
        )

        # Load element — placeholder kW=0; actual shape applied later
        kv_phase = TX_SECONDARY_KV / np.sqrt(3)  # ≈ 0.250 kV
        cmd.Command = (
            f"New Load.load_{cid} "
            f"bus1={bus_cust} phases=1 "
            f"kv={kv_phase:.4f} kw=1 pf=1 "
            f"model=1 status=variable "
            f"vminpu=0.85 vmaxpu=1.15"
        )

    # --- Voltage bases ---
    cmd.Command = (
        f"Set voltagebases=[{TX_PRIMARY_KV}, {TX_SECONDARY_KV}]"
    )
    cmd.Command = "Calcvoltagebases"

    # --- Monitors ---
    # Transformer secondary power
    cmd.Command = (
        "New Monitor.tx_power element=Transformer.MV_LV "
        "terminal=2 mode=1 ppolar=no"
    )
    # Customer voltages
    for cid in customer_bus_map:
        cmd.Command = (
            f"New Monitor.v_{cid} element=Load.load_{cid} "
            f"terminal=1 mode=0"
        )


def attach_loadshapes(dss, customer_bus_map, profiles, day_idx=0):
    """
    Create a LoadShape for each customer from their profile on a given
    day index, and assign it to the corresponding Load element.

    profiles: dict {customer_id: [day_profile_dict, ...]}
        where each day_profile_dict has 'grid' (length-48 numpy array).
    day_idx: which day to simulate (index into the customer's day list).

    Returns the date string of the simulated day (from the first
    customer that has data at that index), or None if no data.
    """
    cmd = dss.Text
    date_str = None

    for cid in customer_bus_map:
        days = profiles.get(cid, [])
        if day_idx >= len(days):
            # No data for this day — set flat zero
            mult_str = ",".join(["0"] * T)
            cmd.Command = (
                f"New Loadshape.shape_{cid} npts={T} "
                f"minterval=30 mult=({mult_str})"
            )
        else:
            day = days[day_idx]
            if date_str is None:
                date_str = str(day["date"])
            grid = day["grid"]
            mult_str = ",".join(f"{v:.6f}" for v in grid)
            cmd.Command = (
                f"New Loadshape.shape_{cid} npts={T} "
                f"minterval=30 mult=({mult_str})"
            )

        cmd.Command = (
            f"Load.load_{cid}.daily=shape_{cid}"
        )

    return date_str


def attach_baseline_shapes(dss, customer_bus_map, profiles, day_idx=0):
    """
    Same as attach_loadshapes but uses the baseline profile (load − pv,
    no battery) computed as grid_baseline = load − pv.
    """
    cmd = dss.Text
    date_str = None

    for cid in customer_bus_map:
        days = profiles.get(cid, [])
        if day_idx >= len(days):
            mult_str = ",".join(["0"] * T)
        else:
            day = days[day_idx]
            if date_str is None:
                date_str = str(day["date"])
            baseline = day["load"] - day["pv"]  # p when b = 0
            mult_str = ",".join(f"{v:.6f}" for v in baseline)

        cmd.Command = (
            f"New Loadshape.shape_{cid} npts={T} "
            f"minterval=30 mult=({mult_str})"
        )
        cmd.Command = f"Load.load_{cid}.daily=shape_{cid}"

    return date_str


# ==========================================================
# SIMULATION
# ==========================================================

def run_daily(dss):
    """Execute a 48-step daily power flow."""
    cmd = dss.Text
    cmd.Command = f"Set mode=daily stepsize=30m number={T}"
    cmd.Command = "Set controlmode=static"
    dss.ActiveCircuit.Solution.Solve()


def collect_voltages(dss, customer_bus_map):
    """
    Read the voltage monitor for each customer.
    Returns dict {cid: array of shape (T,) in per-unit}.
    """
    voltages = {}
    for cid in customer_bus_map:
        mon_name = f"v_{cid}"
        dss.ActiveCircuit.Monitors.Name = mon_name
        # Channel 1 is V1 magnitude for mode=0
        v_mag = np.array(dss.ActiveCircuit.Monitors.Channel(1))
        # Convert to per-unit (monitor records in actual volts for mode=0)
        v_pu = v_mag / V_NOM
        voltages[cid] = v_pu
    return voltages


def collect_tx_power(dss):
    """
    Read the transformer secondary power monitor.
    Returns arrays (p_kw, q_kvar) each of shape (T,).
    """
    dss.ActiveCircuit.Monitors.Name = "tx_power"
    # mode=1 ppolar=no: channels are P1, Q1, P2, Q2, P3, Q3
    p1 = np.array(dss.ActiveCircuit.Monitors.Channel(1))
    p2 = np.array(dss.ActiveCircuit.Monitors.Channel(3))
    p3 = np.array(dss.ActiveCircuit.Monitors.Channel(5))
    q1 = np.array(dss.ActiveCircuit.Monitors.Channel(2))
    q2 = np.array(dss.ActiveCircuit.Monitors.Channel(4))
    q3 = np.array(dss.ActiveCircuit.Monitors.Channel(6))
    p_total = p1 + p2 + p3
    q_total = q1 + q2 + q3
    return p_total, q_total


def collect_losses(dss):
    """Total circuit losses in kW and kvar (snapshot of cumulative)."""
    losses = dss.ActiveCircuit.Losses  # returns (W, var)
    return losses[0] / 1000.0, losses[1] / 1000.0


def simulate_scenario(dss, customer_bus_map, profiles, day_idx,
                      use_baseline=False):
    """
    Full pipeline: build network → attach shapes → solve → collect.
    Returns a results dict.
    """
    build_network(dss, customer_bus_map)

    if use_baseline:
        date_str = attach_baseline_shapes(
            dss, customer_bus_map, profiles, day_idx)
    else:
        date_str = attach_loadshapes(
            dss, customer_bus_map, profiles, day_idx)

    run_daily(dss)

    voltages = collect_voltages(dss, customer_bus_map)
    tx_p, tx_q = collect_tx_power(dss)
    loss_kw, loss_kvar = collect_losses(dss)

    # Voltage statistics
    all_v = np.array(list(voltages.values()))  # (n_cust, T)
    v_min = all_v.min()
    v_max = all_v.max()
    n_violations = int(np.sum(
        (all_v < V_LOWER_PU) | (all_v > V_UPPER_PU)
    ))
    total_points = all_v.size

    return {
        "date": date_str,
        "voltages": voltages,         # {cid: (T,) array in p.u.}
        "tx_p_kw": tx_p,             # (T,) array
        "tx_q_kvar": tx_q,           # (T,) array
        "loss_kw": loss_kw,
        "loss_kvar": loss_kvar,
        "v_min_pu": v_min,
        "v_max_pu": v_max,
        "n_violations": n_violations,
        "total_points": total_points,
    }


def simulate_day_comparison(dss, customer_bus_map, profiles, day_idx):
    """Run both baseline and QP scenarios for one day, return both."""
    logger.info("Simulating day index %d — baseline ...", day_idx)
    base = simulate_scenario(
        dss, customer_bus_map, profiles, day_idx, use_baseline=True)
    logger.info("Simulating day index %d — QP dispatched ...", day_idx)
    qp = simulate_scenario(
        dss, customer_bus_map, profiles, day_idx, use_baseline=False)
    return base, qp


# ==========================================================
# PLOTTING
# ==========================================================

def plot_voltage_envelope(base, qp, date_str=None):
    """
    Plot min/max voltage envelopes across all customers for both
    scenarios, with statutory limits.
    """
    hours = np.arange(T) * DT

    base_all = np.array(list(base["voltages"].values()))
    qp_all = np.array(list(qp["voltages"].values()))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hours, base_all.min(axis=0), base_all.max(axis=0),
                    alpha=0.3, color="salmon", label="Baseline envelope")
    ax.fill_between(hours, qp_all.min(axis=0), qp_all.max(axis=0),
                    alpha=0.3, color="steelblue", label="QP envelope")
    ax.plot(hours, base_all.min(axis=0), color="salmon", lw=0.8)
    ax.plot(hours, base_all.max(axis=0), color="salmon", lw=0.8)
    ax.plot(hours, qp_all.min(axis=0), color="steelblue", lw=0.8)
    ax.plot(hours, qp_all.max(axis=0), color="steelblue", lw=0.8)
    ax.axhline(V_UPPER_PU, color="red", ls="--", lw=1, label="Upper limit")
    ax.axhline(V_LOWER_PU, color="red", ls="--", lw=1, label="Lower limit")
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage (p.u.)")
    title = "Voltage envelope: Baseline vs QP-dispatched"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.90, 1.15)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_substation_power(base, qp, date_str=None):
    """Compare substation active power for both scenarios."""
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, base["tx_p_kw"], color="salmon", lw=1.5,
            label="Baseline")
    ax.plot(hours, qp["tx_p_kw"], color="steelblue", lw=1.5,
            label="QP dispatched")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Substation P (kW)")
    title = "Transformer secondary power"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_voltage_heatmap(result, date_str=None, title_prefix=""):
    """
    Heatmap: customer (y) × time (x), coloured by voltage p.u.
    Useful for spotting which customers and times are near limits.
    """
    cids = sorted(result["voltages"].keys())
    v_matrix = np.array([result["voltages"][c] for c in cids])
    hours = np.arange(T) * DT

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        v_matrix, aspect="auto", cmap="RdYlGn",
        vmin=V_LOWER_PU - 0.02, vmax=V_UPPER_PU + 0.02,
        extent=[0, 24, len(cids), 0],
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Customer index")
    title = f"{title_prefix}Voltage heatmap"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Voltage (p.u.)")
    plt.tight_layout()
    plt.show()


def plot_daily_summary_table(base, qp, date_str=None):
    """Print a comparison summary to stdout."""
    print("\n" + "=" * 60)
    if date_str:
        print(f"  Day: {date_str}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'QP':>12}")
    print("-" * 60)
    print(f"{'V min (p.u.)':<30} {base['v_min_pu']:>12.4f} "
          f"{qp['v_min_pu']:>12.4f}")
    print(f"{'V max (p.u.)':<30} {base['v_max_pu']:>12.4f} "
          f"{qp['v_max_pu']:>12.4f}")
    print(f"{'Voltage violations':<30} "
          f"{base['n_violations']:>12d} {qp['n_violations']:>12d}")
    print(f"{'Total (cust×interval) points':<30} "
          f"{base['total_points']:>12d} {qp['total_points']:>12d}")
    print(f"{'Peak TX power (kW)':<30} "
          f"{np.max(np.abs(base['tx_p_kw'])):>12.1f} "
          f"{np.max(np.abs(qp['tx_p_kw'])):>12.1f}")
    print(f"{'Total losses (kW)':<30} "
          f"{base['loss_kw']:>12.2f} {qp['loss_kw']:>12.2f}")
    print("=" * 60 + "\n")


# ==========================================================
# FULL-YEAR SWEEP
# ==========================================================

def run_full_sweep(dss, customer_bus_map, profiles, max_days=None):
    """
    Simulate every day (or up to max_days) and collect summary stats.
    Returns a DataFrame with one row per day.
    """
    # Determine how many days are available
    n_days = max(len(days) for days in profiles.values())
    if max_days:
        n_days = min(n_days, max_days)
    logger.info("Running full sweep for %d days", n_days)

    records = []
    for d in range(n_days):
        try:
            base, qp = simulate_day_comparison(
                dss, customer_bus_map, profiles, d)
        except Exception as e:
            logger.warning("Day %d failed: %s", d, e)
            continue

        records.append({
            "day_idx": d,
            "date": base["date"],
            "base_v_min": base["v_min_pu"],
            "base_v_max": base["v_max_pu"],
            "base_violations": base["n_violations"],
            "base_peak_tx_kw": np.max(np.abs(base["tx_p_kw"])),
            "base_loss_kw": base["loss_kw"],
            "qp_v_min": qp["v_min_pu"],
            "qp_v_max": qp["v_max_pu"],
            "qp_violations": qp["n_violations"],
            "qp_peak_tx_kw": np.max(np.abs(qp["tx_p_kw"])),
            "qp_loss_kw": qp["loss_kw"],
        })

        if d % 50 == 0:
            logger.info("Completed %d / %d days", d + 1, n_days)

    df = pd.DataFrame(records)
    return df


def plot_sweep_results(sweep_df):
    """Summary plots from the full-year sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Voltage extremes over time
    ax = axes[0, 0]
    ax.plot(sweep_df["base_v_max"], alpha=0.6, label="Baseline V_max")
    ax.plot(sweep_df["qp_v_max"], alpha=0.6, label="QP V_max")
    ax.axhline(V_UPPER_PU, color="red", ls="--", lw=0.8)
    ax.set_ylabel("V max (p.u.)")
    ax.set_title("Daily maximum voltage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sweep_df["base_v_min"], alpha=0.6, label="Baseline V_min")
    ax.plot(sweep_df["qp_v_min"], alpha=0.6, label="QP V_min")
    ax.axhline(V_LOWER_PU, color="red", ls="--", lw=0.8)
    ax.set_ylabel("V min (p.u.)")
    ax.set_title("Daily minimum voltage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Peak transformer loading
    ax = axes[1, 0]
    ax.plot(sweep_df["base_peak_tx_kw"], alpha=0.6, label="Baseline")
    ax.plot(sweep_df["qp_peak_tx_kw"], alpha=0.6, label="QP")
    ax.set_ylabel("Peak |P| (kW)")
    ax.set_xlabel("Day")
    ax.set_title("Daily peak transformer power")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Violations
    ax = axes[1, 1]
    ax.bar(sweep_df.index - 0.2, sweep_df["base_violations"],
           width=0.4, alpha=0.6, label="Baseline")
    ax.bar(sweep_df.index + 0.2, sweep_df["qp_violations"],
           width=0.4, alpha=0.6, label="QP")
    ax.set_ylabel("Voltage violations")
    ax.set_xlabel("Day")
    ax.set_title("Daily voltage violation count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Full sweep: Baseline vs QP battery dispatch", y=1.01)
    plt.tight_layout()
    plt.show()


# ==========================================================
# LOAD PROFILES FROM DISK
# ==========================================================

def load_profiles_from_csv(csv_path):
    """
    Read the long-format CSV produced by osqp_daily_v2.save_profiles()
    and reconstruct the profiles dict:
        {customer_id: [day_profile_dict, ...]}

    Each day_profile_dict has keys:
        date, load, pv, battery, grid, soc, savings
    """
    df = pd.read_csv(csv_path)
    profiles = defaultdict(list)

    for (cust, date), grp in df.groupby(
            ["customer", "date"], sort=True):
        grp = grp.sort_values("interval")
        profiles[int(cust)].append({
            "date": date,
            "load": grp["load_kw"].to_numpy(dtype=np.float64),
            "pv": grp["pv_kw"].to_numpy(dtype=np.float64),
            "battery": grp["battery_kw"].to_numpy(dtype=np.float64),
            "grid": grp["grid_kw"].to_numpy(dtype=np.float64),
            "soc": grp["soc_kwh"].to_numpy(dtype=np.float64),
            "savings": grp["daily_savings"].iloc[0],
        })

    return dict(profiles)


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="OpenDSS validation of QP battery schedules")
    parser.add_argument(
        "--profiles", default="profiles/fit_profiles.csv",
        help="Path to the long-format CSV from osqp_daily_v2")
    parser.add_argument(
        "--full", action="store_true",
        help="Run every day instead of representative days only")
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Cap the number of days in a full sweep")
    parser.add_argument(
        "--summer-day", type=int, default=None,
        help="Day index for summer example (auto-picked if not set)")
    parser.add_argument(
        "--winter-day", type=int, default=None,
        help="Day index for winter example (auto-picked if not set)")
    args = parser.parse_args()

    # --- Load profiles ---
    if not os.path.exists(args.profiles):
        logger.error(
            "Profile CSV not found at %s. "
            "Run osqp_daily_v2.py first to generate it.",
            args.profiles)
        sys.exit(1)

    logger.info("Loading profiles from %s", args.profiles)
    profiles = load_profiles_from_csv(args.profiles)
    customer_ids = sorted(profiles.keys())
    logger.info("Loaded %d customers", len(customer_ids))

    # --- Build customer → bus mapping ---
    customer_bus_map = assign_customers_to_buses(customer_ids)
    logger.info("Assigned %d customers to %d backbone nodes, 3 phases",
                len(customer_bus_map), N_BACKBONE_NODES)

    # --- Get DSS engine ---
    dss = get_dss()

    if args.full:
        # --- Full sweep ---
        sweep_df = run_full_sweep(
            dss, customer_bus_map, profiles, max_days=args.max_days)
        sweep_df.to_csv("opendss_sweep_results.csv", index=False)
        logger.info("Sweep results saved to opendss_sweep_results.csv")
        plot_sweep_results(sweep_df)
    else:
        # --- Representative days ---
        # Pick a summer and winter day by index. Defaults assume the
        # dataset starts 1 July 2010, so day 190 ≈ early Jan (summer),
        # day 0 ≈ 1 July (winter).
        n_days = max(len(d) for d in profiles.values())
        summer_idx = args.summer_day if args.summer_day is not None else min(190, n_days - 1)
        winter_idx = args.winter_day if args.winter_day is not None else 0

        for label, day_idx in [("Summer", summer_idx),
                               ("Winter", winter_idx)]:
            logger.info("=== %s day (index %d) ===", label, day_idx)
            base, qp = simulate_day_comparison(
                dss, customer_bus_map, profiles, day_idx)
            date_str = base["date"] or f"day_{day_idx}"

            plot_daily_summary_table(base, qp, date_str)
            plot_voltage_envelope(base, qp, date_str)
            plot_substation_power(base, qp, date_str)
            plot_voltage_heatmap(base, date_str, title_prefix="Baseline: ")
            plot_voltage_heatmap(qp, date_str, title_prefix="QP: ")


if __name__ == "__main__":
    main()