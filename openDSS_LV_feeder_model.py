"""
openDSS_LV_feeder_model.py
================

Network-level validation of the Ratnam et al. (2015) QP battery schedules
using OpenDSS, driven entirely from Python via dss-python.

Builds a synthetic but realistic Australian LV distribution feeder from
scratch — no external .dss files required. Maps the 55 clean-dataset
customers from osqp_daily onto buses, injects their half-hourly grid
profiles as LoadShapes, runs 48-step daily power flow, and records:

    * Per-node voltage magnitudes (p.u.)
    * Substation (transformer secondary) active power flow (kW)
    * Total circuit losses (kWh)
    * Transformer loading (% of rated kVA)

Two scenarios are compared on each simulated day:

    Baseline  — no battery (p = l − g, the raw net load)
    QP        — battery dispatched (p = l − g − b, from osqp_daily.py)

Usage:
    python openDSS_LV_feeder_model.py                          # representative days, show plots
    python openDSS_LV_feeder_model.py --full                   # every day, show plots
    python openDSS_LV_feeder_model.py --full --save            # every day, save plots to ./figures/
    python openDSS_LV_feeder_model.py --save --output-dir runs # save plots into ./runs/ instead

Prerequisites:
    pip install dss-python numpy pandas matplotlib

Australian LV feeder parameters:
    * 11 kV / 433 V, 200 kVA transformer (Dyn11)
    * 300 m underground backbone, 95 mm² Al XLPE
    * 10 backbone nodes at 30 m spacing, 3-phase 4-wire
    * N customers on single-phase service drops, spread round-robin across phases
    * AS 60038: 230 V nominal, +10 % / −6 % statutory limits
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# OpenDSS engine. The `dss` module is the dss-python wrapper that exposes
# the OpenDSS COM-style API as a Python singleton. If the import fails we
# raise immediately because nothing in this script can run without it.
try:
    from dss import DSS as dss
except ImportError:
    raise ImportError(
        "dss-python is not installed. Install it with: pip install dss-python"
    )

logging.basicConfig(
    level=logging.INFO,                                       # show INFO and above
    format="%(asctime)s %(levelname)s: %(message)s",          # timestamp + level + msg
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONSTANTS
# ==========================================================

T = 48           # number of half-hourly intervals in a day (24 h / 0.5 h)
DT = 0.5         # length of one interval in hours

# --- Voltage limits per AS 60038 (Australian standard) ---
V_NOM = 230.0       # nominal phase-to-neutral voltage in volts
V_UPPER_PU = 1.10   # statutory upper limit: +10 % of nominal
V_LOWER_PU = 0.94   # statutory lower limit: −6 % of nominal

# --- Distribution transformer (MV → LV) ---
TX_KVA = 200.0           # rated apparent power in kVA
TX_PRIMARY_KV = 11.0     # MV (primary) line-to-line voltage in kV
TX_SECONDARY_KV = 0.433  # LV (secondary) line-to-line voltage in kV
TX_R_PCT = 1.5           # winding resistance as percent of base impedance
TX_X_PCT = 4.0           # leakage reactance as percent of base impedance

# --- LV backbone cable: 95 mm² Al XLPE underground (typical Ausgrid) ---
CABLE_R_PER_KM = 0.32    # positive-sequence resistance per km in ohms
CABLE_X_PER_KM = 0.073   # positive-sequence reactance per km in ohms
CABLE_NORM_AMPS = 260.0  # continuous current rating in amps

# --- Backbone geometry: a straight LV feeder running down a street ---
N_BACKBONE_NODES = 10                                  # number of pole-top tap points
SEGMENT_LENGTH_M = 30.0                                # spacing between adjacent tap points
TOTAL_FEEDER_M = N_BACKBONE_NODES * SEGMENT_LENGTH_M   # total feeder length in metres

# --- Service drop: cable from backbone tap to individual customer meter ---
SERVICE_LENGTH_M = 15.0  # length of each service drop
SERVICE_R_PER_KM = 0.64  # higher resistance because conductor is smaller
SERVICE_X_PER_KM = 0.08


# ==========================================================
# CUSTOMER-TO-BUS MAPPING
# ==========================================================

def assign_customers_to_buses(customer_ids, n_nodes=N_BACKBONE_NODES):
    """
    Distribute customers round-robin across 3 phases and evenly along
    the backbone, so each phase carries roughly the same number of houses.

    Returns:
        mapping: dict {customer_id: (backbone_node, phase)}
            backbone_node ∈ [1, n_nodes]
            phase ∈ {1, 2, 3}
    """
    ids = sorted(customer_ids)            # deterministic order across runs
    mapping = {}
    for i, cid in enumerate(ids):
        phase = (i % 3) + 1                # cycle 1, 2, 3, 1, 2, 3, ...
        node = (i // 3) % n_nodes + 1      # advance node every 3 customers
        mapping[cid] = (node, phase)
    return mapping


# ==========================================================
# NETWORK BUILDER
# ==========================================================

def build_network(customer_bus_map):
    """
    Define the full LV feeder circuit in OpenDSS. All elements are
    declared via dss.Text.Command strings using OpenDSS DSL syntax.
    """
    cmd = dss.Text  # short alias for the command pipe

    # --- Reset OpenDSS and start a new circuit ---
    cmd.Command = "Clear"                        # wipe any previous circuit
    cmd.Command = (
        f"New Circuit.AusLV "
        f"basekv={TX_PRIMARY_KV} "               # MV source voltage in kV
        f"pu=1.0 "                               # source set to 1.00 p.u.
        f"phases=3 "                             # three-phase source
        f"bus1=sourcebus "                       # name of the source bus
        f"Isc3=10000 Isc1=10500"                 # short-circuit currents in A
                                                 # (sets source impedance: high
                                                 #  Isc means a stiff MV grid)
    )

    # --- Line code template for the 3-phase backbone cable ---
    seg_km = SEGMENT_LENGTH_M / 1000.0
    cmd.Command = (
        f"New Linecode.backbone nphases=3 "
        f"r1={CABLE_R_PER_KM} x1={CABLE_X_PER_KM} "       # +ve sequence Ω/km
        f"r0={CABLE_R_PER_KM * 3} x0={CABLE_X_PER_KM * 3} "  # zero sequence ≈ 3× +ve
        f"units=km normamps={CABLE_NORM_AMPS}"             # rating in amps
    )

    # --- Line code template for single-phase customer service drops ---
    svc_km = SERVICE_LENGTH_M / 1000.0
    cmd.Command = (
        f"New Linecode.service nphases=1 "
        f"r1={SERVICE_R_PER_KM} x1={SERVICE_X_PER_KM} "
        f"r0={SERVICE_R_PER_KM * 3} x0={SERVICE_X_PER_KM * 3} "
        f"units=km normamps=80"                            # smaller cable, 80 A
    )

    # --- Distribution transformer (Dyn11: delta primary, wye secondary) ---
    cmd.Command = (
        f"New Transformer.MV_LV phases=3 windings=2 "      # 3-phase, 2 windings
        f"buses=[sourcebus, lv_node_0] "                   # primary, secondary buses
        f"conns=[delta, wye] "                             # winding connections
        f"kvs=[{TX_PRIMARY_KV}, {TX_SECONDARY_KV}] "       # rated kV each side
        f"kvas=[{TX_KVA}, {TX_KVA}] "                      # rated kVA each side
        f"%Rs=[{TX_R_PCT / 2}, {TX_R_PCT / 2}] "           # split R equally
        f"xhl={TX_X_PCT}"                                  # H-to-L leakage reactance
    )

    # --- Backbone line segments chaining lv_node_0 → lv_node_1 → ... → lv_node_N ---
    for i in range(N_BACKBONE_NODES):
        cmd.Command = (
            f"New Line.seg_{i} "
            f"bus1=lv_node_{i} bus2=lv_node_{i + 1} "      # connects adjacent nodes
            f"linecode=backbone length={seg_km} units=km"  # uses backbone template
        )

    # --- Service drops + customer load elements ---
    for cid, (node, phase) in customer_bus_map.items():
        bus_bb = f"lv_node_{node}.{phase}"   # backbone tap on a specific phase
        bus_cust = f"cust_{cid}.{phase}"     # customer meter bus on same phase

        # Service line: backbone tap to meter bus
        cmd.Command = (
            f"New Line.svc_{cid} "
            f"bus1={bus_bb} bus2={bus_cust} "
            f"linecode=service length={svc_km} units=km"
        )

        # Load element representing the house's net demand. kw=1 acts as a
        # base value that gets multiplied by the daily LoadShape (which
        # carries the actual signed kW). status=variable allows negative
        # multipliers (i.e. net export back to the grid).
        kv_phase = TX_SECONDARY_KV / np.sqrt(3)   # 433 V LL → ~250 V LN
        cmd.Command = (
            f"New Load.load_{cid} "
            f"bus1={bus_cust} phases=1 "          # single-phase load
            f"kv={kv_phase:.4f} kw=1 pf=1 "       # nominal kV, kW=1, pf=1
            f"model=1 status=variable "           # constant-PQ model, signed
            f"vminpu=0.85 vmaxpu=1.15"            # solver-tolerated voltage range
        )

    # --- Tell OpenDSS what voltage levels exist in the circuit ---
    cmd.Command = f"Set voltagebases=[{TX_PRIMARY_KV}, {TX_SECONDARY_KV}]"
    cmd.Command = "Calcvoltagebases"             # propagate base kV through buses

    # --- Monitors: data recorders attached to specific elements ---
    # Transformer secondary terminal: records active+reactive power per phase
    cmd.Command = (
        "New Monitor.tx_power element=Transformer.MV_LV "
        "terminal=2 mode=1 ppolar=no"            # terminal 2 = LV side; mode 1 = power
                                                 # ppolar=no → P, Q in rectangular form
    )
    # One voltage monitor per customer load element
    for cid in customer_bus_map:
        cmd.Command = (
            f"New Monitor.v_{cid} element=Load.load_{cid} "
            f"terminal=1 mode=0"                 # mode 0 = voltage magnitudes
        )


def attach_loadshapes(customer_bus_map, profiles, day_idx=0):
    """
    Create a LoadShape object for each customer based on the QP-dispatched
    grid profile p_k = l_k − g_k − b_k for the given day, then bind it
    to the customer's Load element via the `daily=` property.

    profiles: {customer_id: [day_profile_dict, ...]}
        each day_profile_dict has key 'grid' (length-T numpy array in kW).
    day_idx: index into the customer's day list.

    Returns the date string of the simulated day.
    """
    cmd = dss.Text
    date_str = None

    for cid in customer_bus_map:
        days = profiles.get(cid, [])
        if day_idx >= len(days):
            # No data for this customer on this day — pad with 48 zeros
            mult_str = ",".join(["0"] * T)
        else:
            day = days[day_idx]
            if date_str is None:
                date_str = str(day["date"])
            grid = day["grid"]                                # signed kW values
            mult_str = ",".join(f"{v:.6f}" for v in grid)     # 6 dp precision

        # Define the LoadShape (npts, minterval in minutes, multipliers)
        cmd.Command = (
            f"New Loadshape.shape_{cid} npts={T} "
            f"minterval=30 mult=({mult_str})"
        )
        # Bind shape to the load's daily simulation slot
        cmd.Command = f"Load.load_{cid}.daily=shape_{cid}"

    return date_str


def attach_baseline_shapes(customer_bus_map, profiles, day_idx=0):
    """
    Same as attach_loadshapes but uses the no-battery baseline:
        p_baseline_k = l_k − g_k    (i.e. b_k = 0 for all k)
    so we can compare network behaviour with vs without battery dispatch.
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
            baseline = day["load"] - day["pv"]                # net of PV only
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

def run_daily():
    """Execute one full day of 48 sequential 30-minute power flows."""
    cmd = dss.Text
    cmd.Command = f"Set mode=daily stepsize=30m number={T}"   # daily simulation mode
    cmd.Command = "Set controlmode=static"                    # no regulator action
    dss.ActiveCircuit.Solution.Solve()                        # solve all 48 steps


def collect_voltages(customer_bus_map):
    """
    Read the voltage monitor for each customer load.
    Returns dict {cid: numpy array of shape (T,) in per-unit}.
    """
    voltages = {}
    for cid in customer_bus_map:
        dss.ActiveCircuit.Monitors.Name = f"v_{cid}"          # activate this monitor
        v_mag = np.array(dss.ActiveCircuit.Monitors.Channel(1))  # ch 1 = |V| in volts
        voltages[cid] = v_mag / V_NOM                         # convert to p.u.
    return voltages


def collect_tx_power():
    """
    Read the transformer secondary power monitor.
    Returns (p_kw, q_kvar) — both numpy arrays of shape (T,) summed across phases.
    """
    dss.ActiveCircuit.Monitors.Name = "tx_power"
    # mode=1, ppolar=no channel layout: P1, Q1, P2, Q2, P3, Q3
    p1 = np.array(dss.ActiveCircuit.Monitors.Channel(1))
    p2 = np.array(dss.ActiveCircuit.Monitors.Channel(3))
    p3 = np.array(dss.ActiveCircuit.Monitors.Channel(5))
    q1 = np.array(dss.ActiveCircuit.Monitors.Channel(2))
    q2 = np.array(dss.ActiveCircuit.Monitors.Channel(4))
    q3 = np.array(dss.ActiveCircuit.Monitors.Channel(6))
    return p1 + p2 + p3, q1 + q2 + q3                         # totals across phases


def collect_losses():
    """Total cumulative circuit losses, returned as (kW, kvar)."""
    losses = dss.ActiveCircuit.Losses                         # OpenDSS returns (W, var)
    return losses[0] / 1000.0, losses[1] / 1000.0             # convert to kW / kvar


def simulate_scenario(customer_bus_map, profiles, day_idx, use_baseline=False):
    """
    End-to-end pipeline for one day under one scenario:
        build network → attach load shapes → solve → collect monitor data.
    Returns a dict of all results.
    """
    build_network(customer_bus_map)                           # rebuild from scratch

    if use_baseline:
        date_str = attach_baseline_shapes(customer_bus_map, profiles, day_idx)
    else:
        date_str = attach_loadshapes(customer_bus_map, profiles, day_idx)

    run_daily()                                               # solve 48 power flows

    voltages = collect_voltages(customer_bus_map)
    tx_p, tx_q = collect_tx_power()
    loss_kw, loss_kvar = collect_losses()

    # Aggregate voltage statistics across all (customer, interval) points
    all_v = np.array(list(voltages.values()))                 # shape (n_cust, T)
    n_violations = int(np.sum(
        (all_v < V_LOWER_PU) | (all_v > V_UPPER_PU)           # outside statutory band
    ))

    return {
        "date": date_str,
        "voltages": voltages,            # {cid: array (T,) in p.u.}
        "tx_p_kw": tx_p,                 # transformer P (T,) in kW
        "tx_q_kvar": tx_q,               # transformer Q (T,) in kvar
        "loss_kw": loss_kw,              # total circuit losses, kW
        "loss_kvar": loss_kvar,          # total circuit losses, kvar
        "v_min_pu": all_v.min(),         # worst undervoltage seen
        "v_max_pu": all_v.max(),         # worst overvoltage seen
        "n_violations": n_violations,    # count of out-of-spec points
        "total_points": all_v.size,      # total (customer × interval) samples
    }


def simulate_day_comparison(customer_bus_map, profiles, day_idx):
    """Run baseline and QP scenarios for one day and return both result dicts."""
    logger.info("Simulating day index %d — baseline ...", day_idx)
    base = simulate_scenario(customer_bus_map, profiles, day_idx, use_baseline=True)
    logger.info("Simulating day index %d — QP dispatched ...", day_idx)
    qp = simulate_scenario(customer_bus_map, profiles, day_idx, use_baseline=False)
    return base, qp


# ==========================================================
# PLOTTING
# ==========================================================

# Module-level output directory. When None, plots are shown interactively
# via plt.show(). When set to a directory path, plots are written there as
# PNG files and not shown — useful when running --full and you don't want
# to babysit dozens of figure windows.
OUTPUT_DIR = None


def _safe_filename(s):
    """Sanitise a string for use as part of a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _finalise_plot(name, date_str=None, subdir=None):
    """
    Either show the current matplotlib figure interactively or save it
    to OUTPUT_DIR depending on the global. Closes the figure either way
    so memory doesn't leak across many days.

    name     : base filename (no extension)
    date_str : optional date appended to the filename for traceability
    subdir   : optional sub-folder under OUTPUT_DIR (e.g. 'baseline', 'qp')
    """
    plt.tight_layout()
    if OUTPUT_DIR is None:
        plt.show()
    else:
        target_dir = OUTPUT_DIR
        if subdir:
            target_dir = os.path.join(target_dir, _safe_filename(subdir))
        os.makedirs(target_dir, exist_ok=True)
        parts = [name]
        if date_str:
            parts.append(_safe_filename(date_str))
        fname = "_".join(parts) + ".png"
        path = os.path.join(target_dir, fname)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
        plt.close()


def plot_voltage_envelope(base, qp, date_str=None):
    """
    Min/max voltage envelope across all customers vs time, both scenarios,
    overlaid with the AS 60038 statutory limits.
    """
    hours = np.arange(T) * DT                                 # x-axis: 0, 0.5, ..., 23.5

    base_all = np.array(list(base["voltages"].values()))      # (n_cust, T)
    qp_all = np.array(list(qp["voltages"].values()))

    fig, ax = plt.subplots(figsize=(10, 5))
    # Shaded bands span the (min, max) across customers at each interval
    ax.fill_between(hours, base_all.min(axis=0), base_all.max(axis=0),
                    alpha=0.3, color="salmon", label="Baseline envelope")
    ax.fill_between(hours, qp_all.min(axis=0), qp_all.max(axis=0),
                    alpha=0.3, color="steelblue", label="QP envelope")
    # Outline the band edges for readability
    ax.plot(hours, base_all.min(axis=0), color="salmon", lw=0.8)
    ax.plot(hours, base_all.max(axis=0), color="salmon", lw=0.8)
    ax.plot(hours, qp_all.min(axis=0), color="steelblue", lw=0.8)
    ax.plot(hours, qp_all.max(axis=0), color="steelblue", lw=0.8)
    # Statutory limits
    ax.axhline(V_UPPER_PU, color="red", ls="--", lw=1, label="Upper limit")
    ax.axhline(V_LOWER_PU, color="red", ls="--", lw=1, label="Lower limit")
    ax.axhline(1.0, color="black", lw=0.5)                    # 1.0 p.u. reference
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
    _finalise_plot("voltage_envelope", date_str)


def plot_substation_power(base, qp, date_str=None):
    """Compare aggregate transformer secondary active power for both scenarios."""
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, base["tx_p_kw"], color="salmon", lw=1.5, label="Baseline")
    ax.plot(hours, qp["tx_p_kw"], color="steelblue", lw=1.5, label="QP dispatched")
    ax.axhline(0, color="black", lw=0.5)                      # zero-flow reference
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Substation P (kW)")                        # +ve = MV→LV import
                                                              # −ve = LV→MV export
    title = "Transformer secondary power"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)
    _finalise_plot("substation_power", date_str)


def plot_voltage_heatmap(result, date_str=None, title_prefix="",
                         adaptive=True):
    """
    2D heatmap: customer index (y-axis) × time (x-axis), coloured by voltage.

    Colour scheme: diverging blue-white-red, centred at the nominal voltage
    of 1.0 p.u. Blue = undervoltage, red = overvoltage.

    adaptive : if True (default), the colour limits stretch to fit the
        data's actual range (saturating at 1.5× the worst deviation seen).
        This makes subtle differences visible when the whole feeder is
        operating in a narrow voltage band — typical of an LV feeder
        sitting consistently above or below nominal due to source/tap
        configuration.
        If False, uses the absolute statutory limits [V_LOWER_PU, V_UPPER_PU]
        regardless of the actual data range. Useful when you want to
        compare two heatmaps with the same colour scale.
    """
    cids = sorted(result["voltages"].keys())
    v_matrix = np.array([result["voltages"][c] for c in cids])  # (n_cust, T)

    # Choose colour limits. Adaptive mode amplifies whatever deviations
    # exist; statutory mode anchors to the AS 60038 limits regardless.
    if adaptive:
        deviation = max(abs(v_matrix.min() - 1.0), abs(v_matrix.max() - 1.0))
        deviation = max(deviation, 0.005)  # floor to avoid divide-by-zero
        vmin = 1.0 - deviation
        vmax = 1.0 + deviation
    else:
        vmin, vmax = V_LOWER_PU, V_UPPER_PU
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        v_matrix, aspect="auto", cmap="coolwarm",
        norm=norm,
        extent=[0, 24, len(cids), 0],
        interpolation="nearest",
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Customer index")
    title = f"{title_prefix}Voltage heatmap"
    if date_str:
        title += f" ({date_str})"
    if adaptive:
        title += f"  [scale: {vmin:.3f}–{vmax:.3f} p.u.]"
    ax.set_title(title)

    # Build colorbar ticks: nominal in the middle, plus the actual limits.
    # In adaptive mode, also annotate where the statutory limits sit
    # relative to the visible scale so the viewer can tell whether the
    # data is anywhere near violating them.
    ticks = sorted({vmin, 1.0, vmax})
    cbar = plt.colorbar(im, ax=ax, label="Voltage (p.u.)",
                        ticks=ticks, extend="both")
    labels = []
    for t in ticks:
        lbl = f"{t:.3f}"
        if abs(t - 1.0) < 1e-9:
            lbl += "\n(nominal)"
        elif abs(t - V_LOWER_PU) < 1e-9:
            lbl += "\n(stat. lower)"
        elif abs(t - V_UPPER_PU) < 1e-9:
            lbl += "\n(stat. upper)"
        labels.append(lbl)
    cbar.ax.set_yticklabels(labels)

    subdir = None
    if title_prefix:
        subdir = "heatmaps_" + _safe_filename(title_prefix.strip(": ").lower())
    _finalise_plot("voltage_heatmap", date_str, subdir=subdir)


def plot_voltage_delta_heatmap(base, qp, date_str=None):
    """
    Heatmap of (baseline − QP) voltage at every (customer, interval).

    Interpretation:
      * Positive (red)  = baseline was higher than QP
                          → QP brought voltage DOWN
                          → improvement during over-voltage periods
      * Negative (blue) = baseline was lower than QP
                          → QP brought voltage UP
                          → improvement during under-voltage periods
      * White (~0)      = QP made no difference at this point

    This is much more informative than two near-identical absolute
    heatmaps when the whole feeder is sitting in a narrow voltage band.
    """
    cids = sorted(base["voltages"].keys())
    base_mat = np.array([base["voltages"][c] for c in cids])
    qp_mat = np.array([qp["voltages"][c] for c in cids])
    delta = base_mat - qp_mat                          # (n_cust, T)

    # Symmetric scale around zero, sized to the actual largest |delta|
    abs_max = max(abs(delta.min()), abs(delta.max()), 0.001)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        delta, aspect="auto", cmap="coolwarm",         # blue−white−red
        norm=norm,
        extent=[0, 24, len(cids), 0],
        interpolation="nearest",
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Customer index")
    title = "Voltage delta: baseline − QP"
    if date_str:
        title += f" ({date_str})"
    title += f"  [max |Δ| = {abs_max*1000:.1f} mV/V]"
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, label="Δ voltage (p.u.)",
                        extend="both")
    # Annotate what each side of zero means
    cbar.ax.text(
        1.5, 1.02, "QP lower\n(over-V relief)",
        transform=cbar.ax.transAxes, fontsize=8, ha="left", va="bottom",
        color="darkred",
    )
    cbar.ax.text(
        1.5, -0.02, "QP higher\n(under-V relief)",
        transform=cbar.ax.transAxes, fontsize=8, ha="left", va="top",
        color="darkblue",
    )

    # Brief stats panel inside the plot
    pct_improved = 100.0 * np.mean(np.abs(delta) > 0.001)
    mean_improvement = np.mean(np.abs(delta)) * 1000
    stats = (f"|Δ| > 1 mV/V at {pct_improved:.0f}% of points\n"
             f"mean |Δ| = {mean_improvement:.1f} mV/V")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    _finalise_plot("voltage_delta", date_str, subdir="heatmaps_delta")


def plot_daily_summary_table(base, qp, date_str=None):
    """
    Print a side-by-side comparison table of key daily metrics to stdout.
    When OUTPUT_DIR is set, the same table is also appended to
    summaries.txt inside the output directory so it persists across runs.
    """
    lines = []
    lines.append("=" * 60)
    if date_str:
        lines.append(f"  Day: {date_str}")
    lines.append("=" * 60)
    lines.append(f"{'Metric':<30} {'Baseline':>12} {'QP':>12}")
    lines.append("-" * 60)
    lines.append(f"{'V min (p.u.)':<30} {base['v_min_pu']:>12.4f} {qp['v_min_pu']:>12.4f}")
    lines.append(f"{'V max (p.u.)':<30} {base['v_max_pu']:>12.4f} {qp['v_max_pu']:>12.4f}")
    lines.append(f"{'Voltage violations':<30} {base['n_violations']:>12d} {qp['n_violations']:>12d}")
    lines.append(f"{'Total (cust×interval) points':<30} {base['total_points']:>12d} {qp['total_points']:>12d}")
    lines.append(f"{'Peak TX power (kW)':<30} "
                 f"{np.max(np.abs(base['tx_p_kw'])):>12.1f} "
                 f"{np.max(np.abs(qp['tx_p_kw'])):>12.1f}")
    lines.append(f"{'Total losses (kW)':<30} {base['loss_kw']:>12.2f} {qp['loss_kw']:>12.2f}")
    lines.append("=" * 60)

    text = "\n" + "\n".join(lines) + "\n"
    print(text)

    if OUTPUT_DIR is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "summaries.txt"), "a") as fh:
            fh.write(text)


# ==========================================================
# FULL-YEAR SWEEP
# ==========================================================

def run_full_sweep(customer_bus_map, profiles, max_days=None,
                   per_day_plots=False):
    """
    Loop over every day in the dataset (optionally capped at max_days),
    run the baseline-vs-QP comparison, and accumulate one summary row
    per day. Returns a pandas DataFrame.

    per_day_plots : if True, also generate the four per-day plots
        (envelope, substation power, baseline heatmap, QP heatmap) for
        every simulated day. Only sensible in combination with --save,
        otherwise you'll be drowning in interactive figure windows.
    """
    n_days = max(len(days) for days in profiles.values())
    if max_days:
        n_days = min(n_days, max_days)
    logger.info("Running full sweep for %d days", n_days)

    records = []
    for d in range(n_days):
        try:
            base, qp = simulate_day_comparison(customer_bus_map, profiles, d)
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

        if per_day_plots:
            date_str = base["date"] or f"day_{d}"
            plot_daily_summary_table(base, qp, date_str)
            plot_voltage_envelope(base, qp, date_str)
            plot_substation_power(base, qp, date_str)
            plot_voltage_heatmap(base, date_str, title_prefix="Baseline: ")
            plot_voltage_heatmap(qp, date_str, title_prefix="QP: ")
            plot_voltage_delta_heatmap(base, qp, date_str)

        if d % 50 == 0:
            logger.info("Completed %d / %d days", d + 1, n_days)

    return pd.DataFrame(records)


def plot_sweep_results(sweep_df):
    """Four-panel summary chart for the full-year sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: daily V_max for both scenarios + upper limit
    ax = axes[0, 0]
    ax.plot(sweep_df["base_v_max"], alpha=0.6, label="Baseline V_max")
    ax.plot(sweep_df["qp_v_max"], alpha=0.6, label="QP V_max")
    ax.axhline(V_UPPER_PU, color="red", ls="--", lw=0.8)
    ax.set_ylabel("V max (p.u.)")
    ax.set_title("Daily maximum voltage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Top-right: daily V_min for both scenarios + lower limit
    ax = axes[0, 1]
    ax.plot(sweep_df["base_v_min"], alpha=0.6, label="Baseline V_min")
    ax.plot(sweep_df["qp_v_min"], alpha=0.6, label="QP V_min")
    ax.axhline(V_LOWER_PU, color="red", ls="--", lw=0.8)
    ax.set_ylabel("V min (p.u.)")
    ax.set_title("Daily minimum voltage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-left: peak transformer loading per day
    ax = axes[1, 0]
    ax.plot(sweep_df["base_peak_tx_kw"], alpha=0.6, label="Baseline")
    ax.plot(sweep_df["qp_peak_tx_kw"], alpha=0.6, label="QP")
    ax.set_ylabel("Peak |P| (kW)")
    ax.set_xlabel("Day")
    ax.set_title("Daily peak transformer power")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: violation counts per day
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
    _finalise_plot("sweep_summary")


# ==========================================================
# LOAD PROFILES FROM DISK
# ==========================================================

def load_profiles_from_csv(csv_path):
    """
    Read the long-format CSV produced by osqp_daily_v2.save_profiles()
    and reconstruct the nested dictionary:
        {customer_id: [day_profile_dict, ...]}

    Each day_profile_dict has keys:
        date, load, pv, battery, grid, soc, savings
    """
    df = pd.read_csv(csv_path)
    profiles = defaultdict(list)                              # auto-create empty lists

    # Group rows by (customer, date) and assemble one dict per group
    for (cust, date), grp in df.groupby(["customer", "date"], sort=True):
        grp = grp.sort_values("interval")                     # ensure 1..48 order
        profiles[int(cust)].append({
            "date": date,
            "load": grp["load_kw"].to_numpy(dtype=np.float64),
            "pv": grp["pv_kw"].to_numpy(dtype=np.float64),
            "battery": grp["battery_kw"].to_numpy(dtype=np.float64),
            "grid": grp["grid_kw"].to_numpy(dtype=np.float64),
            "soc": grp["soc_kwh"].to_numpy(dtype=np.float64),
            "savings": grp["daily_savings"].iloc[0],          # scalar per day
        })

    return dict(profiles)                                     # convert back to dict


# ==========================================================
# MAIN
# ==========================================================

def main():
    global OUTPUT_DIR

    # --- Command-line argument parsing ---
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
    parser.add_argument(
        "--save", action="store_true",
        help="Save figures to disk instead of opening interactive windows. "
             "Highly recommended with --full so you don't have to close "
             "hundreds of plot windows manually.")
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to save figures into when --save is set "
             "(default: ./figures). Created automatically if it doesn't exist.")
    parser.add_argument(
        "--per-day-plots", action="store_true",
        help="With --full, also generate the four per-day plots for every "
             "simulated day (envelope, substation power, two heatmaps). "
             "Implies --save unless you really want to babysit the windows.")
    args = parser.parse_args()

    # --- Configure plot output mode ---
    if args.save:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Plots will be saved to %s/", OUTPUT_DIR)
        # Use a non-interactive matplotlib backend so plt.savefig works
        # even on headless servers (no display available)
        import matplotlib
        matplotlib.use("Agg", force=True)

    # --- Verify the QP profile CSV exists before proceeding ---
    if not os.path.exists(args.profiles):
        logger.error(
            "Profile CSV not found at %s. "
            "Run osqp_daily_v2.py first to generate it.",
            args.profiles)
        sys.exit(1)

    # --- Load and inspect QP profiles ---
    logger.info("Loading profiles from %s", args.profiles)
    profiles = load_profiles_from_csv(args.profiles)
    customer_ids = sorted(profiles.keys())
    logger.info("Loaded %d customers", len(customer_ids))

    # --- Map customers onto network buses ---
    customer_bus_map = assign_customers_to_buses(customer_ids)
    logger.info("Assigned %d customers to %d backbone nodes, 3 phases",
                len(customer_bus_map), N_BACKBONE_NODES)

    if args.full:
        # --- Year-long sweep + summary plots ---
        sweep_df = run_full_sweep(customer_bus_map, profiles,
                                  max_days=args.max_days,
                                  per_day_plots=args.per_day_plots)
        # Save the per-day metrics CSV next to the figures when --save,
        # otherwise drop it in the working directory as before.
        csv_path = os.path.join(OUTPUT_DIR, "opendss_sweep_results.csv") \
            if args.save else "opendss_sweep_results.csv"
        sweep_df.to_csv(csv_path, index=False)
        logger.info("Sweep results saved to %s", csv_path)
        plot_sweep_results(sweep_df)
    else:
        # --- Representative days only ---
        # Defaults assume the dataset starts 1 July 2010, so:
        #   day 190 ≈ early January (peak summer)
        #   day 0   ≈ 1 July (peak winter)
        n_days = max(len(d) for d in profiles.values())
        summer_idx = args.summer_day if args.summer_day is not None else min(190, n_days - 1)
        winter_idx = args.winter_day if args.winter_day is not None else 0

        for label, day_idx in [("Summer", summer_idx), ("Winter", winter_idx)]:
            logger.info("=== %s day (index %d) ===", label, day_idx)
            base, qp = simulate_day_comparison(customer_bus_map, profiles, day_idx)
            date_str = base["date"] or f"day_{day_idx}"

            plot_daily_summary_table(base, qp, date_str)
            plot_voltage_envelope(base, qp, date_str)
            plot_substation_power(base, qp, date_str)
            plot_voltage_heatmap(base, date_str, title_prefix="Baseline: ")
            plot_voltage_heatmap(qp, date_str, title_prefix="QP: ")
            plot_voltage_delta_heatmap(base, qp, date_str)


if __name__ == "__main__":
    main()