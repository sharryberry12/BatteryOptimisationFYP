"""
ieee_13_bus_openDSS.py
==========================

Network-level validation of the Ratnam et al. (2015) QP battery schedules
using OpenDSS, driven entirely from Python via dss-python.

Builds the IEEE 13 Node Test Feeder — a standard 4.16 kV radial
distribution network with unbalanced loads, voltage regulators, and
mixed overhead/underground lines. Adds residential LV distribution
transformers at key load buses, then maps the 55 clean-dataset customers
from osqp_daily_v2 onto LV service drops. Injects their half-hourly grid
profiles as LoadShapes, runs 48-step daily power flow, and records:

    * Per-node voltage magnitudes (p.u.)
    * Substation (transformer secondary) active power flow (kW)
    * Total circuit losses (kWh)
    * Transformer loading (% of rated kVA)

Two scenarios are compared on each simulated day:

    Baseline  — no battery (p = l − g, the raw net load)
    QP        — battery dispatched (p = l − g − b, from osqp_daily_openDSS.py)

IEEE 13 Bus Topology (from IEEE PES Test Feeder):

    SourceBus ─── [Sub Xfmr 115/4.16 kV] ─── 650
                                               │
                                          [Reg 1,2,3]
                                               │
                                              632 ───── 633 ─── [XFM-1] ─── 634
                                             / | \\
                                           645  670     671 ─── 692 ─── 675
                                            |         / | \\
                                           646     684  680  (switch)
                                                  / \\
                                                611  652

Customer integration:
    * 5 LV distribution transformers (4.16 kV / 0.4 kV, Dyn11) added at
      buses 632, 671, 675, 652, 634 to create LV laterals
    * 55 customers mapped round-robin across these LV zones and 3 phases
    * Each customer connected via a 15 m single-phase service drop

Usage:
    python ieee_13_bus_openDSS.py                          # representative days, show plots
    python ieee_13_bus_openDSS.py --full                   # every day, show plots
    python ieee_13_bus_openDSS.py --full --save            # every day, save plots to ./figures/
    python ieee_13_bus_openDSS.py --save --output-dir runs # save plots into ./runs/ instead

Prerequisites:
    pip install dss-python numpy pandas matplotlib

References:
    * IEEE 13 Node Test Feeder — IEEE PES Distribution System Analysis
      Subcommittee (https://cmte.ieee.org/pes-testfeeders/resources/)
    * OpenDSS reference implementation: IEEETestCases/13Bus/IEEE13Nodeckt.dss
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

# OpenDSS engine via dss-python
try:
    from dss import DSS as dss
except ImportError:
    raise ImportError(
        "dss-python is not installed. Install it with: pip install dss-python"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONSTANTS
# ==========================================================

T = 48           # number of half-hourly intervals in a day
DT = 0.5         # length of one interval in hours

# --- IEEE 13 bus voltage levels ---
MV_KV = 4.16         # medium-voltage line-to-line in kV
LV_KV = 0.4          # LV line-to-line in kV (Australian 400/230 V)
HV_KV = 115.0        # high-voltage (source) in kV
V_NOM = LV_KV / np.sqrt(3) * 1000.0   # nominal LN voltage in volts (~230 V)

# --- Voltage limits per AS 60038 (for LV customers) ---
V_UPPER_PU = 1.10    # statutory upper limit: +10 % of nominal
V_LOWER_PU = 0.94    # statutory lower limit: −6 % of nominal

# --- LV distribution transformer specs (added at load buses) ---
LV_TX_KVA = 200.0        # rated apparent power per LV transformer
LV_TX_R_PCT = 1.5        # winding resistance as % of base impedance
LV_TX_X_PCT = 4.0        # leakage reactance as % of base impedance

# --- LV service drop cable ---
SERVICE_LENGTH_M = 15.0   # length of each customer service drop
SERVICE_R_PER_KM = 0.64   # Ω/km (small conductor, ~16 mm² Cu)
SERVICE_X_PER_KM = 0.08   # Ω/km

# --- LV buses where distribution transformers are placed ---
# These are selected IEEE 13 bus load buses where residential
# customers will be connected via step-down transformers.
LV_ZONES = ["632", "671", "675", "652", "634"]


# ==========================================================
# CUSTOMER-TO-BUS MAPPING
# ==========================================================

def assign_customers_to_buses(customer_ids, lv_zones=LV_ZONES):
    """
    Distribute customers round-robin across LV zones and 3 phases.

    Returns:
        mapping: dict {customer_id: (zone_bus, phase)}
            zone_bus ∈ LV_ZONES (e.g. "632", "671", ...)
            phase ∈ {1, 2, 3}
    """
    ids = sorted(customer_ids)
    mapping = {}
    n_zones = len(lv_zones)
    for i, cid in enumerate(ids):
        phase = (i % 3) + 1                        # cycle 1, 2, 3
        zone_idx = (i // 3) % n_zones               # cycle through zones
        mapping[cid] = (lv_zones[zone_idx], phase)
    return mapping


# ==========================================================
# IEEE 13 BUS NETWORK BUILDER
# ==========================================================

def build_network(customer_bus_map):
    """
    Define the full IEEE 13 Node Test Feeder circuit in OpenDSS,
    augmented with LV distribution transformers and customer loads.

    The MV backbone follows the standard IEEE 13 bus specification:
    - 115 kV source → 5 MVA substation transformer → 4.16 kV
    - 3 single-phase voltage regulators at bus 650
    - 12 line segments using 7 line configurations (601–607)
    - Spot loads, distributed load, capacitor banks
    - Inline transformer XFM-1 (4.16/0.48 kV) at bus 633→634
    """
    cmd = dss.Text

    # ──────────────────────────────────────────────────────────
    # 1. CIRCUIT AND SOURCE
    # ──────────────────────────────────────────────────────────
    cmd.Command = "Clear"
    cmd.Command = (
        f"New Circuit.IEEE13Nodeckt "
        f"basekv={HV_KV} pu=1.0001 phases=3 "
        f"bus1=SourceBus "
        f"Angle=30 "                                  # advance 30° so results
                                                       # match published angles
        f"MVAsc3=20000 MVASC1=21000"                   # stiff infinite bus
    )

    # ──────────────────────────────────────────────────────────
    # 2. SUBSTATION TRANSFORMER (115 kV → 4.16 kV)
    # ──────────────────────────────────────────────────────────
    # The published test case starts at 1.0 p.u. at bus 650. To make
    # this happen, we reduce the sub transformer impedance to nearly
    # zero using in-line RPN math (divide by 1000).
    cmd.Command = (
        "New Transformer.Sub Phases=3 Windings=2 XHL=(8 1000 /)"
    )
    cmd.Command = (
        f"~ wdg=1 bus=SourceBus conn=delta kv={HV_KV} "
        f"kva=5000 %r=(.5 1000 /)"
    )
    cmd.Command = (
        f"~ wdg=2 bus=650 conn=wye kv={MV_KV} "
        f"kva=5000 %r=(.5 1000 /)"
    )

    # ──────────────────────────────────────────────────────────
    # 3. VOLTAGE REGULATORS (3 single-phase, at bus 650→632)
    # ──────────────────────────────────────────────────────────
    for i, phase in enumerate([1, 2, 3], start=1):
        reg_name = f"Reg{i}"
        cmd.Command = (
            f"New Transformer.{reg_name} phases=1 bank=reg1 "
            f"XHL=0.01 kvas=[1666 1666] kvs=[{MV_KV / np.sqrt(3):.4f} "
            f"{MV_KV / np.sqrt(3):.4f}] %loadloss=0.01"
        )
        cmd.Command = (
            f"~ wdg=1 bus=650.{phase} conn=wye kv={MV_KV / np.sqrt(3):.4f} "
            f"kva=1666"
        )
        cmd.Command = (
            f"~ wdg=2 bus=RG60.{phase} conn=wye kv={MV_KV / np.sqrt(3):.4f} "
            f"kva=1666"
        )
        cmd.Command = (
            f"New Regcontrol.{reg_name} transformer={reg_name} winding=2 "
            f"vreg=122 band=2 ptratio=20 ctprim=700 R=3 X=9"
        )

    # ──────────────────────────────────────────────────────────
    # 4. LINE CODES (impedance matrices in Ω/mile)
    # ──────────────────────────────────────────────────────────
    # Config 601: 3-phase, BACN, 556.5 26/7 ACSR, 4/0 neutral, spacing 500
    cmd.Command = (
        "New Linecode.mtx601 nphases=3 BaseFreq=60 "
        "rmatrix=(0.3465 | 0.1560 0.3375 | 0.1580 0.1535 0.3414) "
        "xmatrix=(1.0179 | 0.5017 1.0478 | 0.4236 0.3849 1.0348) "
        "units=mi"
    )
    # Config 602: 3-phase, CABN, 4/0 6/1 ACSR, spacing 500
    cmd.Command = (
        "New Linecode.mtx602 nphases=3 BaseFreq=60 "
        "rmatrix=(0.7526 | 0.1580 0.7475 | 0.1560 0.1535 0.7436) "
        "xmatrix=(1.1814 | 0.4236 1.1983 | 0.5017 0.3849 1.2112) "
        "units=mi"
    )
    # Config 603: 2-phase, CBN, 1/0 ACSR, spacing 505
    cmd.Command = (
        "New Linecode.mtx603 nphases=2 BaseFreq=60 "
        "rmatrix=(1.3238 | 0.2066 1.3294) "
        "xmatrix=(1.3569 | 0.4591 1.3471) "
        "units=mi"
    )
    # Config 604: 2-phase, ACN, 1/0 ACSR, spacing 505
    cmd.Command = (
        "New Linecode.mtx604 nphases=2 BaseFreq=60 "
        "rmatrix=(1.3238 | 0.2066 1.3294) "
        "xmatrix=(1.3569 | 0.4591 1.3471) "
        "units=mi"
    )
    # Config 605: 1-phase, CN, 1/0 ACSR, spacing 510
    cmd.Command = (
        "New Linecode.mtx605 nphases=1 BaseFreq=60 "
        "rmatrix=(1.3292) "
        "xmatrix=(1.3475) "
        "units=mi"
    )
    # Config 606: 3-phase underground, 250k AA CN, spacing 515
    cmd.Command = (
        "New Linecode.mtx606 nphases=3 BaseFreq=60 "
        "rmatrix=(0.7982 | 0.3192 0.7891 | 0.2849 0.3192 0.7982) "
        "xmatrix=(0.4463 | 0.0328 0.4041 | -0.0143 0.0328 0.4463) "
        "Cmatrix=(383.948 | 0 383.948 | 0 0 383.948) "
        "units=mi"
    )
    # Config 607: 1-phase underground, 1/0 AA TS, spacing 520
    cmd.Command = (
        "New Linecode.mtx607 nphases=1 BaseFreq=60 "
        "rmatrix=(1.3425) "
        "xmatrix=(0.5124) "
        "units=mi"
    )

    # ──────────────────────────────────────────────────────────
    # 5. LINE SEGMENTS
    # ──────────────────────────────────────────────────────────
    # Note: bus 650 connects to regulator output bus RG60.
    # All line lengths in feet (converted to miles for OpenDSS).

    line_data = [
        # (name, bus1, bus2, phases, length_ft, linecode, bus1_nodes, bus2_nodes)
        ("650632",  "RG60",  "632",   3, 2000, "mtx601", "1.2.3", "1.2.3"),
        ("632670",  "632",   "670",   3, 667,  "mtx601", "1.2.3", "1.2.3"),
        ("670671",  "670",   "671",   3, 1333, "mtx601", "1.2.3", "1.2.3"),
        ("671680",  "671",   "680",   3, 1000, "mtx601", "1.2.3", "1.2.3"),
        ("632633",  "632",   "633",   3, 500,  "mtx602", "1.2.3", "1.2.3"),
        ("632645",  "632",   "645",   2, 500,  "mtx603", "3.2",   "3.2"),
        ("645646",  "645",   "646",   2, 300,  "mtx603", "3.2",   "3.2"),
        ("671684",  "671",   "684",   2, 300,  "mtx604", "1.3",   "1.3"),
        ("684611",  "684",   "611",   1, 300,  "mtx605", "3",     "3"),
        ("684652",  "684",   "652",   1, 800,  "mtx607", "1",     "1"),
        ("692675",  "692",   "675",   3, 500,  "mtx606", "1.2.3", "1.2.3"),
    ]

    for name, b1, b2, nph, length_ft, lc, n1, n2 in line_data:
        length_mi = length_ft / 5280.0
        cmd.Command = (
            f"New Line.{name} "
            f"Bus1={b1}.{n1} Bus2={b2}.{n2} "
            f"phases={nph} linecode={lc} "
            f"Length={length_mi:.6f} units=mi"
        )

    # Switch between 671 and 692 (zero-impedance connector)
    cmd.Command = (
        "New Line.671692 Bus1=671.1.2.3 Bus2=692.1.2.3 "
        "Switch=y r1=1e-4 r0=1e-4 x1=0 x0=0 c1=0 c0=0 Length=0.001 units=mi"
    )

    # ──────────────────────────────────────────────────────────
    # 6. IN-LINE TRANSFORMER XFM-1 (4.16 kV → 0.48 kV at bus 634)
    # ──────────────────────────────────────────────────────────
    cmd.Command = (
        "New Transformer.XFM1 Phases=3 Windings=2 XHL=2"
    )
    cmd.Command = (
        f"~ wdg=1 bus=633 conn=Wye kv={MV_KV} kva=500 %r=0.55"
    )
    cmd.Command = (
        "~ wdg=2 bus=634 conn=Wye kv=0.48 kva=500 %r=0.55"
    )

    # ──────────────────────────────────────────────────────────
    # 7. EXISTING SPOT LOADS (IEEE 13 bus standard loads)
    # ──────────────────────────────────────────────────────────
    # These remain as background loads representing the original
    # feeder loading. Customer loads from the QP profiles are
    # added on top via LV distribution transformers.

    # Bus 634 (0.48 kV, Wye-PQ)
    cmd.Command = "New Load.634a Bus1=634.1 Phases=1 Conn=Wye Model=1 kV=0.277 kW=160 kvar=110"
    cmd.Command = "New Load.634b Bus1=634.2 Phases=1 Conn=Wye Model=1 kV=0.277 kW=120 kvar=90"
    cmd.Command = "New Load.634c Bus1=634.3 Phases=1 Conn=Wye Model=1 kV=0.277 kW=120 kvar=90"

    # Bus 645 (4.16 kV, Wye-PQ, phase B only)
    cmd.Command = f"New Load.645 Bus1=645.2 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=170 kvar=125"

    # Bus 646 (4.16 kV, Delta-Z, phase B-C)
    cmd.Command = f"New Load.646 Bus1=646.2.3 Phases=1 Conn=Delta Model=2 kV={MV_KV} kW=230 kvar=132"

    # Bus 652 (4.16 kV, Wye-Z, phase A only)
    cmd.Command = f"New Load.652 Bus1=652.1 Phases=1 Conn=Wye Model=2 kV={MV_KV / np.sqrt(3):.4f} kW=128 kvar=86"

    # Bus 671 (4.16 kV, Delta-PQ, 3-phase)
    cmd.Command = f"New Load.671 Bus1=671.1.2.3 Phases=3 Conn=Delta Model=1 kV={MV_KV} kW=1155 kvar=660"

    # Bus 675 (4.16 kV, Wye-PQ)
    cmd.Command = f"New Load.675a Bus1=675.1 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=485 kvar=190"
    cmd.Command = f"New Load.675b Bus1=675.2 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=68 kvar=60"
    cmd.Command = f"New Load.675c Bus1=675.3 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=290 kvar=212"

    # Bus 692 (4.16 kV, Delta-I, phase C only)
    cmd.Command = f"New Load.692 Bus1=692.3.1 Phases=1 Conn=Delta Model=5 kV={MV_KV} kW=170 kvar=151"

    # Bus 611 (4.16 kV, Wye-I, phase C only)
    cmd.Command = f"New Load.611 Bus1=611.3 Phases=1 Conn=Wye Model=5 kV={MV_KV / np.sqrt(3):.4f} kW=170 kvar=80"

    # Bus 670 (distributed load at 1/3 from 632 on line 632-671)
    cmd.Command = f"New Load.670a Bus1=670.1 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=17 kvar=10"
    cmd.Command = f"New Load.670b Bus1=670.2 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=66 kvar=38"
    cmd.Command = f"New Load.670c Bus1=670.3 Phases=1 Conn=Wye Model=1 kV={MV_KV / np.sqrt(3):.4f} kW=117 kvar=68"

    # ──────────────────────────────────────────────────────────
    # 8. SHUNT CAPACITORS
    # ──────────────────────────────────────────────────────────
    # Bus 675: 3-phase, 600 kvar total
    cmd.Command = f"New Capacitor.Cap1 Bus1=675 phases=3 kvar=600 kv={MV_KV}"
    # Bus 611: 1-phase on phase C, 100 kvar
    cmd.Command = f"New Capacitor.Cap2 Bus1=611.3 phases=1 kvar=100 kv={MV_KV / np.sqrt(3):.4f}"

    # ──────────────────────────────────────────────────────────
    # 9. LV DISTRIBUTION TRANSFORMERS + CUSTOMER LOADS
    # ──────────────────────────────────────────────────────────
    # At each LV zone bus, add a 3-phase step-down transformer
    # from MV to 0.4 kV, with a short LV busbar. Customers then
    # connect via single-phase service drops off this LV busbar.

    # Service drop line code (single-phase, small conductor)
    svc_km = SERVICE_LENGTH_M / 1000.0
    cmd.Command = (
        f"New Linecode.service nphases=1 "
        f"r1={SERVICE_R_PER_KM} x1={SERVICE_X_PER_KM} "
        f"r0={SERVICE_R_PER_KM * 3} x0={SERVICE_X_PER_KM * 3} "
        f"units=km normamps=80"
    )

    # Build LV transformers at each zone bus
    for zone_bus in LV_ZONES:
        lv_bus_name = f"lv_{zone_bus}"

        if zone_bus == "634":
            # Bus 634 is already at 0.48 kV via XFM-1. Add another
            # small transformer 0.48 kV → 0.4 kV for our LV customers,
            # or just connect at the existing 0.48 kV level.
            # For simplicity, use 0.48 kV here (close to 0.4 kV).
            cmd.Command = (
                f"New Transformer.LV_{zone_bus} Phases=3 Windings=2 "
                f"XHL={LV_TX_X_PCT} "
                f"buses=[634, {lv_bus_name}] "
                f"conns=[wye, wye] "
                f"kvs=[0.48, {LV_KV}] "
                f"kvas=[{LV_TX_KVA}, {LV_TX_KVA}] "
                f"%Rs=[{LV_TX_R_PCT / 2}, {LV_TX_R_PCT / 2}]"
            )
        else:
            # Standard MV → LV distribution transformer (Dyn11)
            cmd.Command = (
                f"New Transformer.LV_{zone_bus} Phases=3 Windings=2 "
                f"XHL={LV_TX_X_PCT} "
                f"buses=[{zone_bus}, {lv_bus_name}] "
                f"conns=[delta, wye] "
                f"kvs=[{MV_KV}, {LV_KV}] "
                f"kvas=[{LV_TX_KVA}, {LV_TX_KVA}] "
                f"%Rs=[{LV_TX_R_PCT / 2}, {LV_TX_R_PCT / 2}]"
            )

    # Create customer service drops and load elements
    kv_phase = LV_KV / np.sqrt(3)   # 400 V LL → ~231 V LN
    for cid, (zone_bus, phase) in customer_bus_map.items():
        lv_bus_name = f"lv_{zone_bus}"
        bus_lv = f"{lv_bus_name}.{phase}"        # LV busbar tap on a phase
        bus_cust = f"cust_{cid}.{phase}"         # customer meter bus

        # Service line: LV busbar → customer meter
        cmd.Command = (
            f"New Line.svc_{cid} "
            f"bus1={bus_lv} bus2={bus_cust} "
            f"linecode=service length={svc_km} units=km"
        )

        # Load element: kW=1 base, multiplied by LoadShape.
        # status=variable allows negative multipliers (net export).
        cmd.Command = (
            f"New Load.load_{cid} "
            f"bus1={bus_cust} phases=1 "
            f"kv={kv_phase:.4f} kw=1 pf=1 "
            f"model=1 status=variable "
            f"vminpu=0.85 vmaxpu=1.15"
        )

    # ──────────────────────────────────────────────────────────
    # 10. VOLTAGE BASES AND SOLUTION SETUP
    # ──────────────────────────────────────────────────────────
    cmd.Command = f"Set voltagebases=[{HV_KV}, {MV_KV}, 0.48, {LV_KV}]"
    cmd.Command = "Calcvoltagebases"

    # ──────────────────────────────────────────────────────────
    # 11. MONITORS
    # ──────────────────────────────────────────────────────────
    # Substation transformer secondary power monitor
    cmd.Command = (
        "New Monitor.tx_power element=Transformer.Sub "
        "terminal=2 mode=1 ppolar=no"
    )
    # One voltage monitor per customer load element
    for cid in customer_bus_map:
        cmd.Command = (
            f"New Monitor.v_{cid} element=Load.load_{cid} "
            f"terminal=1 mode=0"
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
            mult_str = ",".join(["0"] * T)
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
        cmd.Command = f"Load.load_{cid}.daily=shape_{cid}"

    return date_str


def attach_baseline_shapes(customer_bus_map, profiles, day_idx=0):
    """
    Same as attach_loadshapes but uses the no-battery baseline:
        p_baseline_k = l_k − g_k    (i.e. b_k = 0 for all k)
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
            baseline = day["load"] - day["pv"]
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
    cmd.Command = f"Set mode=daily stepsize=30m number={T}"
    cmd.Command = "Set controlmode=static"
    dss.ActiveCircuit.Solution.Solve()


def collect_voltages(customer_bus_map):
    """
    Read the voltage monitor for each customer load.
    Returns dict {cid: numpy array of shape (T,) in per-unit}.
    """
    voltages = {}
    for cid in customer_bus_map:
        dss.ActiveCircuit.Monitors.Name = f"v_{cid}"
        v_mag = np.array(dss.ActiveCircuit.Monitors.Channel(1))
        voltages[cid] = v_mag / V_NOM
    return voltages


def collect_tx_power():
    """
    Read the substation transformer secondary power monitor.
    Returns (p_kw, q_kvar) — both numpy arrays of shape (T,) summed across phases.
    """
    dss.ActiveCircuit.Monitors.Name = "tx_power"
    p1 = np.array(dss.ActiveCircuit.Monitors.Channel(1))
    p2 = np.array(dss.ActiveCircuit.Monitors.Channel(3))
    p3 = np.array(dss.ActiveCircuit.Monitors.Channel(5))
    q1 = np.array(dss.ActiveCircuit.Monitors.Channel(2))
    q2 = np.array(dss.ActiveCircuit.Monitors.Channel(4))
    q3 = np.array(dss.ActiveCircuit.Monitors.Channel(6))
    return p1 + p2 + p3, q1 + q2 + q3


def collect_losses():
    """Total cumulative circuit losses, returned as (kW, kvar)."""
    losses = dss.ActiveCircuit.Losses
    return losses[0] / 1000.0, losses[1] / 1000.0


def simulate_scenario(customer_bus_map, profiles, day_idx, use_baseline=False):
    """
    End-to-end pipeline for one day under one scenario:
        build network → attach load shapes → solve → collect monitor data.
    Returns a dict of all results.
    """
    build_network(customer_bus_map)

    if use_baseline:
        date_str = attach_baseline_shapes(customer_bus_map, profiles, day_idx)
    else:
        date_str = attach_loadshapes(customer_bus_map, profiles, day_idx)

    run_daily()

    voltages = collect_voltages(customer_bus_map)
    tx_p, tx_q = collect_tx_power()
    loss_kw, loss_kvar = collect_losses()

    all_v = np.array(list(voltages.values()))
    n_violations = int(np.sum(
        (all_v < V_LOWER_PU) | (all_v > V_UPPER_PU)
    ))

    return {
        "date": date_str,
        "voltages": voltages,
        "tx_p_kw": tx_p,
        "tx_q_kvar": tx_q,
        "loss_kw": loss_kw,
        "loss_kvar": loss_kvar,
        "v_min_pu": all_v.min(),
        "v_max_pu": all_v.max(),
        "n_violations": n_violations,
        "total_points": all_v.size,
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

OUTPUT_DIR = None


def _safe_filename(s):
    """Sanitise a string for use as part of a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _finalise_plot(name, date_str=None, subdir=None):
    """
    Either show the current matplotlib figure interactively or save it
    to OUTPUT_DIR depending on the global.
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
    title = "IEEE 13 Bus — Voltage envelope: Baseline vs QP-dispatched"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.90, 1.15)
    ax.grid(alpha=0.3)
    _finalise_plot("voltage_envelope", date_str)


def plot_substation_power(base, qp, date_str=None):
    """Compare aggregate substation transformer secondary active power."""
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, base["tx_p_kw"], color="salmon", lw=1.5, label="Baseline")
    ax.plot(hours, qp["tx_p_kw"], color="steelblue", lw=1.5, label="QP dispatched")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Substation P (kW)")
    title = "IEEE 13 Bus — Substation transformer power"
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
    Diverging blue-white-red, centred at 1.0 p.u.
    """
    cids = sorted(result["voltages"].keys())
    v_matrix = np.array([result["voltages"][c] for c in cids])

    if adaptive:
        deviation = max(abs(v_matrix.min() - 1.0), abs(v_matrix.max() - 1.0))
        deviation = max(deviation, 0.005)
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
    title = f"{title_prefix}IEEE 13 Bus — Voltage heatmap"
    if date_str:
        title += f" ({date_str})"
    if adaptive:
        title += f"  [scale: {vmin:.3f}–{vmax:.3f} p.u.]"
    ax.set_title(title)

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

    Positive (red)  = baseline higher → QP brought voltage DOWN (over-V relief)
    Negative (blue) = baseline lower  → QP brought voltage UP   (under-V relief)
    """
    cids = sorted(base["voltages"].keys())
    base_mat = np.array([base["voltages"][c] for c in cids])
    qp_mat = np.array([qp["voltages"][c] for c in cids])
    delta = base_mat - qp_mat

    abs_max = max(abs(delta.min()), abs(delta.max()), 0.001)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        delta, aspect="auto", cmap="coolwarm",
        norm=norm,
        extent=[0, 24, len(cids), 0],
        interpolation="nearest",
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Customer index")
    title = "IEEE 13 Bus — Voltage delta: baseline − QP"
    if date_str:
        title += f" ({date_str})"
    title += f"  [max |Δ| = {abs_max*1000:.1f} mV/V]"
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, label="Δ voltage (p.u.)", extend="both")
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

    pct_improved = 100.0 * np.mean(np.abs(delta) > 0.001)
    mean_improvement = np.mean(np.abs(delta)) * 1000
    stats = (f"|Δ| > 1 mV/V at {pct_improved:.0f}% of points\n"
             f"mean |Δ| = {mean_improvement:.1f} mV/V")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    _finalise_plot("voltage_delta", date_str, subdir="heatmaps_delta")


def plot_daily_summary_table(base, qp, date_str=None):
    """Print a side-by-side comparison table of key daily metrics."""
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
    Loop over every day in the dataset, run baseline-vs-QP comparison,
    and accumulate one summary row per day.
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

    ax = axes[1, 0]
    ax.plot(sweep_df["base_peak_tx_kw"], alpha=0.6, label="Baseline")
    ax.plot(sweep_df["qp_peak_tx_kw"], alpha=0.6, label="QP")
    ax.set_ylabel("Peak |P| (kW)")
    ax.set_xlabel("Day")
    ax.set_title("Daily peak substation power")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

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

    plt.suptitle("IEEE 13 Bus — Full sweep: Baseline vs QP battery dispatch", y=1.01)
    _finalise_plot("sweep_summary")


# ==========================================================
# LOAD PROFILES FROM DISK
# ==========================================================

def load_profiles_from_csv(csv_path):
    """
    Read the long-format CSV produced by osqp_daily_v2.save_profiles()
    and reconstruct the nested dictionary:
        {customer_id: [day_profile_dict, ...]}
    """
    df = pd.read_csv(csv_path)
    profiles = defaultdict(list)

    for (cust, date), grp in df.groupby(["customer", "date"], sort=True):
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
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="IEEE 13 Bus OpenDSS validation of QP battery schedules")
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
        help="Save figures to disk instead of interactive windows")
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to save figures into when --save is set")
    parser.add_argument(
        "--per-day-plots", action="store_true",
        help="With --full, also generate per-day plots for every day")
    args = parser.parse_args()

    if args.save:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Plots will be saved to %s/", OUTPUT_DIR)
        import matplotlib
        matplotlib.use("Agg", force=True)

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

    customer_bus_map = assign_customers_to_buses(customer_ids)
    logger.info("Assigned %d customers to %d LV zones on IEEE 13 bus feeder",
                len(customer_bus_map), len(LV_ZONES))

    # Log the customer distribution across zones
    zone_counts = defaultdict(int)
    for _, (zone, _) in customer_bus_map.items():
        zone_counts[zone] += 1
    for zone in LV_ZONES:
        logger.info("  Zone bus %s: %d customers", zone, zone_counts[zone])

    if args.full:
        sweep_df = run_full_sweep(customer_bus_map, profiles,
                                  max_days=args.max_days,
                                  per_day_plots=args.per_day_plots)
        csv_path = os.path.join(OUTPUT_DIR, "ieee13_sweep_results.csv") \
            if args.save else "ieee13_sweep_results.csv"
        sweep_df.to_csv(csv_path, index=False)
        logger.info("Sweep results saved to %s", csv_path)
        plot_sweep_results(sweep_df)
    else:
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