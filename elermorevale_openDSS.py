"""
elermorevale_openDSS.py
===================

Port of the Elermore Vale (Wallsend, NSW) GridLAB-D network model to
OpenDSS, built entirely in Python via dss-python.

The original GridLAB-D model covers an Ausgrid 11 kV feeder in the
Newcastle/Lake Macquarie area with:
  - 132/11 kV zone substation (50 MVA, Dyn, with OLTC regulator)
  - 52 sections of 11 kV overhead line (z-matrix impedances)
  - 23 distribution transformers (11 kV / 433 V, 200-1000 kVA)
  - ~2,100 LV line segments (overhead + underground)
  - 1,785 residential loads (single-phase, constant-P)
  - 155 rooftop PV systems
  - 40 Redflow battery storage units (10 kWh, 5 kW)

Line impedances are extracted from common/Line Configs.glm:
  - 11 kV configs use z-matrix format (z11..z33 in Ohm/mile)
  - LV configs reference named conductors with per-mile resistance

This version adds profile-driven daily simulation and plotting,
matching the interface of openDSS_LV_feeder_model.py. It reads the
half-hourly CSV produced by osqp_daily_v2.save_profiles(), maps the
55 QP-scheduled customers onto a subset of network loads, and compares
baseline (no battery) vs QP-dispatched scenarios with:

    * Per-node voltage envelopes and heatmaps
    * Substation transformer active power flow
    * Total circuit losses
    * Voltage violation counts per AS 60038

Usage:
    python elermorevale_openDSS.py                                        # build + snapshot (original)
    python elermorevale_openDSS.py --profiles profiles/fit_profiles.csv   # daily sim
    python elermorevale_openDSS.py --profiles ... --full --save           # full sweep

Prerequisites:
    pip install dss-python numpy pandas matplotlib
"""

import argparse
import glob
import logging
import os
import re
import sys
from collections import defaultdict

import numpy as np

# --- OpenDSS engine (dss-python) ---
try:
    from dss import DSS as dss
except ImportError:
    raise ImportError(
        "dss-python is not installed. Install with: pip install dss-python"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==================================================================
# GLM PARSER
# ==================================================================

def parse_glm(filepath):
    """
    Parse a GridLAB-D .glm file into a list of (object_type, {props}).
    Strips // comments. Handles the flat object style used in the
    Ausgrid models (no deeply nested braces).
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    objs = []
    for m in re.finditer(r"object\s+(\w+)\s*\{([^}]*)\}", text, re.DOTALL):
        props = {}
        for pm in re.finditer(r"([\w.]+)\s+([^;]+);", m.group(2)):
            props[pm.group(1).strip()] = pm.group(2).strip()
        objs.append((m.group(1), props))
    return objs


def parse_all_glm(glm_dir):
    """
    Walk the Elermorevale directory tree and parse every .glm file.
    Returns a flat list of (source_filename, object_type, props_dict).
    """
    all_objs = []
    for fp in sorted(glob.glob(os.path.join(glm_dir, "**", "*.glm"),
                               recursive=True)):
        if "__MACOSX" in fp:
            continue
        for otype, props in parse_glm(fp):
            all_objs.append((os.path.basename(fp), otype, props))
    return all_objs


def parse_line_configs(common_dir):
    """
    Parse common/Line Configs.glm and return two dicts:
      conductors : {name: (resistance_ohm_per_mile, summer_rating_A)}
      configs    : {config_name: props_dict}
    """
    path = os.path.join(common_dir, "Line Configs.glm")
    objs = parse_glm(path)
    conductors = {}
    configs = {}
    for ot, p in objs:
        if ot in ("overhead_line_conductor", "underground_line_conductor"):
            name = p.get("name", "")
            r = gfloat(p.get("conductor_resistance",
                             p.get("resistance", "0")))
            rating = gfloat(p.get("rating.summer.continuous", "0"))
            conductors[name] = (r, rating)
        elif ot == "line_configuration":
            configs[p.get("name", "")] = p
    return conductors, configs


# ==================================================================
# IMPEDANCE EXTRACTION
# ==================================================================

def extract_impedances(conductors, configs):
    """
    Build a unified linecode table:
        {config_name: (r1_ohm_per_km, x1_ohm_per_km, rating_A, nphases)}

    - Elermore_line_config_* : from z-matrix z11 field (Ohm/mile -> Ohm/km)
    - conf_OHLine_* / conf_UGLine_* : from conductor resistance + estimated x
    """
    MI_TO_KM = 1.60934
    result = {}

    for name, p in configs.items():
        # --- Z-matrix format (Elermore 11 kV configs) ---
        if "z11" in p:
            try:
                z11 = complex(p["z11"].replace(" ", ""))
            except (ValueError, TypeError):
                z11 = 0 + 0j
            r_km = z11.real / MI_TO_KM
            x_km = z11.imag / MI_TO_KM
            if r_km < 0.001:
                rating = 1000       # busbar / jumper
            elif r_km < 0.15:
                rating = 400
            elif r_km < 0.30:
                rating = 300
            else:
                rating = 250
            result[name] = (round(r_km, 6), round(x_km, 6), rating, 3)
            continue

        # --- Conductor-reference format (LV configs) ---
        cond_a = p.get("conductor_A", "")
        if not cond_a:
            continue
        r_mi, rating = conductors.get(cond_a, (0, 0))
        r_km = r_mi / MI_TO_KM
        x_km = 0.08 if "UG" in name else 0.25
        nph = 3 if "conductor_B" in p else 1
        result[name] = (round(r_km, 4), round(x_km, 4), round(rating, 1), nph)

    return result


# ==================================================================
# HELPERS
# ==================================================================

def glm_phases_to_dss(phases_str):
    """
    Convert GridLAB-D phase notation to OpenDSS bus suffix and count.
        'AN'   -> ('.1', 1)      'BN'  -> ('.2', 1)
        'CN'   -> ('.3', 1)      'ABCN'-> ('.1.2.3', 3)
    """
    clean = phases_str.replace("N", "").replace("D", "").strip()
    mapping = {"A": "1", "B": "2", "C": "3"}
    parts = [mapping[c] for c in clean if c in mapping]
    if not parts:
        parts = ["1", "2", "3"]
    return "." + ".".join(parts), len(parts)


def safe_name(name):
    """Sanitise a GridLAB-D name for use as an OpenDSS element name."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if s and (s[0] == "_" or s[0].isdigit()):
        s = "E" + s
    return s


def gfloat(value, default=0.0):
    """Parse a GLM numeric value, stripping trailing units (e.g. '11.59 m^2')."""
    if value is None:
        return float(default)
    s = str(value).strip()
    m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else float(default)


# ==================================================================
# NETWORK BUILDER
# ==================================================================

def build_elermorevale(glm_dir, common_dir, skip_generators=False):
    """
    Parse all Elermorevale GLM files + common/Line Configs.glm,
    then build the complete OpenDSS model.

    skip_generators : if True, omit PV generators, batteries, and
        RegControl. Use this for profile-driven simulation where
        LoadShapes already carry the net grid profile p = l − g − b,
        so adding separate Generator/Storage elements would double-count.

    Returns a dict of element counts.
    """
    cmd = dss.Text

    # ---- Parse all source files ----
    logger.info("Parsing GLM files from %s ...", glm_dir)
    all_objs = parse_all_glm(glm_dir)
    logger.info("Parsed %d GridLAB-D objects", len(all_objs))

    logger.info("Parsing line configs from %s ...", common_dir)
    conductors, lc_configs = parse_line_configs(common_dir)
    logger.info("Found %d conductors, %d line configurations",
                len(conductors), len(lc_configs))

    linecodes = extract_impedances(conductors, lc_configs)
    logger.info("Extracted %d linecodes with real impedance data", len(linecodes))

    # Index parsed objects by type for easy lookup
    by_type = {}
    for src, otype, props in all_objs:
        by_type.setdefault(otype, []).append((src, props))

    # ----------------------------------------------------------------
    # BUILD PARENT CHAIN RESOLVER
    # ----------------------------------------------------------------
    # In GridLAB-D the object hierarchy is:
    #   load → parent=triplex_meter → parent=triplex_node → connected by lines
    # We need to follow this chain so loads connect to actual network buses.
    # Build a universal lookup: object_name → parent_name for every object
    # that declares a parent, then resolve recursively.
    parent_of = {}
    for otype_list in by_type.values():
        for _, p in otype_list:
            obj_name = p.get("name", "")
            obj_parent = p.get("parent", "")
            if obj_name and obj_parent:
                parent_of[obj_name] = obj_parent

    def resolve_bus(name, max_depth=10):
        """Follow the parent chain to find the ultimate network bus."""
        visited = set()
        current = name
        for _ in range(max_depth):
            if current not in parent_of or current in visited:
                return current
            visited.add(current)
            current = parent_of[current]
        return current

    logger.info("Built parent chain resolver with %d entries", len(parent_of))

    # Element counters
    n_lines = n_sw = n_tx = n_loads = n_pv = n_batt = 0
    unmapped = set()

    # ================================================================
    # 1. CIRCUIT + 132 kV SOURCE
    # ================================================================
    cmd.Command = "Clear"
    cmd.Command = (
        "New Circuit.Elermorevale "       # circuit name
        "basekv=132 "                     # 132 kV base
        "pu=1.0 "                         # source at nominal voltage
        "phases=3 "                       # three-phase
        "bus1=Jesmond_132kV_Bus "          # source bus name (from GLM)
        "Isc3=20000 Isc1=21000"           # stiff source impedance
    )

    # ================================================================
    # 2. LINECODES (from common/Line Configs.glm)
    # ================================================================
    logger.info("Defining %d linecodes ...", len(linecodes))
    for lc_name, (r1, x1, amps, nph) in linecodes.items():
        cmd.Command = (
            f"New Linecode.{safe_name(lc_name)} nphases={nph} "
            f"r1={r1} x1={x1} "              # positive-sequence impedance
            f"r0={r1*3} x0={x1*3} "           # zero-sequence ~ 3x positive
            f"units=km normamps={amps}"       # Ohm/km (converted from Ohm/mile)
        )
    # Fallbacks for any config not found in the common files
    cmd.Command = (
        "New Linecode.fallback_3ph nphases=3 "
        "r1=0.4 x1=0.25 r0=1.2 x0=0.75 units=km normamps=200"
    )
    cmd.Command = (
        "New Linecode.fallback_1ph nphases=1 "
        "r1=1.2 x1=0.3 r0=3.6 x0=0.9 units=km normamps=80"
    )

    # ================================================================
    # 3. ZONE SUBSTATION: 132 / 11 kV TRANSFORMER + OLTC
    # ================================================================
    logger.info("Building 132/11 kV zone substation ...")

    # Main transformer: 50 MVA, Delta primary, Wye-grounded secondary
    # From GLM: impedance 0.0075 + 0.358j pu -> %R=0.75, %X=35.8
    cmd.Command = (
        "New Transformer.TXZoneSub phases=3 windings=2 "
        "buses=[Jesmond_132kV_Bus, BusZoneSubOLTC] "
        "conns=[delta, wye] "                 # Dyn connection
        "kvs=[132, 11] "                      # 132 kV / 11 kV
        "kvas=[50000, 50000] "                # 50 MVA
        "%Rs=[0.375, 0.375] "                 # total %R = 0.75
        "xhl=35.8"                            # leakage reactance
    )

    # OLTC: modelled as a unity-ratio autotransformer with RegControl
    # Simplification of the GridLAB-D LDC lookup_table + aggregate_transform
    cmd.Command = (
        "New Transformer.OLTC phases=3 windings=2 "
        "buses=[BusZoneSubOLTC, BusZoneSub11kV] "
        "conns=[wye, wye] "
        "kvs=[11, 11] "                       # unity ratio; taps do the work
        "kvas=[50000, 50000] "
        "%Rs=[0.001, 0.001] xhl=0.01 "        # near-zero impedance
        "taps=[1.0, 1.0]"
    )
    if not skip_generators:
        cmd.Command = (
            "New RegControl.OLTC_ctrl "
            "transformer=OLTC winding=2 "
            "vreg=110 band=2 "                    # target voltage on 120V base
            "ptratio=100 "                        # PT ratio for 11 kV
            "delay=3 "                            # tap-change delay in seconds
            "maxtapchange=1 tapnum=0"
        )

    # ================================================================
    # 4. ALL LINES (11 kV + LV overhead + LV underground)
    # ================================================================
    logger.info("Building lines ...")
    for src, p in (by_type.get("overhead_line", []) +
                   by_type.get("underground_line", []) +
                   by_type.get("triplex_line", [])):
        name = p.get("name", f"line_{n_lines}")
        from_b = p.get("from", "")
        to_b = p.get("to", "")
        phases = p.get("phases", "ABC")
        config = p.get("configuration", "")
        length_raw = p.get("length", "1")

        # Parse length — strip any unit suffix like " m" or " ft"
        length_m = float(re.sub(r"[^\d.]", "", length_raw)) if length_raw else 1.0

        # Determine phase suffix and count
        suffix, nph = glm_phases_to_dss(phases)
        bus1 = from_b + suffix
        bus2 = to_b + suffix

        # Match to a real linecode, or fall back
        if config in linecodes:
            lc = safe_name(config)
        else:
            if config:
                unmapped.add(config)
            lc = "fallback_3ph" if nph == 3 else "fallback_1ph"

        cmd.Command = (
            f"New Line.{safe_name(name)} "
            f"bus1={bus1} bus2={bus2} "
            f"linecode={lc} "
            f"length={length_m:.2f} units=m"
        )
        n_lines += 1

    # ================================================================
    # 5. SWITCHES + FUSES (modelled as 1 mm lines)
    # ================================================================
    logger.info("Building switches and fuses ...")
    for src, p in by_type.get("switch", []) + by_type.get("fuse", []):
        name = p.get("name", f"sw_{n_sw}")
        from_b = p.get("from", "")
        to_b = p.get("to", "")
        phases = p.get("phases", "ABC")
        status = p.get("status", "CLOSED")

        suffix, nph = glm_phases_to_dss(phases)
        lc = "fallback_3ph" if nph == 3 else "fallback_1ph"
        sn = safe_name(name)

        cmd.Command = (
            f"New Line.{sn} "
            f"bus1={from_b}{suffix} bus2={to_b}{suffix} "
            f"linecode={lc} length=0.001 units=m"
        )
        # Open the switch if the GLM says it's open
        if "OPEN" in status.upper():
            cmd.Command = f"Open Line.{sn} 1"
        n_sw += 1

    # ================================================================
    # 6. DISTRIBUTION TRANSFORMERS (11 kV / 433 V)
    # ================================================================
    logger.info("Building distribution transformers ...")

    # Index transformer_configuration objects by name
    tx_cfgs = {}
    for _, p in by_type.get("transformer_configuration", []):
        tx_cfgs[p.get("name", "")] = p

    for src, p in by_type.get("transformer", []):
        name = p.get("name", "")
        if "TXZoneSub" in name:           # zone sub already built above
            continue
        from_b = p.get("from", "")        # 11 kV bus
        to_b = p.get("to", "")            # LV bus
        cfg_name = p.get("configuration", "")
        cfg = tx_cfgs.get(cfg_name, {})

        # Extract rated values from the config
        kva = float(cfg.get("power_rating", "300"))
        pri_kv = float(cfg.get("primary_voltage", "11000")) / 1000.0
        sec_kv = float(cfg.get("secondary_voltage", "433")) / 1000.0
        r_pct = float(cfg.get("resistance", "0.012")) * 100.0   # pu -> %
        x_pct = float(cfg.get("reactance", "0.039")) * 100.0

        cmd.Command = (
            f"New Transformer.{safe_name(name)} phases=3 windings=2 "
            f"buses=[{from_b}, {to_b}] "
            f"conns=[delta, wye] "            # Dyn connection per GLM
            f"kvs=[{pri_kv:.3f}, {sec_kv:.3f}] "
            f"kvas=[{kva}, {kva}] "
            f"%Rs=[{r_pct/2:.3f}, {r_pct/2:.3f}] "
            f"xhl={x_pct:.3f}"
        )
        n_tx += 1

    # ================================================================
    # 7. RESIDENTIAL LOADS
    # ================================================================
    logger.info("Building %d loads ...", len(by_type.get("load", [])))
    n_resolved = 0
    for src, p in by_type.get("load", []):
        name = p.get("name", f"load_{n_loads}")
        parent = p.get("parent", "")      # may be a meter, not a bus
        phases = p.get("phases", "AN")
        nom_v = gfloat(p.get("nominal_voltage", "240"), 240.0)

        # Resolve parent chain: load → meter → node (the actual bus)
        resolved = resolve_bus(parent) if parent else parent
        if resolved != parent and parent:
            n_resolved += 1

        suffix, nph = glm_phases_to_dss(phases)
        bus = (resolved + suffix) if resolved else (name + suffix)
        kv_ph = nom_v / 1000.0            # 240 V -> 0.240 kV

        # Default load: 3 kW at 0.95 pf, constant-P model.
        # The GridLAB-D model uses temperature-dependent load transforms
        # which are not ported; use LoadShapes for time-series runs.
        cmd.Command = (
            f"New Load.{safe_name(name)} "
            f"bus1={bus} phases={nph} "
            f"kv={kv_ph:.4f} kw=3.0 pf=0.95 "
            f"model=1 "                       # constant P+Q
            f"status=variable "               # allows negative (export)
            f"vminpu=0.85 vmaxpu=1.15"
        )
        n_loads += 1

    if n_resolved > 0:
        logger.info("Resolved %d load parents through meter/node chain",
                    n_resolved)

    # ================================================================
    # 8. PV GENERATORS + BATTERIES (skip in profile mode)
    # ================================================================
    # In profile-driven mode, LoadShapes already contain the net grid
    # profile p = l − g − b. Adding Generator/Storage elements on top
    # would double-count PV generation and battery dispatch.
    if skip_generators:
        logger.info("Skipping PV/batteries/RegControl (profile mode — "
                    "net load already includes PV and battery effects)")
    else:
        logger.info("Building %d PV systems ...", len(by_type.get("solar", [])))

        # Build inverter -> parent bus lookup
        inv_parents = {}
        for _, p in by_type.get("inverter", []):
            inv_parents[p.get("name", "")] = p.get("parent", "")

        for src, p in by_type.get("solar", []):
            name = p.get("name", f"pv_{n_pv}")
            parent_inv = p.get("parent", "")  # parent is the inverter object
            area = gfloat(p.get("area", "25"), 25.0)
            rated_kw = area * 0.20            # ~200 W/m^2 at STC

            # Resolve chain: solar -> inverter -> meter -> node (actual bus)
            bus = inv_parents.get(parent_inv, parent_inv)
            bus = resolve_bus(bus)

            cmd.Command = (
                f"New Generator.{safe_name(name)} "
                f"bus1={bus}.1 phases=1 "         # single-phase at the service point
                f"kv=0.240 kw={rated_kw:.1f} "
                f"pf=1 model=1"
            )
            n_pv += 1

        # ================================================================
        # 9. BATTERY STORAGE (Redflow units from Generators2.glm)
        # ================================================================
        logger.info("Building %d batteries ...", len(by_type.get("battery", [])))
        for src, p in by_type.get("battery", []):
            name = p.get("name", f"batt_{n_batt}")
            parent = p.get("parent", "")      # LV service-point bus
            parent = resolve_bus(parent) if parent else parent
            p_max = gfloat(p.get("P_Max", "5000"), 5000.0) / 1000.0   # W -> kW
            e_max = gfloat(p.get("E_Max", "10000"), 10000.0) / 1000.0  # Wh -> kWh
            eff = gfloat(p.get("base_efficiency", "0.86"), 0.86) * 100.0

            cmd.Command = (
                f"New Storage.{safe_name(name)} "
                f"bus1={parent}.1 phases=1 "
                f"kv=0.240 "
                f"kwrated={p_max:.1f} "           # max charge/discharge power
                f"kwhrated={e_max:.1f} "          # energy capacity
                f"kwhstored={e_max * 0.5:.1f} "   # initial SOC at 50%
                f"%EffCharge={eff:.1f} "
                f"%EffDischarge={eff:.1f} "
                f"%IdlingkW=0.18 "                # 180 W parasitic (from GLM)
                f"model=1 state=IDLING"           # start idle, dispatch externally
            )
            n_batt += 1

    # ================================================================
    # 10. VOLTAGE BASES + FINALISE
    # ================================================================
    cmd.Command = "Set voltagebases=[132, 11, 0.433]"
    cmd.Command = "Calcvoltagebases"

    if unmapped:
        logger.warning(
            "Fallback impedance used for %d unmapped line configs: %s",
            len(unmapped), unmapped)

    stats = {
        "lines": n_lines,
        "switches_and_fuses": n_sw,
        "distribution_transformers": n_tx,
        "loads": n_loads,
        "pv_systems": n_pv,
        "batteries": n_batt,
        "linecodes_from_real_data": len(linecodes),
        "unmapped_line_configs": len(unmapped),
        "parent_chain_entries": len(parent_of),
    }
    logger.info("Network built: %s", stats)
    return stats


# ==================================================================
# VERIFICATION (original snapshot mode)
# ==================================================================

def solve_snapshot():
    """Run a single snapshot power flow and report key metrics."""
    cmd = dss.Text
    cmd.Command = "Set mode=snapshot"
    cmd.Command = "Set controlmode=off"
    cmd.Command = "Set maxcontroliter=50"
    cmd.Command = "Set maxiterations=100"
    # Re-initialise voltage vector from voltage bases so Newton-Raphson
    # starts from a reasonable flat-start, not a stale zero-voltage state.
    cmd.Command = "Calcvoltagebases"
    try:
        dss.ActiveCircuit.Solution.Solve()
    except Exception as exc:
        # #485 (Max Control Iterations Exceeded) is a warning, not a hard
        # failure — the power-flow result is still usable.
        if "485" in str(exc):
            logger.warning("Control loop didn't settle (#485); using last solution.")
        else:
            raise

    converged = dss.ActiveCircuit.Solution.Converged
    logger.info("Power flow converged: %s", converged)
    if not converged:
        logger.warning("Power flow did NOT converge. Check topology.")
        return False

    # Total circuit losses
    losses = dss.ActiveCircuit.Losses       # returns (watts, vars)
    logger.info("Total losses: %.1f kW, %.1f kvar",
                losses[0] / 1000, losses[1] / 1000)

    # Bus voltage summary
    all_v = np.array(dss.ActiveCircuit.AllBusVmagPu)
    valid = all_v[all_v > 0.01]             # filter zero-injection buses
    if len(valid) > 0:
        logger.info("Bus voltages (p.u.): min=%.4f  mean=%.4f  max=%.4f",
                    valid.min(), valid.mean(), valid.max())
        n_over = int(np.sum(valid > 1.10))
        n_under = int(np.sum(valid < 0.94))
        logger.info("Buses outside AS 60038: %d over, %d under (of %d total)",
                    n_over, n_under, len(valid))

    # Element counts from the DSS engine
    logger.info(
        "DSS element counts: Lines=%d  Loads=%d  Transformers=%d  Generators=%d",
        dss.ActiveCircuit.Lines.Count,
        dss.ActiveCircuit.Loads.Count,
        dss.ActiveCircuit.Transformers.Count,
        dss.ActiveCircuit.Generators.Count,
    )
    return True


def export_dss_summary(path="elermorevale_summary.txt"):
    """Write the OpenDSS summary report to a text file."""
    dss.Text.Command = f"Export summary {path}"
    logger.info("Summary exported to %s", path)


# ==================================================================
# PROFILE-DRIVEN SIMULATION
# ==================================================================
#
# Everything below this line makes the Elermore Vale model accept
# the same QP-dispatched profile inputs as openDSS_LV_feeder_model.py.
#
# The workflow is:
#   1. Build the full Elermore Vale network (all ~1,785 loads)
#   2. Enumerate load elements in the DSS circuit
#   3. Map ALL loads to the 55 OSQP customers via round-robin recycling
#      (each of the 55 profiles is reused ~32 times, analogous to the
#      paper's approach of duplicating 291 clean Ausgrid customers to
#      fill 845 slots). This gives realistic aggregate loading.
#   4. Override every load to kw=1 with a half-hourly LoadShape
#   5. Attach voltage monitors to an evenly-spaced SUBSET of loads
#      (monitoring all ~1,785 would be excessive) + substation TX
#   6. Run 48-step daily simulation, collect results, and plot
#
# This replaces the previous approach that zeroed all non-profiled
# loads, which left the 50 MVA network at <0.06% utilisation and
# produced flat, uninformative results.
# ==================================================================

# --- Validation constants (AS 60038) ---
T = 48                  # half-hourly intervals per day
DT = 0.5                # hours per interval
V_NOM = 230.0           # AS 60038 nominal phase-to-neutral voltage
V_UPPER_PU = 1.10       # statutory upper limit: +10 %
V_LOWER_PU = 0.94       # statutory lower limit: −6 %

# Module-level output directory for plots. None = interactive display.
OUTPUT_DIR = None


# ==================================================================
# LOAD PROFILES FROM DISK
# ==================================================================

def load_profiles_from_csv(csv_path):
    """
    Read the long-format CSV produced by osqp_daily_v2.save_profiles()
    and reconstruct the nested dictionary:
        {customer_id: [day_profile_dict, ...]}

    Each day_profile_dict has keys:
        date, load, pv, battery, grid, soc, savings
    """
    import pandas as pd
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


# ==================================================================
# CUSTOMER-TO-LOAD MAPPING
# ==================================================================

def get_network_load_names():
    """
    Enumerate all Load element names from the currently-loaded DSS circuit.
    Must be called after build_elermorevale().
    """
    names = []
    i = dss.ActiveCircuit.Loads.First
    while i:
        names.append(dss.ActiveCircuit.Loads.Name)
        i = dss.ActiveCircuit.Loads.Next
    return sorted(names)


def map_customers_to_network_loads(customer_ids, load_names):
    """
    Map EVERY Load element in the Elermore Vale network to one of the
    55 OSQP customer profiles, cycling round-robin. This mirrors the
    paper's approach of duplicating 291 clean Ausgrid customers to fill
    845 aggregate-member slots.

    With 55 customers and ~1,785 loads, each customer profile is reused
    about 32 times. The result is realistic aggregate loading: peak
    aggregate demand will be ~32× what the 55 customers alone produce.

    Returns:
        load_customer_map : {load_element_name: customer_id}
            — maps EVERY load to one of the 55 customer IDs
    """
    sorted_loads = sorted(load_names)
    sorted_customers = sorted(customer_ids)
    n_loads = len(sorted_loads)
    n_cust = len(sorted_customers)

    if n_cust == 0 or n_loads == 0:
        return {}

    mapping = {}
    for i, lname in enumerate(sorted_loads):
        cid = sorted_customers[i % n_cust]   # round-robin reuse
        mapping[lname] = cid

    logger.info("Mapped ALL %d network loads to %d OSQP customers "
                "(each profile reused ~%d times)",
                n_loads, n_cust, n_loads // n_cust)
    return mapping


def select_monitored_loads(load_customer_map, n_monitors=100):
    """
    Pick an evenly-spaced subset of loads for voltage monitoring.
    Monitoring all ~1,785 loads would create excessive monitor overhead.
    The subset is spread across the sorted load list so it spans the
    full feeder topology.

    Returns:
        monitored_loads : list of load element names (subset of keys)
    """
    all_loads = sorted(load_customer_map.keys())
    n = len(all_loads)
    if n <= n_monitors:
        return all_loads                     # monitor everything if small
    step = n // n_monitors
    selected = [all_loads[i * step] for i in range(n_monitors)]
    logger.info("Selected %d monitored loads out of %d (every %d-th)",
                len(selected), n, step)
    return selected


# ==================================================================
# MONITORS
# ==================================================================

def add_monitors(monitored_loads):
    """
    Attach OpenDSS monitors after the network has been built:
      - One power monitor on the zone substation transformer (TXZoneSub,
        terminal 2 = 11 kV side), recording P and Q per phase.
      - One voltage monitor per profiled load element.
    """
    cmd = dss.Text

    # Substation transformer secondary: mode=1 records power, ppolar=no
    # gives P,Q in rectangular form rather than S,angle.
    cmd.Command = (
        "New Monitor.tx_power element=Transformer.TXZoneSub "
        "terminal=2 mode=1 ppolar=no"
    )

    # Per-load voltage monitors: mode=0 records voltage magnitudes.
    for lname in monitored_loads:
        cmd.Command = (
            f"New Monitor.v_{lname} element=Load.{lname} "
            f"terminal=1 mode=0"
        )


# ==================================================================
# LOADSHAPE ATTACHMENT
# ==================================================================

def attach_loadshapes(load_customer_map, profiles, day_idx=0):
    """
    Create a LoadShape object for each load in the network using the
    QP-dispatched grid profile p_k = l_k − g_k − b_k, then bind it
    to the Load element via the daily= property.

    Every load in load_customer_map gets a profile (all ~1,785 loads).
    The load's kw is overridden to 1 (base multiplier) and pf to 1,
    so the LoadShape mult values carry the actual signed kW directly.

    Returns the date string of the simulated day.
    """
    cmd = dss.Text
    date_str = None

    for lname, cid in load_customer_map.items():
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
            f"New Loadshape.shape_{lname} npts={T} "
            f"minterval=30 mult=({mult_str})"
        )
        # Override load to use kw=1 base with shape as multiplier
        cmd.Command = f"Load.{lname}.kw=1"
        cmd.Command = f"Load.{lname}.pf=1"
        cmd.Command = f"Load.{lname}.daily=shape_{lname}"

    return date_str


def attach_baseline_shapes(load_customer_map, profiles, day_idx=0):
    """
    Same as attach_loadshapes but uses the no-battery baseline:
        p_baseline_k = l_k − g_k    (i.e. b_k = 0 for all k)
    Every load in load_customer_map gets a baseline profile.
    """
    cmd = dss.Text
    date_str = None

    for lname, cid in load_customer_map.items():
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
            f"New Loadshape.shape_{lname} npts={T} "
            f"minterval=30 mult=({mult_str})"
        )
        cmd.Command = f"Load.{lname}.kw=1"
        cmd.Command = f"Load.{lname}.pf=1"
        cmd.Command = f"Load.{lname}.daily=shape_{lname}"

    return date_str


# ==================================================================
# SIMULATION ENGINE
# ==================================================================

def run_daily():
    """Execute one full day of 48 sequential 30-minute power flows."""
    cmd = dss.Text
    cmd.Command = f"Set mode=daily stepsize=30m number={T}"
    cmd.Command = "Set controlmode=off"
    cmd.Command = "Set maxcontroliter=50"
    cmd.Command = "Set maxiterations=100"
    # Re-initialise voltage vector so Newton-Raphson starts from
    # flat-start (1.0 p.u.), not a stale zero-voltage state.
    cmd.Command = "Calcvoltagebases"
    try:
        dss.ActiveCircuit.Solution.Solve()
    except Exception as exc:
        if "485" in str(exc):
            logger.warning("Daily run: control loop didn't settle (#485); using last solution.")
        else:
            raise
    # Flush monitor sample buffers so Channel() returns recorded data.
    # Without this, some dss-python versions return empty/zero arrays.
    dss.ActiveCircuit.Monitors.SaveAll()


def collect_voltages(monitored_loads):
    """
    Read the voltage monitor for each monitored load.
    Returns dict {load_name: numpy array of shape (T,) in per-unit}.

    The monitor records absolute voltage in volts (mode=0, channel 1).
    We convert to per-unit relative to V_NOM (230 V, AS 60038 nominal).
    """
    voltages = {}
    n_empty = 0
    for lname in monitored_loads:
        try:
            dss.ActiveCircuit.Monitors.Name = f"v_{lname}"
            v_mag = np.array(dss.ActiveCircuit.Monitors.Channel(1))
        except Exception:
            v_mag = np.zeros(T)

        if len(v_mag) == 0:
            v_mag = np.zeros(T)
            n_empty += 1
        elif len(v_mag) != T:
            # Pad or truncate to exactly T points
            padded = np.zeros(T)
            n = min(len(v_mag), T)
            padded[:n] = v_mag[:n]
            v_mag = padded

        voltages[lname] = v_mag / V_NOM

    if n_empty > 0:
        logger.warning("%d of %d voltage monitors returned empty data",
                       n_empty, len(monitored_loads))

    # Debug: log first monitor's raw values to help diagnose V_NOM issues
    if monitored_loads:
        first = monitored_loads[0]
        raw = voltages[first] * V_NOM       # undo the division
        logger.info("Monitor debug — %s raw volts: min=%.1f mean=%.1f max=%.1f",
                    first, raw.min(), raw.mean(), raw.max())

    return voltages


def collect_tx_power():
    """
    Read the zone substation transformer power monitor.
    Returns (p_kw, q_kvar) — numpy arrays of shape (T,) summed across
    three phases. Positive = power flowing MV → LV (import from grid).
    """
    dss.ActiveCircuit.Monitors.Name = "tx_power"
    # mode=1, ppolar=no → channels: P1, Q1, P2, Q2, P3, Q3
    p1 = np.array(dss.ActiveCircuit.Monitors.Channel(1))
    p2 = np.array(dss.ActiveCircuit.Monitors.Channel(3))
    p3 = np.array(dss.ActiveCircuit.Monitors.Channel(5))
    q1 = np.array(dss.ActiveCircuit.Monitors.Channel(2))
    q2 = np.array(dss.ActiveCircuit.Monitors.Channel(4))
    q3 = np.array(dss.ActiveCircuit.Monitors.Channel(6))
    return p1 + p2 + p3, q1 + q2 + q3


def collect_losses():
    """Total circuit losses at the final timestep, returned as (kW, kvar)."""
    losses = dss.ActiveCircuit.Losses          # OpenDSS returns (W, var)
    return losses[0] / 1000.0, losses[1] / 1000.0


def simulate_scenario(glm_dir, common_dir, load_customer_map,
                      monitored_loads, profiles,
                      day_idx, use_baseline=False):
    """
    End-to-end pipeline for one day under one scenario:
        build network → add monitors → attach load shapes → solve → collect.

    load_customer_map : {load_name: customer_id} for ALL loads (~1,785)
    monitored_loads   : list of load names for voltage monitoring (subset)

    The network is rebuilt from scratch each call to avoid stale state.
    """
    build_elermorevale(glm_dir, common_dir, skip_generators=True)

    add_monitors(monitored_loads)

    if use_baseline:
        date_str = attach_baseline_shapes(load_customer_map, profiles, day_idx)
    else:
        date_str = attach_loadshapes(load_customer_map, profiles, day_idx)

    run_daily()

    voltages = collect_voltages(monitored_loads)
    tx_p, tx_q = collect_tx_power()
    loss_kw, loss_kvar = collect_losses()

    all_v = np.array(list(voltages.values()))          # shape (n_monitored, T)
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


def simulate_day_comparison(glm_dir, common_dir, load_customer_map,
                            monitored_loads, profiles, day_idx):
    """Run baseline and QP scenarios for one day and return both."""
    logger.info("Simulating day index %d — baseline ...", day_idx)
    base = simulate_scenario(glm_dir, common_dir, load_customer_map,
                             monitored_loads, profiles, day_idx,
                             use_baseline=True)
    logger.info("Simulating day index %d — QP dispatched ...", day_idx)
    qp = simulate_scenario(glm_dir, common_dir, load_customer_map,
                           monitored_loads, profiles, day_idx,
                           use_baseline=False)
    return base, qp


# ==================================================================
# PLOTTING
# ==================================================================

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _safe_filename(s):
    """Sanitise a string for use as part of a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _finalise_plot(name, date_str=None, subdir=None):
    """
    Either show the current figure interactively or save it to
    OUTPUT_DIR. Closes the figure either way to prevent memory leaks.
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
    Min/max voltage envelope across all monitored loads vs time,
    overlaid with AS 60038 statutory limits.
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
    title = "Elermore Vale — Voltage envelope: Baseline vs QP"
    if date_str:
        title += f" ({date_str})"
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.90, 1.15)
    ax.grid(alpha=0.3)
    _finalise_plot("voltage_envelope", date_str)


def plot_substation_power(base, qp, date_str=None):
    """Zone substation transformer secondary active power, both scenarios."""
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, base["tx_p_kw"], color="salmon", lw=1.5, label="Baseline")
    ax.plot(hours, qp["tx_p_kw"], color="steelblue", lw=1.5, label="QP dispatched")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Substation P (kW)")
    title = "Elermore Vale — Zone sub transformer power"
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
    2D heatmap: monitored-load index (y) × time (x), coloured by voltage.
    Diverging blue-white-red centred at 1.0 p.u.
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
    ax.set_ylabel("Monitored load index")
    title = f"{title_prefix}Elermore Vale — Voltage heatmap"
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
    Heatmap of (baseline − QP) voltage at every (load, interval).

    Positive (red)  = QP brought voltage down (over-voltage relief).
    Negative (blue) = QP brought voltage up (under-voltage relief).
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
    ax.set_ylabel("Monitored load index")
    title = "Elermore Vale — Voltage delta: baseline − QP"
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
    lines.append(f"{'Total (load×interval) points':<30} {base['total_points']:>12d} {qp['total_points']:>12d}")
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


# ==================================================================
# FULL-YEAR SWEEP
# ==================================================================

def run_full_sweep(glm_dir, common_dir, load_customer_map,
                   monitored_loads, profiles,
                   max_days=None, per_day_plots=False):
    """
    Loop over every day in the dataset (optionally capped), run baseline
    vs QP comparison, and accumulate one summary row per day.
    Returns a pandas DataFrame.
    """
    import pandas as pd
    n_days = max(len(days) for days in profiles.values())
    if max_days:
        n_days = min(n_days, max_days)
    logger.info("Running full sweep for %d days", n_days)

    records = []
    for d in range(n_days):
        try:
            base, qp = simulate_day_comparison(
                glm_dir, common_dir, load_customer_map,
                monitored_loads, profiles, d)
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
    ax.set_title("Daily peak zone sub transformer power")
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

    plt.suptitle("Elermore Vale — Full sweep: Baseline vs QP battery dispatch",
                 y=1.01)
    _finalise_plot("sweep_summary")


# ==================================================================
# MAIN
# ==================================================================

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Build the Elermore Vale OpenDSS model from "
                    "GridLAB-D source files, with optional profile-driven "
                    "daily simulation and baseline-vs-QP comparison plots")

    # --- Original arguments (network build) ---
    parser.add_argument(
        "--glm-dir", default="Elermorevale",
        help="Path to the Elermorevale/ directory containing .glm files")
    parser.add_argument(
        "--common-dir", default="common",
        help="Path to the common/ directory containing Line Configs.glm")
    parser.add_argument(
        "--export", action="store_true",
        help="Export the DSS summary report after solving (snapshot mode)")

    # --- Profile-driven simulation arguments ---
    parser.add_argument(
        "--profiles", default=None,
        help="Path to the long-format CSV from osqp_daily_v2. "
             "When provided, runs daily simulation instead of snapshot.")
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
        help="Save figures to disk instead of opening interactive windows")
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to save figures into when --save is set")
    parser.add_argument(
        "--per-day-plots", action="store_true",
        help="With --full, generate per-day plots for every simulated day")

    args = parser.parse_args()

    # --- Validate GLM inputs ---
    if not os.path.isdir(args.glm_dir):
        logger.error("GLM directory not found: %s", args.glm_dir)
        sys.exit(1)
    lc_path = os.path.join(args.common_dir, "Line Configs.glm")
    if not os.path.isfile(lc_path):
        logger.error("Line Configs.glm not found in %s", args.common_dir)
        sys.exit(1)

    # =============================================================
    # MODE 1: Original snapshot (no --profiles)
    # =============================================================
    if args.profiles is None:
        stats = build_elermorevale(args.glm_dir, args.common_dir)
        logger.info("Running snapshot power flow ...")
        ok = solve_snapshot()
        if args.export and ok:
            export_dss_summary()
        logger.info("Done. Model is loaded in the DSS engine.")
        return stats

    # =============================================================
    # MODE 2: Profile-driven daily simulation (--profiles given)
    # =============================================================

    # --- Configure plot output ---
    if args.save:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Plots will be saved to %s/", OUTPUT_DIR)
        import matplotlib
        matplotlib.use("Agg", force=True)

    # --- Load QP profiles ---
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

    # --- Build network once to discover load element names ---
    logger.info("Building network to enumerate loads ...")
    build_elermorevale(args.glm_dir, args.common_dir, skip_generators=True)
    load_names = get_network_load_names()
    logger.info("Found %d load elements in the network", len(load_names))

    # --- Map OSQP customers to network loads ---
    load_customer_map = map_customers_to_network_loads(customer_ids, load_names)

    # --- Select a representative subset for voltage monitoring ---
    monitored_loads = select_monitored_loads(load_customer_map, n_monitors=100)
    logger.info("Voltage monitors on %d loads (of %d total)",
                len(monitored_loads), len(load_customer_map))

    if args.full:
        # --- Full sweep ---
        import pandas as pd
        sweep_df = run_full_sweep(
            args.glm_dir, args.common_dir,
            load_customer_map, monitored_loads, profiles,
            max_days=args.max_days,
            per_day_plots=args.per_day_plots)

        csv_path = os.path.join(OUTPUT_DIR, "opendss_sweep_results.csv") \
            if args.save else "opendss_sweep_results.csv"
        sweep_df.to_csv(csv_path, index=False)
        logger.info("Sweep results saved to %s", csv_path)
        plot_sweep_results(sweep_df)
    else:
        # --- Representative days ---
        n_days = max(len(d) for d in profiles.values())
        summer_idx = args.summer_day if args.summer_day is not None \
            else min(190, n_days - 1)
        winter_idx = args.winter_day if args.winter_day is not None else 0

        for label, day_idx in [("Summer", summer_idx), ("Winter", winter_idx)]:
            logger.info("=== %s day (index %d) ===", label, day_idx)
            base, qp = simulate_day_comparison(
                args.glm_dir, args.common_dir,
                load_customer_map, monitored_loads, profiles, day_idx)
            date_str = base["date"] or f"day_{day_idx}"

            plot_daily_summary_table(base, qp, date_str)
            plot_voltage_envelope(base, qp, date_str)
            plot_substation_power(base, qp, date_str)
            plot_voltage_heatmap(base, date_str, title_prefix="Baseline: ")
            plot_voltage_heatmap(qp, date_str, title_prefix="QP: ")
            plot_voltage_delta_heatmap(base, qp, date_str)

    logger.info("Done.")


if __name__ == "__main__":
    main()