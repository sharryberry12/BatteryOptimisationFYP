"""
compare_voltages.py -- Level 4 cross-validation: GridLAB-D vs OpenDSS.

Prerequisites (both from the repo root):
    python validation/gen_harness.py
    gridlabd validation/harness.glm          # writes validation/voltages_gld.csv

This script then:
  1. solves the OpenDSS translation at the same operating point
     (load-only model, 1 kW @ 0.95 pf; loads absent from the GridLAB-D
     harness -- those outside the subs/ files -- are zeroed for parity);
  2. joins the two solutions per (bus, phase) on the RAW lowercased GLM
     node name: build_elermorevale creates buses with raw GLM names --
     only ELEMENT names are sanitised through safe_name() -- and OpenDSS
     lowercases bus names internally;
  3. reports |V| deviation statistics split by voltage level and writes
     validation/voltage_comparison.csv;
  4. exits non-zero if the comparison itself is broken: join coverage of
     the live OpenDSS node-phases below MIN_DSS_COVERAGE, any bus dead in
     one engine but live in the other, or a collapsed GridLAB-D node.
     A previous revision only PRINTED an unmatched counter, which let a
     join-key bug silently drop 77 of the 101 11 kV backbone buses while
     the script still reported a clean ~1 % agreement.

Expected agreement (MODEL_VERIFICATION.md): ~1 % mean at 11 kV, a few %
at LV extremities -- the translation drops z-matrix mutual coupling and
uses estimated LV reactances, so the measured gap IS the result.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import elermorevale_openDSS as ev  # noqa: E402

GLD_DUMP = REPO / "validation" / "voltages_gld.csv"
OUT_CSV = REPO / "validation" / "voltage_comparison.csv"

LOAD_KW = 1.0
# Line-to-neutral voltage bases (V); bases differ by >25x so nearest-ratio
# inference per node is unambiguous.
BASES = {"132kV": 76210.23553, "11kV": 6350.85296, "LV": 433.0 / math.sqrt(3)}
PHASES = {"A": "1", "B": "2", "C": "3"}

# Fatal-condition thresholds. The comparison must FAIL, not just print,
# when its own inputs are broken.
DEAD_PU = 0.01            # DSS |V| below this = de-energised node
GLD_COLLAPSE_V = 100.0    # GLD 0 < |V| < this = collapsed node (the lowest
#                           L-N base is ~250 V; sagging below 100 V is not a
#                           power-flow result). Exactly 0 V = absent phase.
MIN_DSS_COVERAGE = 0.99   # live DSS node-phases that must find a GLD partner


def subs_load_names():
    """Names of the loads present in the GridLAB-D harness (subs/ files)."""
    names = set()
    for fp in sorted((REPO / "Elermorevale" / "subs").glob("*.glm")):
        for otype, props in ev.parse_glm(str(fp)):
            if otype == "load" and props.get("name"):
                names.add(props["name"])
    return names


def solve_opendss(harness_loads):
    ev.build_elermorevale(str(REPO / "Elermorevale"), str(REPO / "common"),
                          skip_generators=True)
    ev.dss.Text.Command = f"BatchEdit Load..* kW={LOAD_KW}"
    ckt = ev.dss.ActiveCircuit
    # Load ELEMENT names are sanitised by the builder, so the keep-set is
    # safe_name'd; bus names (the join key) are raw.
    keep = {ev.safe_name(n).lower() for n in harness_loads}
    zeroed = []
    idx = ckt.Loads.First
    while idx:
        if ckt.Loads.Name.lower() not in keep:
            zeroed.append(ckt.Loads.Name)
        idx = ckt.Loads.Next
    for name in zeroed:
        ev.dss.Text.Command = f"Load.{name}.kW=0.0001"
        ev.dss.Text.Command = f"Load.{name}.kvar=0"
    assert ev.solve_snapshot(), "OpenDSS power flow did not converge"
    vpu = dict(zip((n.lower() for n in ckt.AllNodeNames),
                   ckt.AllBusVmagPu))
    return vpu, zeroed


def load_gld_pu():
    """({(raw_bus_name_lower, phase_digit): (v_pu, level)}, collapsed).

    Absent phases dump as exactly 0 V and are skipped; a nonzero magnitude
    below GLD_COLLAPSE_V is a collapsed node -- a solver failure, not an
    absent phase -- and is returned separately so main() can fail on it.
    """
    result, collapsed = {}, []
    with open(GLD_DUMP, newline="") as fh:
        fh.readline()                       # '# ... run at ...' banner
        for row in csv.DictReader(fh):
            name = row["node_name"].lower()
            for ph, digit in PHASES.items():
                mag = float(row[f"volt{ph}_mag"])
                if mag < 1.0:               # phase not present on this node
                    continue
                if mag < GLD_COLLAPSE_V:    # present but collapsed
                    collapsed.append(f"{name}.{digit}")
                    continue
                level = min(BASES, key=lambda k: abs(mag / BASES[k] - 1.0))
                result[(name, digit)] = (mag / BASES[level], level)
    return result, collapsed


def main():
    harness_loads = subs_load_names()
    print(f"loads in GridLAB-D harness: {len(harness_loads)}")

    dss_vpu, zeroed = solve_opendss(harness_loads)
    print(f"OpenDSS loads zeroed for parity (outside subs/): {len(zeroed)}")

    gld_vpu, gld_collapsed = load_gld_pu()
    print(f"GridLAB-D node-phases: {len(gld_vpu)}, "
          f"OpenDSS node-phases: {len(dss_vpu)}")

    # ---- join (GLD -> DSS) -------------------------------------------
    rows, no_dss_bus, dead_disagree = [], 0, []
    for (bus, ph), (v_gld, level) in gld_vpu.items():
        v_dss = dss_vpu.get(f"{bus}.{ph}")
        if v_dss is None:
            no_dss_bus += 1     # loads/meters: GLD nodes, never DSS buses
            continue
        if v_dss < DEAD_PU:     # live in GridLAB-D, dead in OpenDSS
            dead_disagree.append(f"{bus}.{ph}")
            continue
        rows.append({"bus": bus, "phase": ph, "level": level,
                     "v_gld_pu": round(v_gld, 6), "v_dss_pu": round(v_dss, 6),
                     "delta_pu": round(v_dss - v_gld, 6)})

    # ---- coverage (DSS -> GLD) ---------------------------------------
    # Checked from the DSS side because it is the strict subset: every DSS
    # bus is a GLM object name, so every LIVE DSS node-phase must find a
    # GLD partner. The GLD side legitimately holds extras (loads and
    # intermediate meters are nodes in GridLAB-D but not buses in OpenDSS).
    gld_keys = {f"{b}.{p}" for (b, p) in gld_vpu}
    live_dss = [k for k, v in dss_vpu.items() if v >= DEAD_PU]
    unjoined_dss = sorted(k for k in live_dss if k not in gld_keys)
    coverage = 1.0 - len(unjoined_dss) / max(len(live_dss), 1)

    if not rows:
        raise SystemExit("no node-phases matched between the two solutions "
                         "-- check that both prerequisite steps ran")
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"matched node-phases: {len(rows)}")
    print(f"  GLD-only node-phases (loads/meters, expected): {no_dss_bus}")
    print(f"  DSS join coverage: {coverage:.2%} "
          f"({len(unjoined_dss)} live DSS node-phases unmatched)")
    for level in ("132kV", "11kV", "LV"):
        d = np.array([abs(r["delta_pu"]) for r in rows if r["level"] == level])
        if d.size == 0:
            continue
        print(f"  {level:6s} n={d.size:5d}  |dV| mean={d.mean()*100:.3f}%  "
              f"p95={np.percentile(d, 95)*100:.3f}%  max={d.max()*100:.3f}%")
    worst = sorted(rows, key=lambda r: -abs(r["delta_pu"]))[:5]
    print("worst 5:")
    for r in worst:
        print(f"  {r['bus']}.{r['phase']} [{r['level']}] "
              f"gld={r['v_gld_pu']:.4f} dss={r['v_dss_pu']:.4f} "
              f"delta={r['delta_pu']*100:+.2f}%")
    print(f"written: {OUT_CSV.relative_to(REPO)}")

    # ---- fatal conditions (after the report so the stats stay visible) --
    failures = []
    if dead_disagree:
        failures.append(
            f"{len(dead_disagree)} node-phases live in GridLAB-D but dead "
            f"(<{DEAD_PU} pu) in OpenDSS, e.g. {dead_disagree[:5]} -- the "
            "all-zero Converged=True defect (MODEL_VERIFICATION.md), not a "
            "join miss")
    if gld_collapsed:
        failures.append(
            f"{len(gld_collapsed)} collapsed GridLAB-D node-phases "
            f"(0 < |V| < {GLD_COLLAPSE_V:.0f} V), e.g. {gld_collapsed[:5]}")
    if coverage < MIN_DSS_COVERAGE:
        failures.append(
            f"DSS join coverage {coverage:.2%} is below the "
            f"{MIN_DSS_COVERAGE:.0%} bound, e.g. {unjoined_dss[:5]} -- "
            "join-key regression?")
    if failures:
        raise SystemExit("VALIDATION INVALID:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
