"""
gen_harness.py -- generate the GridLAB-D validation harness (Level 4).

Produces validation/stripped/*.glm + validation/harness.glm: the Elermore
Vale network at one frozen operating point, comparable to the OpenDSS
Level 3 configuration (load-only model, 1 kW @ 0.95 pf per household):

- zone_stripped.glm : SWING source pinned to 1.0 pu on the transformer's
  rated 132 kV side (the original runs the source at 1.053 pu), OLTC
  regulator forced to MANUAL with taps at neutral -- matching the OpenDSS
  model's unity-ratio OLTC with controls off.
- subs/*.glm copies : solar + inverter blocks removed (matches
  skip_generators=True), every load's ZIP/schedule power properties
  replaced with constant_power_<phase> lines totalling 1000 + j328.7 VA
  (1 kW, 0.95 pf) split evenly across the load's declared phases -- the
  TOTAL matches the OpenDSS side's `BatchEdit Load..* kW=1.0`, which is
  1 kW per load, not per phase.
- harness.glm       : one-instant clock + powerflow module + includes +
  a voltdump writing every node voltage to voltages_gld.csv.

Run:  python validation/gen_harness.py
Then: gridlabd validation/harness.glm   (from the repo root)
"""

import cmath
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLM = REPO / "Elermorevale"
OUT = Path(__file__).resolve().parent
STRIPPED = OUT / "stripped"

LOAD_KW = 1.0                       # match tests/test_physics_sanity.REALISTIC_KW
LOAD_PF = 0.95
VLN_132 = 76210.23553               # transformer rated primary, V line-neutral

# Load properties tied to the temperature-transform machinery (runtime
# classes) or prior power specs -- all removed and replaced.
POWER_PROP = re.compile(
    r"^\s*(base_power|power_fraction|current_fraction|impedance_fraction|"
    r"power_pf|current_pf|impedance_pf|constant_power)_[ABC](_real)?\b")

GEN_BLOCK = re.compile(r"object\s+(solar|inverter)\s*\{[^}]*\}\s*;?", re.DOTALL)
LOAD_BLOCK = re.compile(r"(object\s+load\s*\{)([^}]*)(\})", re.DOTALL)
UG_COND_BLOCK = re.compile(
    r"(object\s+underground_line_conductor\s*\{)([^}]*)(\})", re.DOTALL)
SHIELD_PROP = re.compile(r"^\s*shield_(gmr|resistance)\b")


def constant_power_va():
    p_va = LOAD_KW * 1000.0
    q_va = p_va * math.tan(math.acos(LOAD_PF))
    return p_va, q_va


def rewrite_load(match):
    head, body, tail = match.groups()
    kept = [ln for ln in body.splitlines() if not POWER_PROP.match(ln)]
    phases_m = re.search(r"phases\s+([A-Z]+)", body)
    phases = [c for c in (phases_m.group(1) if phases_m else "") if c in "ABC"]
    if not phases:
        name_m = re.search(r"name\s+(\S+?);", body)
        raise SystemExit(
            f"load {name_m.group(1) if name_m else '<unnamed>'} declares no "
            "phases -- its power would silently become 0 W in the harness")
    # Split the TOTAL across phases: OpenDSS's `kW=1.0` is per LOAD, so a
    # per-phase 1 kW here would put 3-phase loads at 3x the OpenDSS
    # operating point and fold the difference into the voltage comparison.
    p_va, q_va = constant_power_va()
    n_ph = len(phases)
    for ph in phases:
        kept.append(
            f"\tconstant_power_{ph} {p_va / n_ph:.1f}{q_va / n_ph:+.1f}j;")
    return head + "\n".join(kept) + "\n" + tail


def strip_network_text(text, label):
    """Strip generator blocks and rewrite loads in one file's text.

    Pure transformation (no IO) so main() can validate EVERY file before
    writing ANY output, and so tests can exercise the guards. Raises
    SystemExit when the flat-brace regexes leave residue: an unmatched
    `object solar/inverter` opener, or a brace imbalance from a truncated
    match ([^}]* stops at the first '}' inside a nested block)."""
    text, n_gen = GEN_BLOCK.subn("", text)
    text, n_loads = LOAD_BLOCK.subn(rewrite_load, text)
    if re.search(r"object\s+(solar|inverter)\b", text):
        raise SystemExit(
            f"{label}: solar/inverter object survived stripping -- "
            "GEN_BLOCK no longer matches the source syntax")
    if text.count("{") != text.count("}"):
        raise SystemExit(
            f"{label}: brace imbalance after stripping -- a nested block "
            "truncated the flat-brace regex match")
    return text, n_gen, n_loads


def write_line_configs_stripped():
    """GridLAB-D 5.x rejects UG conductors that specify BOTH concentric-
    neutral and tape-shield parameters (the 3.x-era source does). The
    cables declare neutral_strands, so treat them as concentric neutral
    and drop the shield_* lines. GridLAB-D-side only: the OpenDSS
    translation uses conductor_resistance alone."""
    def fix(match):
        head, body, tail = match.groups()
        if "neutral_gmr" in body and ("shield_gmr" in body
                                      or "shield_resistance" in body):
            body = "\n".join(ln for ln in body.splitlines()
                             if not SHIELD_PROP.match(ln)) + "\n"
        return head + body + tail

    text = (REPO / "common" / "Line Configs.glm").read_text(
        encoding="utf-8", errors="replace")
    text, n = UG_COND_BLOCK.subn(fix, text)
    (STRIPPED / "line_configs_stripped.glm").write_text(text, encoding="utf-8")
    return n


def write_zone_stripped():
    """Powerflow-only zone substation: original objects minus the LDC
    runtime-class transforms, source at 1.0 pu, regulator manual/neutral."""
    a120 = cmath.exp(1j * math.radians(-120))
    vb = VLN_132 * a120
    vc = VLN_132 * a120.conjugate()
    (STRIPPED / "zone_stripped.glm").write_text(f"""\
// Generated by validation/gen_harness.py -- do not edit by hand.
// Source pinned to 1.0 pu of the transformer's rated primary (the original
// model's SWING runs at 80,250 V LN = 1.053 pu); OLTC forced to neutral.
object meter {{
\tname Jesmond_132kV_Bus;
\tphases ABC;
\tnominal_voltage {VLN_132:.5f};
\tvoltage_A {VLN_132:.5f};
\tvoltage_B {vb.real:.5f}{vb.imag:+.5f}j;
\tvoltage_C {vc.real:.5f}{vc.imag:+.5f}j;
\tbustype SWING;
}};
object transformer {{
\tname TXZoneSub;
\tphases ABC;
\tfrom Jesmond_132kV_Bus;
\tto BusZoneSubOLTC;
\tconfiguration conf_TXZoneSub;
}};
object transformer_configuration {{
\tname conf_TXZoneSub;
\tconnect_type DELTA_GWYE;
\tinstall_type VAULT;
\tprimary_voltage 76210.23553 V;
\tsecondary_voltage 6350.85296108588 V;
\tpower_rating 50 MVA;
\timpedance 0.0075+0.3580j;
}};
object meter {{
\tname BusZoneSubOLTC;
\tphases ABC;
\tnominal_voltage 6350.85 V;
\tbustype PQ;
}};
object regulator {{
\tname TXZoneSubOLTC;
\tphases ABC;
\tfrom BusZoneSubOLTC;
\tto BusZoneSub11kV;
\tconfiguration conf_TXZoneSubOLTC;
}};
object regulator_configuration {{
\tname conf_TXZoneSubOLTC;
\tconnect_type WYE_WYE;
\traise_taps 16;
\tlower_taps 10;
\tregulation 0.2;
\tband_center 6350.85;
\tband_width 128;
\tControl MANUAL;
\ttap_pos_A 0;
\ttap_pos_B 0;
\ttap_pos_C 0;
}};
""", encoding="utf-8")


def write_harness(sub_names):
    # GridLAB-D resolves #include relative to the process CWD, so all paths
    # assume the documented invocation from the repo root.
    includes = "\n".join(
        f'#include "validation/stripped/{n}"'
        for n in ["zone_stripped.glm"] + sub_names)
    (OUT / "harness.glm").write_text(f"""\
// Generated by validation/gen_harness.py -- do not edit by hand.
// One-instant snapshot of the Elermore Vale network for cross-validation
// against the OpenDSS translation (MODEL_VERIFICATION.md, Level 4).
// Run from the repo root: gridlabd validation/harness.glm
clock {{
\ttimezone PST+8PDT;
\tstarttime '2012-01-01 00:00:00';
\tstoptime '2012-01-01 00:00:00';
}}
#include "common/ModulePowerflow.glm"
#include "validation/stripped/line_configs_stripped.glm"
#include "Elermorevale/TransformerConfigs.glm"
#include "Elermorevale/elermorevale11kV.glm"
{includes}
object voltdump {{
\tfilename validation/voltages_gld.csv;
\tmode POLAR;
}}
""", encoding="utf-8")


def main():
    # Stage every subs file in memory first: rewrite_load and the residue
    # guards raise SystemExit on a bad input, and aborting mid-write would
    # leave validation/stripped/ half-regenerated next to a stale
    # harness.glm that GridLAB-D would happily solve.
    staged, total_gen, total_loads = [], 0, 0
    for src in sorted((GLM / "subs").glob("*.glm")):
        text = src.read_text(encoding="utf-8", errors="replace")
        text, n_gen, n_loads = strip_network_text(text, src.name)
        staged.append((src.name, text))
        total_gen += n_gen
        total_loads += n_loads
    if total_gen == 0:
        raise SystemExit("no solar/inverter blocks matched anywhere -- "
                         "GEN_BLOCK regex rot? (the model has ~316)")

    STRIPPED.mkdir(parents=True, exist_ok=True)
    write_line_configs_stripped()
    write_zone_stripped()
    for name, text in staged:
        (STRIPPED / name).write_text(text, encoding="utf-8")
    sub_names = [name for name, _ in staged]
    write_harness(sub_names)

    p_va, q_va = constant_power_va()
    print(f"harness written: {len(sub_names)} subs files, {total_gen} "
          f"solar/inverter blocks removed, {total_loads} loads overridden "
          f"to {p_va:.0f}{q_va:+.0f}j VA total each (split across phases)")
    print(f"expected aggregate: {total_loads * LOAD_KW:.0f} kW "
          f"(OpenDSS side has 1810 loads -- reconcile any count difference "
          f"before comparing substation totals)")


if __name__ == "__main__":
    main()
