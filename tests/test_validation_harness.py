"""
Tests for the Level 4 cross-validation harness (validation/) and the
translation contracts it depends on.

Added after a review of the first harness revision found that the
published agreement figures were computed over a silently truncated join
and a mismatched operating point:

  - compare_voltages joined GridLAB-D names through safe_name() while the
    builder creates buses with RAW names, silently excluding 77 of the
    101 underscore-prefixed 11 kV backbone buses
    (test_dss_bus_names_join_to_raw_glm_names);
  - gen_harness gave 3-phase loads 1 kW PER PHASE against OpenDSS's
    1 kW per load (test_three_phase_load_total_power_is_split);
  - the builder passed the L-N base as `kv` for 3-phase loads, which
    OpenDSS reads as L-L (test_multiphase_load_kv_uses_line_to_line_base);
  - the regex text surgery could truncate silently or leave residue
    (test_*_fails_loudly; gen_harness now validates before writing).

Group 1 is pure text (no repo data, no DSS engine); Group 2 builds the
real circuit via the session-scoped conftest fixtures.
"""

import re
import sys

import pytest

from conftest import REPO_ROOT, requires_glm_sources

VALIDATION_DIR = REPO_ROOT / "network" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import gen_harness  # noqa: E402


# ==========================================================
# GROUP 1 -- gen_harness unit tests (synthetic GLM text)
# ==========================================================

def _load_block(phases, name="ldtest"):
    return (
        "object load {\n"
        f"\tname {name};\n"
        f"\tphases {phases};\n"
        "\tnominal_voltage 240;\n"
        "\tbase_power_A 1200;\n"
        "}"
    )


def _phase_powers(text):
    """{phase: complex VA} parsed from the constant_power_<ph> lines."""
    powers = {}
    for m in re.finditer(
            r"constant_power_([ABC])\s+([\d.]+)([+-][\d.]+)j", text):
        powers[m.group(1)] = float(m.group(2)) + 1j * float(m.group(3))
    return powers


def test_single_phase_load_gets_full_power():
    out = gen_harness.LOAD_BLOCK.sub(gen_harness.rewrite_load,
                                     _load_block("AN"))
    powers = _phase_powers(out)
    assert set(powers) == {"A"}
    assert powers["A"].real == pytest.approx(1000.0, abs=0.1)
    assert powers["A"].imag == pytest.approx(328.7, abs=0.1)


def test_three_phase_load_total_power_is_split():
    """The OpenDSS side sets `kW=1.0` per LOAD; the GridLAB-D side must
    total 1 kW across A+B+C, split evenly -- not 1 kW per phase, which
    put the 7 three-phase sites at 3x the OpenDSS operating point."""
    out = gen_harness.LOAD_BLOCK.sub(gen_harness.rewrite_load,
                                     _load_block("ABCN"))
    powers = _phase_powers(out)
    assert set(powers) == {"A", "B", "C"}
    total = sum(powers.values())
    assert total.real == pytest.approx(1000.0, abs=0.2)
    assert total.imag == pytest.approx(328.7, abs=0.2)
    per_phase = [p.real for p in powers.values()]
    assert max(per_phase) - min(per_phase) < 0.2


def test_prior_power_properties_are_removed():
    out = gen_harness.LOAD_BLOCK.sub(gen_harness.rewrite_load,
                                     _load_block("AN"))
    assert "base_power_A" not in out


def test_phaseless_load_fails_loudly():
    with pytest.raises(SystemExit, match="declares no"):
        gen_harness.LOAD_BLOCK.sub(gen_harness.rewrite_load,
                                   _load_block("N"))


def test_generator_blocks_are_stripped():
    text = ("object solar {\n\tname pv1;\n}\n"
            "object inverter {\n\tname inv1;\n};\n"
            + _load_block("AN"))
    out, n_gen, n_loads = gen_harness.strip_network_text(text, "t.glm")
    assert n_gen == 2
    assert n_loads == 1
    assert "solar" not in out
    assert "inverter" not in out


def test_unmatched_generator_block_fails_loudly():
    """GLM inheritance syntax (`object solar:123 {`) does not match
    GEN_BLOCK; the residue check must refuse to write a harness that
    would solve WITH PV against OpenDSS's skip_generators=True."""
    text = "object solar:123 {\n\tname pv1;\n}\n" + _load_block("AN")
    with pytest.raises(SystemExit, match="survived stripping"):
        gen_harness.strip_network_text(text, "t.glm")


def test_truncated_generator_block_fails_loudly():
    """A nested block truncates GEN_BLOCK's flat-brace [^}]* match,
    leaving orphaned text; the brace-balance check must catch it."""
    text = ("object solar {\n\tname pv1;\n"
            "\tobject inner { x 1; };\n};\n" + _load_block("AN"))
    with pytest.raises(SystemExit, match="brace imbalance"):
        gen_harness.strip_network_text(text, "t.glm")


# ==========================================================
# GROUP 2 -- engine invariants the Level 4 join depends on
# ==========================================================

@requires_glm_sources
def test_dss_bus_names_join_to_raw_glm_names(built_circuit, glm_objects):
    """compare_voltages joins per (bus, phase) on the raw lowercased GLM
    name. Pin the contract: every DSS bus is either a raw GLM object name
    or one of the zone-substation buses the builder synthesises. (The
    first harness revision applied safe_name() to one side of the join,
    silently dropping every underscore-prefixed backbone bus.)"""
    _, dss = built_circuit
    glm_names = {p["name"].lower() for _, _, p in glm_objects
                 if p.get("name")}
    synthesised = {"jesmond_132kv_bus", "buszonesuboltc", "buszonesub11kv"}
    stray = [b for b in dss.ActiveCircuit.AllBusNames
             if b.lower() not in glm_names and b.lower() not in synthesised]
    assert not stray, f"DSS buses with no raw-GLM join partner: {stray[:10]}"


@requires_glm_sources
def test_multiphase_load_kv_uses_line_to_line_base(ev, built_circuit,
                                                   glm_objects):
    """OpenDSS reads Load kv as line-to-neutral for 1-phase loads but
    line-to-LINE for 2/3-phase loads; GLM nominal_voltage is always L-N.
    With kv=0.240 a 3-phase load's per-phase base is 138.6 V (~1.8 pu at
    a ~250 V bus), so the constant-P model silently clamps to constant-Z
    overdraw above vmaxpu."""
    _, dss = built_circuit
    ckt = dss.ActiveCircuit
    nominal = {ev.safe_name(p["name"]).lower():
               ev.gfloat(p.get("nominal_voltage", "240"), 240.0)
               for _, otype, p in glm_objects
               if otype == "load" and p.get("name")}
    bad, n_multi = [], 0
    idx = ckt.Loads.First
    while idx:
        nph = ckt.ActiveCktElement.NumPhases
        nom_v = nominal.get(ckt.Loads.Name.lower(), 240.0)
        if nph > 1:
            n_multi += 1
            expected = nom_v * 3 ** 0.5 / 1000.0
        else:
            expected = nom_v / 1000.0
        if abs(ckt.Loads.kV - expected) > 1e-3:
            bad.append((ckt.Loads.Name, nph, ckt.Loads.kV, round(expected, 4)))
        idx = ckt.Loads.Next
    assert n_multi >= 1, "no multi-phase loads found -- test is vacuous"
    assert not bad, f"loads with wrong kv base: {bad[:10]}"
