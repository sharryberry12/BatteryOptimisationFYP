"""
Level 2 -- translation invariants: the real GLM source vs the built OpenDSS
circuit (see MODEL_VERIFICATION.md).

Two groups:
  1. Source invariants -- parse the actual Elermorevale/ + common/ files and
     check the properties the builder silently relies on (flat braces, name
     injectivity, every reference resolvable, physical impedances).
  2. Engine reconciliation -- build the full DSS circuit (sub-second) and
     check element counts, total load power, and connectivity against the
     GLM source, so dropped/merged/orphaned elements cannot pass unnoticed.

The census constants below are ground truth measured from the checked-in
GLM sources (2026-08). If the GLM files are ever regenerated, re-derive
them (see MODEL_VERIFICATION.md) rather than deleting the assertions.
"""

import re
from collections import Counter

import pytest

from conftest import COMMON_DIR, GLM_DIR, requires_glm_sources

pytestmark = requires_glm_sources

# Ground-truth census of the checked-in GLM source (measured, not assumed).
EXPECTED_GLM_CENSUS = {
    "load": 1810,
    "solar": 155,
    "battery": 40,
    "transformer": 24,               # 23 distribution + 1 zone substation
    "transformer_configuration": 24,
    "overhead_line": 1743,
    "underground_line": 422,
    "switch": 246,
    "fuse": 40,
    "inverter": 161,
}
TOTAL_GLM_OBJECTS = 7958
# Of the 1,810 GLM `load` objects, 25 live in generators/Generators2.glm and
# are BlueGen fuel-cell CHP units (not households); 14 of those declare a
# phase their service-point meter does not carry. The builder skips all 25
# (elermorevale_openDSS.is_chp_load) -- see MODEL_VERIFICATION.md defect #6.
EXPECTED_CHP_UNITS = 25
EXPECTED_CHP_PHASE_MISMATCHES = 14

# GLM object types the builder maps into each OpenDSS element namespace.
BRANCH_TYPES = ("overhead_line", "underground_line", "triplex_line",
                "switch", "fuse")
# Batteries share the Generator namespace: Storage elements destabilise the
# engine on this network (MODEL_VERIFICATION.md "Known defects"), so the
# builder models them as dispatchable generators.
DSS_NAMESPACES = {
    "Line": BRANCH_TYPES,
    "Transformer": ("transformer",),
    "Load": ("load",),
    "Generator": ("solar", "battery"),
}


# ==========================================================
# GROUP 1 -- source invariants (parse-only, no DSS engine)
# ==========================================================

def test_flat_brace_assumption_holds(ev):
    """parse_glm's regex drops objects with nested braces; assert the real
    files contain none by comparing against a raw 'object X {' count."""
    mismatches = []
    total_raw = total_parsed = 0
    for fp in sorted(GLM_DIR.rglob("*.glm")):
        if "__MACOSX" in str(fp):
            continue
        text = re.sub(r"//[^\n]*", "", fp.read_text(encoding="utf-8",
                                                    errors="replace"))
        raw = len(re.findall(r"object\s+\w+\s*\{", text))
        parsed = len(ev.parse_glm(str(fp)))
        total_raw += raw
        total_parsed += parsed
        if raw != parsed:
            mismatches.append((fp.name, raw, parsed))
    assert not mismatches, f"parser dropped objects in: {mismatches}"
    assert total_raw == total_parsed == TOTAL_GLM_OBJECTS


def test_glm_object_census(glm_objects):
    """Counts of every object type the builder consumes, pinned to the
    measured ground truth of the checked-in model."""
    counts = Counter(otype for _, otype, _ in glm_objects)
    for otype, expected in EXPECTED_GLM_CENSUS.items():
        assert counts[otype] == expected, (
            f"{otype}: found {counts[otype]}, expected {expected}")
    assert len(glm_objects) == TOTAL_GLM_OBJECTS


def test_every_line_configuration_resolves(glm_objects, line_tables):
    """No line may silently fall back to the guessed fallback impedance:
    every configuration referenced by a line must yield a real linecode."""
    _, _, linecodes = line_tables
    referenced = {p.get("configuration", "")
                  for _, otype, p in glm_objects
                  if otype in ("overhead_line", "underground_line",
                               "triplex_line") and p.get("configuration")}
    unmapped = referenced - set(linecodes)
    assert not unmapped, f"configs with no linecode (fallback used): {unmapped}"


def test_every_conductor_reference_resolves(line_tables):
    """LV configs referencing a conductor missing from the conductor table
    silently get r=0 (a superconducting line); assert none do."""
    conductors, configs, _ = line_tables
    missing = {p["conductor_A"] for p in configs.values()
               if p.get("conductor_A") and p["conductor_A"] not in conductors}
    assert not missing, f"configs reference unknown conductors: {missing}"


def test_every_transformer_configuration_resolves(glm_objects):
    """The builder falls back to a default 300 kVA rating when a transformer's
    configuration is missing (tx_cfgs.get(cfg, {})); assert it never does."""
    tx_cfg_names = {p.get("name", "") for _, otype, p in glm_objects
                    if otype == "transformer_configuration"}
    unresolved = [p.get("name") for _, otype, p in glm_objects
                  if otype == "transformer"
                  and "TXZoneSub" not in p.get("name", "")
                  and p.get("configuration", "") not in tx_cfg_names]
    assert not unresolved, f"transformers with missing config: {unresolved}"


def test_safe_name_injective_per_namespace(ev, glm_objects):
    """Two GLM names sanitising to the same DSS name would silently merge
    elements. Check injectivity within each OpenDSS namespace."""
    for namespace, otypes in DSS_NAMESPACES.items():
        names = [p["name"] for _, otype, p in glm_objects
                 if otype in otypes and p.get("name")]
        sanitised = Counter(ev.safe_name(n) for n in names)
        dupes = [n for n, c in sanitised.items() if c > 1]
        assert not dupes, f"{namespace} name collisions after safe_name: {dupes}"


def test_referenced_impedances_are_physical(glm_objects, line_tables):
    """Every linecode actually used by a line must have 0 < r and 0 <= x in
    a plausible Ohm/km range, a positive rating, and 1 or 3 phases."""
    _, _, linecodes = line_tables
    referenced = {p.get("configuration", "")
                  for _, otype, p in glm_objects
                  if otype in ("overhead_line", "underground_line",
                               "triplex_line") and p.get("configuration")}
    for cfg in referenced:
        r, x, rating, nph = linecodes[cfg]
        assert 0.0 < r < 10.0, f"{cfg}: r={r} Ohm/km implausible"
        assert 0.0 <= x < 1.0, f"{cfg}: x={x} Ohm/km implausible"
        assert rating > 0, f"{cfg}: rating={rating}"
        assert nph in (1, 3), f"{cfg}: nphases={nph}"


def test_all_loads_connect_to_the_network(glm_objects):
    """Follow each load's parent chain (load -> meter -> node) and assert it
    terminates at a bus that some branch element actually connects to --
    the pre-engine version of the isolated-load check."""
    parent_of = {p["name"]: p["parent"] for _, _, p in glm_objects
                 if p.get("name") and p.get("parent")}

    def resolve(name, max_depth=10):
        seen, current = set(), name
        for _ in range(max_depth):
            if current not in parent_of or current in seen:
                return current
            seen.add(current)
            current = parent_of[current]
        return current

    endpoints = set()
    for _, otype, p in glm_objects:
        if otype in BRANCH_TYPES + ("transformer", "regulator"):
            endpoints.add(p.get("from", ""))
            endpoints.add(p.get("to", ""))

    orphans = [p["name"] for _, otype, p in glm_objects
               if otype == "load"
               and resolve(p.get("parent", "")) not in endpoints]
    assert not orphans, f"loads not reachable from any branch: {orphans[:10]}"


def test_residential_load_phases_are_subset_of_service_point(ev, glm_objects):
    """A load can only be energised on phases its parent (meter/node)
    actually carries -- GridLAB-D's own child-phase rule. The builder trusts
    each load's declared `phases`, so a mismatch produces a floating 0 V
    node-phase that Topology.NumIsolatedLoads cannot see. Every residential
    load must satisfy the rule; the BlueGen CHP `load`s in Generators2.glm
    are the measured exception (14 of 25) and are why the builder skips them."""
    phases_of = {p["name"]: p.get("phases", "") for _, _, p in glm_objects
                 if p.get("name")}
    letters = lambda s: {c for c in s if c in "ABC"}
    bad_res, n_chp, bad_chp = [], 0, 0
    for src, otype, p in glm_objects:
        if otype != "load":
            continue
        parent = p.get("parent", "")
        if parent not in phases_of:
            continue                      # orphan check lives in another test
        subset = letters(p.get("phases", "")) <= letters(phases_of[parent])
        if ev.is_chp_load(src):
            n_chp += 1
            bad_chp += (not subset)
        elif not subset:
            bad_res.append((p["name"], p.get("phases"), phases_of[parent]))
    assert not bad_res, f"residential loads off their parent's phases: {bad_res[:10]}"
    assert n_chp == EXPECTED_CHP_UNITS
    assert bad_chp == EXPECTED_CHP_PHASE_MISMATCHES


# ==========================================================
# GROUP 2 -- engine reconciliation (builds the DSS circuit)
# ==========================================================

def test_builder_stats_match_glm_census(built_circuit):
    stats, _ = built_circuit
    # every GLM load is either a built residential Load or a skipped CHP unit
    assert stats["chp_units_skipped"] == EXPECTED_CHP_UNITS
    assert (stats["loads"] + stats["chp_units_skipped"]
            == EXPECTED_GLM_CENSUS["load"])
    assert stats["pv_systems"] == EXPECTED_GLM_CENSUS["solar"]
    assert stats["batteries"] == EXPECTED_GLM_CENSUS["battery"]
    # zone substation is excluded from the distribution-transformer count
    assert stats["distribution_transformers"] == (
        EXPECTED_GLM_CENSUS["transformer"] - 1)
    assert stats["lines"] == (EXPECTED_GLM_CENSUS["overhead_line"]
                              + EXPECTED_GLM_CENSUS["underground_line"])
    assert stats["switches_and_fuses"] == (EXPECTED_GLM_CENSUS["switch"]
                                           + EXPECTED_GLM_CENSUS["fuse"])
    assert stats["unmapped_line_configs"] == 0


def test_engine_element_counts_match_source(built_circuit):
    """The DSS engine must hold exactly one element per GLM object -- a
    collision or a rejected New command would show up here."""
    stats, dss = built_circuit
    ckt = dss.ActiveCircuit
    assert ckt.Lines.Count == stats["lines"] + stats["switches_and_fuses"]
    assert ckt.Loads.Count == stats["loads"]
    # +2: the manually built zone-sub transformer and OLTC autotransformer
    assert ckt.Transformers.Count == stats["distribution_transformers"] + 2
    # batteries are modelled as generators (see DSS_NAMESPACES note)
    assert ckt.Generators.Count == stats["pv_systems"] + stats["batteries"]
    n_storage = sum(1 for name in ckt.AllElementNames
                    if name.lower().startswith("storage."))
    assert n_storage == 0


def test_engine_total_load_power(built_circuit):
    """Every load is created at the 3.0 kW default, so total connected load
    must equal 3 * n_loads exactly -- catches dropped or merged loads even
    if counts were somehow fudged."""
    stats, dss = built_circuit
    loads = dss.ActiveCircuit.Loads
    total_kw, idx = 0.0, loads.First
    while idx:
        total_kw += loads.kW
        idx = loads.Next
    assert total_kw == pytest.approx(3.0 * stats["loads"], abs=1e-6)


@pytest.mark.parametrize("skip_generators, oltc, expected", [
    (False, False, 1),      # full model: regulator is part of the network
    (True, False, 0),       # profile mode: skipped with the generators
    (True, True, 1),        # profile mode with --oltc
])
def test_zone_regcontrol_built_only_when_requested(ev, skip_generators,
                                                   oltc, expected):
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR),
                          skip_generators=skip_generators, oltc=oltc)
    assert ev.dss.ActiveCircuit.RegControls.Count == expected


def test_zone_regcontrol_settings_target_one_pu(ev):
    """The RegControl must regulate the 11 kV winding to 1.0 pu on a 120 V
    base: PT ratio = (11 kV / sqrt3) / 120 = 52.92, vreg = 120, band = 2.4 V
    (+-1 %, mirroring the GLM band_width 128 V / 52.92). The pre-2026-08-14
    values (vreg=110, ptratio=100) implied an 11,000 V L-N target and were
    only harmless because every solve ran controlmode=off. Tap range mirrors
    the GLM regulator_configuration (raise 16 / lower 10 at 1.25 %)."""
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR),
                          skip_generators=True, oltc=True)

    def q(prop):
        ev.dss.Text.Command = f"? {prop}"
        return float(ev.dss.Text.Result)

    assert q("RegControl.OLTC_ctrl.vreg") == pytest.approx(120.0)
    assert q("RegControl.OLTC_ctrl.band") == pytest.approx(2.4)
    assert q("RegControl.OLTC_ctrl.ptratio") == pytest.approx(
        11000.0 / 3 ** 0.5 / 120.0, rel=1e-3)
    assert q("Transformer.OLTC.maxtap") == pytest.approx(1.0 + 16 * 0.0125)
    assert q("Transformer.OLTC.mintap") == pytest.approx(1.0 - 10 * 0.0125)
    assert q("Transformer.OLTC.numtaps") == 26


def test_engine_no_isolated_elements(built_circuit):
    """OpenDSS's own topology processor must find every branch and load
    connected to the energised source -- the definitive islanding check."""
    _, dss = built_circuit
    topology = dss.ActiveCircuit.Topology
    assert topology.NumIsolatedBranches == 0
    assert topology.NumIsolatedLoads == 0
