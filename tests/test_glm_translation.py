"""
Level 1 -- unit tests for the pure GLM->OpenDSS translation functions in
elermorevale_openDSS.py (see MODEL_VERIFICATION.md).

Every test here uses synthetic inputs only: no repo data files, no OpenDSS
engine state. The two highest-value targets are the Ohm/mile -> Ohm/km
conversion (a silent 61% impedance error if wrong) and the GLM parser's
flat-brace assumption (silently dropped objects if violated).
"""

import pytest

MI_TO_KM = 1.60934


# ==========================================================
# gfloat -- GLM numeric parsing with unit suffixes
# ==========================================================

@pytest.mark.parametrize("raw, expected", [
    ("11.59 m^2", 11.59),        # unit suffix stripped
    ("240", 240.0),
    ("-0.5", -0.5),
    ("+3.2", 3.2),
    (".5", 0.5),
    ("1.5e-3 Ohm/mile", 0.0015),  # scientific notation with unit
    (42, 42.0),                   # non-string input
])
def test_gfloat_parses(ev, raw, expected):
    assert ev.gfloat(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "garbage", "m^2"])
def test_gfloat_falls_back_to_default(ev, raw):
    assert ev.gfloat(raw, default=7.5) == 7.5


# ==========================================================
# glm_length_m -- line lengths with GridLAB-D unit semantics
# ==========================================================

@pytest.mark.parametrize("raw, metres", [
    ("68.9 m", 68.9),            # explicit metres (all LV lengths)
    ("754.59", 754.59 * 0.3048),  # bare = FEET, GridLAB-D's default
    ("1.2 km", 1200.0),
    ("100 ft", 30.48),
    ("100 ft;", 30.48),           # tolerate trailing semicolon
])
def test_glm_length_m(ev, raw, metres):
    assert ev.glm_length_m(raw) == pytest.approx(metres, rel=1e-6)


def test_glm_length_m_default(ev):
    assert ev.glm_length_m(None, default=2.5) == 2.5
    assert ev.glm_length_m("", default=2.5) == 2.5


def test_glm_length_m_numeric_zero_is_zero(ev):
    # 0 is a value, not an absence -- must not fall through to the default
    assert ev.glm_length_m(0, default=2.5) == 0.0


# ==========================================================
# glm_phases_to_dss -- phase notation mapping
# ==========================================================

@pytest.mark.parametrize("glm, expected", [
    ("AN", (".1", 1)),
    ("BN", (".2", 1)),
    ("CN", (".3", 1)),
    ("ABCN", (".1.2.3", 3)),
    ("ABC", (".1.2.3", 3)),
    ("ABCD", (".1.2.3", 3)),      # delta marker stripped
    ("AS", (".1", 1)),            # triplex 'S' ignored, phase kept
])
def test_phase_mapping(ev, glm, expected):
    assert ev.glm_phases_to_dss(glm) == expected


@pytest.mark.parametrize("glm", ["", "N", "D"])
def test_phase_mapping_falls_back_to_three_phase(ev, glm):
    assert ev.glm_phases_to_dss(glm) == (".1.2.3", 3)


# ==========================================================
# safe_name -- GLM name -> valid OpenDSS element name
# ==========================================================

@pytest.mark.parametrize("raw, expected", [
    ("Bus_1", "Bus_1"),           # already safe: unchanged
    ("TX-33/1", "TX_33_1"),       # punctuation replaced
    ("a b.c", "a_b_c"),
    ("7eleven", "E7eleven"),      # leading digit prefixed
    ("_hidden", "E_hidden"),      # leading underscore prefixed
])
def test_safe_name(ev, raw, expected):
    assert ev.safe_name(raw) == expected


def test_safe_name_idempotent(ev):
    for raw in ("Bus_1", "TX-33/1", "7eleven", "_hidden"):
        once = ev.safe_name(raw)
        assert ev.safe_name(once) == once


# ==========================================================
# parse_glm -- flat GLM object parser
# ==========================================================

GLM_SAMPLE = """
// file header comment
object overhead_line {
    name OH_1;            // inline comment
    from busA;
    to busB;
    length 120.5 m;
    configuration cfg_1;
}
object load {
    name L1;
    parent meterX;
    nominal_voltage 240;
    rating.summer.continuous 355.0;
}
object switch { }
"""


def test_parse_glm_objects_and_props(ev, tmp_path):
    fp = tmp_path / "sample.glm"
    fp.write_text(GLM_SAMPLE, encoding="utf-8")
    objs = ev.parse_glm(str(fp))

    assert [otype for otype, _ in objs] == ["overhead_line", "load", "switch"]

    oh = objs[0][1]
    assert oh["name"] == "OH_1"
    assert oh["from"] == "busA"
    assert oh["length"] == "120.5 m"          # raw value; gfloat strips later

    load = objs[1][1]
    assert load["nominal_voltage"] == "240"
    assert load["rating.summer.continuous"] == "355.0"  # dotted keys parse

    assert objs[2][1] == {}                   # empty object -> empty props


def test_parse_glm_strips_comments(ev, tmp_path):
    fp = tmp_path / "comments.glm"
    fp.write_text(
        "// object fake { name NOPE; }\n"
        "object load { name L1; } // trailing\n",
        encoding="utf-8")
    objs = ev.parse_glm(str(fp))
    assert len(objs) == 1
    assert objs[0][1]["name"] == "L1"


# ==========================================================
# parse_line_configs -- conductor + configuration tables
# ==========================================================

LINE_CONFIGS_SAMPLE = """
object overhead_line_conductor {
    name cond_OH_7/.064CU;
    conductor_resistance 1.503;
    rating.summer.continuous 149.0;
}
object underground_line_conductor {
    name cond_UG_25mm;
    resistance 0.884;
    rating.summer.continuous 110.0;
}
object line_configuration {
    name conf_OHLine_1ph;
    conductor_A cond_OH_7/.064CU;
}
object line_configuration {
    name Elermore_line_config_1;
    z11 0.306+0.627j;
    z12 0.101+0.209j;
    z13 0.101+0.209j;
    z21 0.101+0.209j;
    z22 0.306+0.627j;
    z23 0.101+0.209j;
    z31 0.101+0.209j;
    z32 0.101+0.209j;
    z33 0.306+0.627j;
}
"""


@pytest.fixture()
def parsed_line_tables(ev, tmp_path):
    (tmp_path / "Line Configs.glm").write_text(
        LINE_CONFIGS_SAMPLE, encoding="utf-8")
    return ev.parse_line_configs(str(tmp_path))


def test_parse_line_configs_conductors(parsed_line_tables):
    conductors, configs, conductors_full, spacings = parsed_line_tables
    assert conductors["cond_OH_7/.064CU"] == (1.503, 149.0)
    # underground conductors use the 'resistance' property alias
    assert conductors["cond_UG_25mm"] == (0.884, 110.0)
    assert set(configs) == {"conf_OHLine_1ph", "Elermore_line_config_1"}
    # full props keep the object type for the Carson OH/UG dispatch
    assert conductors_full["cond_OH_7/.064CU"]["__type"] ==         "overhead_line_conductor"
    assert conductors_full["cond_UG_25mm"]["__type"] ==         "underground_line_conductor"
    assert spacings == {}


# ==========================================================
# extract_impedances -- linecode spec table
# ==========================================================

def test_zmatrix_sequence_ohm_per_mile_to_km(ev):
    """z1 = z11 - z12, z0 = z11 + 2 z12, Ohm/mile -> Ohm/km. Keeping only
    z11 (the pre-2026-08 reduction) overstated the balanced impedance ~2x
    wherever the source matrices carry mutual terms."""
    configs = {"E1": {"z11": "0.306+0.627j", "z12": "0.101+0.209j",
                      "z13": "0.101+0.209j", "z21": "0.101+0.209j",
                      "z22": "0.306+0.627j", "z23": "0.101+0.209j",
                      "z31": "0.101+0.209j", "z32": "0.101+0.209j",
                      "z33": "0.306+0.627j"}}
    spec = ev.extract_impedances({}, configs)["E1"]
    assert spec["kind"] == "seq"
    assert spec["r1"] == pytest.approx((0.306 - 0.101) / MI_TO_KM, abs=1e-5)
    assert spec["x1"] == pytest.approx((0.627 - 0.209) / MI_TO_KM, abs=1e-5)
    assert spec["r0"] == pytest.approx((0.306 + 2 * 0.101) / MI_TO_KM,
                                       abs=1e-5)
    assert spec["x0"] == pytest.approx((0.627 + 2 * 0.209) / MI_TO_KM,
                                       abs=1e-5)
    assert spec["nph"] == 3


def test_zmatrix_implied_one_phase_diagonal_is_z11(ev):
    """OpenDSS rebuilds a 1-phase line from a seq code as (z0 + 2 z1)/3 --
    with the balanced reduction that is exactly z11, matching GridLAB-D's
    declared-phase submatrix behaviour."""
    configs = {"E1": {"z11": "0.306+0.627j", "z12": "0.101+0.209j",
                      "z13": "0.101+0.209j", "z21": "0.101+0.209j",
                      "z22": "0.306+0.627j", "z23": "0.101+0.209j",
                      "z31": "0.101+0.209j", "z32": "0.101+0.209j",
                      "z33": "0.306+0.627j"}}
    spec = ev.extract_impedances({}, configs)["E1"]
    zs = complex(spec["r0"] + 2 * spec["r1"], spec["x0"] + 2 * spec["x1"]) / 3
    assert zs.real == pytest.approx(0.306 / MI_TO_KM, abs=1e-5)
    assert zs.imag == pytest.approx(0.627 / MI_TO_KM, abs=1e-5)


@pytest.mark.parametrize("z11, expected_rating", [
    ("0.001+0j", 1000),    # r ~ 0.0006 Ohm/km -> busbar/jumper tier
    ("0.16+0.1j", 400),    # r ~ 0.099        -> heavy conductor tier
    ("0.30+0.2j", 300),    # r ~ 0.186        -> medium tier
    ("0.60+0.3j", 250),    # r ~ 0.373        -> light tier
])
def test_zmatrix_rating_tiers(ev, z11, expected_rating):
    result = ev.extract_impedances({}, {"cfg": {"z11": z11}})
    assert result["cfg"]["amps"] == expected_rating


def test_zmatrix_malformed_z11_degrades_to_zero(ev):
    spec = ev.extract_impedances(
        {}, {"cfg": {"z11": "not-a-number"}})["cfg"]
    assert (spec["r1"], spec["x1"]) == (0.0, 0.0)


def test_conductor_reference_becomes_carson_spec(ev):
    """Conductor-reference configs defer to the GridLAB-D-faithful Carson
    computation (per line phase set, at line-emission time); the spec
    carries the config, the OH/UG dispatch and the rating."""
    conductors = {"c_oh": (0.5, 150.0)}
    full = {"c_oh": {"geometric_mean_radius": "0.03", "resistance": "0.5",
                     "__type": "overhead_line_conductor"}}
    configs = {"conf_OHLine_x": {"conductor_A": "c_oh",
                                 "spacing": "sp"}}
    spec = ev.extract_impedances(conductors, configs, full, {})["conf_OHLine_x"]
    assert spec["kind"] == "carson"
    assert spec["is_ug"] is False
    assert spec["amps"] == 150.0
    assert spec["nph"] == 1
    assert spec["config"] is configs["conf_OHLine_x"]


def test_conductor_reference_ug_dispatch_from_object_type(ev):
    """OH/UG is decided by the conductor object's type when available,
    not the config name."""
    conductors = {"c": (0.8, 110.0)}
    full = {"c": {"conductor_gmr": "0.02", "conductor_resistance": "0.8",
                  "__type": "underground_line_conductor"}}
    configs = {"conf_weird_name": {"conductor_A": "c", "conductor_B": "c"}}
    spec = ev.extract_impedances(conductors, configs, full, {})["conf_weird_name"]
    assert spec["is_ug"] is True
    assert spec["nph"] == 3


def test_config_without_impedance_source_is_omitted(ev):
    configs = {"spacer_only": {"spacing": "some_spacing"}}
    assert ev.extract_impedances({}, configs) == {}


# ==========================================================
# is_chp_load -- BlueGen CHP units masquerading as GLM loads
# ==========================================================

@pytest.mark.parametrize("src, expected", [
    ("Generators2.glm", True),
    ("generators2.glm", True),
    (r"Elermorevale\generators\Generators2.glm", True),   # full path
    ("Generators.glm", True),
    ("HP00016304GTX00000001.glm", False),               # a subs/ file
    ("elermorevale11kV.glm", False),
])
def test_is_chp_load_by_source_file(ev, src, expected):
    assert ev.is_chp_load(src) is expected


# ==========================================================
# parse_profile_dates -- both writer formats, no dateutil fallback
# ==========================================================

@pytest.mark.parametrize("raw", [
    ["1-Jul-10", "7-Jan-11"],                       # osqp_daily*.save_profiles
    ["2010-07-01", "2011-01-07"],                   # vpp_export (ISO)
    ["2010-07-01 00:00:00", "2011-01-07 00:00:00"], # re-serialised Timestamps
])
def test_parse_profile_dates_known_formats(ev, raw):
    import pandas as pd
    out = ev.parse_profile_dates(pd.Series(raw))
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2010-07-01", "2011-01-07"]


def test_parse_profile_dates_rejects_unknown_format(ev):
    import pandas as pd
    with pytest.raises(ValueError, match="match none of"):
        ev.parse_profile_dates(pd.Series(["07/01/2010", "01/07/2011"]))


# ==========================================================
# assert_monitors_energised -- dead-monitor guard for daily runs
# ==========================================================

def test_assert_monitors_energised_accepts_live_monitors(ev):
    import numpy as np
    ev.assert_monitors_energised({"a": np.full(ev.T, 0.98),
                                  "b": np.full(ev.T, 1.05)})


def test_assert_monitors_energised_rejects_dead_monitor(ev):
    import numpy as np
    with pytest.raises(RuntimeError, match=r"1 of 2 .*dead_load \(48/48"):
        ev.assert_monitors_energised({"live": np.full(ev.T, 0.98),
                                      "dead_load": np.zeros(ev.T)})


def test_assert_monitors_energised_rejects_truncated_day(ev):
    """A daily solve that aborts part-way leaves zero-padded monitor
    buffers -- a single 0 V sample must trip the guard too."""
    import numpy as np
    v = np.full(ev.T, 0.99)
    v[-1] = 0.0
    with pytest.raises(RuntimeError, match=r"partial \(1/48"):
        ev.assert_monitors_energised({"partial": v})
