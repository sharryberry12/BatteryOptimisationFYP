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
    z22 0.306+0.627j;
    z33 0.306+0.627j;
}
"""


@pytest.fixture()
def parsed_line_tables(ev, tmp_path):
    (tmp_path / "Line Configs.glm").write_text(
        LINE_CONFIGS_SAMPLE, encoding="utf-8")
    return ev.parse_line_configs(str(tmp_path))


def test_parse_line_configs_conductors(parsed_line_tables):
    conductors, configs = parsed_line_tables
    assert conductors["cond_OH_7/.064CU"] == (1.503, 149.0)
    # underground conductors use the 'resistance' property alias
    assert conductors["cond_UG_25mm"] == (0.884, 110.0)
    assert set(configs) == {"conf_OHLine_1ph", "Elermore_line_config_1"}


# ==========================================================
# extract_impedances -- the unit-conversion core
# ==========================================================

def test_zmatrix_ohm_per_mile_to_km(ev):
    configs = {"E1": {"z11": "0.306+0.627j"}}
    r, x, rating, nph = ev.extract_impedances({}, configs)["E1"]
    assert r == pytest.approx(0.306 / MI_TO_KM, abs=1e-5)
    assert x == pytest.approx(0.627 / MI_TO_KM, abs=1e-5)
    assert nph == 3


@pytest.mark.parametrize("z11, expected_rating", [
    ("0.001+0j", 1000),    # r ~ 0.0006 Ohm/km -> busbar/jumper tier
    ("0.16+0.1j", 400),    # r ~ 0.099        -> heavy conductor tier
    ("0.30+0.2j", 300),    # r ~ 0.186        -> medium tier
    ("0.60+0.3j", 250),    # r ~ 0.373        -> light tier
])
def test_zmatrix_rating_tiers(ev, z11, expected_rating):
    result = ev.extract_impedances({}, {"cfg": {"z11": z11}})
    assert result["cfg"][2] == expected_rating


def test_zmatrix_malformed_z11_degrades_to_zero(ev):
    r, x, rating, nph = ev.extract_impedances(
        {}, {"cfg": {"z11": "not-a-number"}})["cfg"]
    assert (r, x) == (0.0, 0.0)


def test_conductor_reference_single_phase_overhead(ev):
    conductors = {"c_oh": (0.5, 150.0)}
    configs = {"conf_OHLine_x": {"conductor_A": "c_oh"}}
    r, x, rating, nph = ev.extract_impedances(conductors, configs)["conf_OHLine_x"]
    assert r == pytest.approx(0.5 / MI_TO_KM, abs=1e-4)
    assert x == 0.25              # estimated overhead reactance (documented)
    assert rating == 150.0
    assert nph == 1


def test_conductor_reference_underground_and_three_phase(ev):
    conductors = {"c_ug": (0.8, 110.0)}
    configs = {"conf_UGLine_x": {"conductor_A": "c_ug", "conductor_B": "c_ug"}}
    r, x, rating, nph = ev.extract_impedances(conductors, configs)["conf_UGLine_x"]
    assert x == 0.08              # estimated underground reactance (documented)
    assert nph == 3


def test_config_without_impedance_source_is_omitted(ev):
    configs = {"spacer_only": {"spacing": "some_spacing"}}
    assert ev.extract_impedances({}, configs) == {}
