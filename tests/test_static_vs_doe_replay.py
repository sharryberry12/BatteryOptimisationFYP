"""
Unit tests for studies/static_vs_doe_replay.py (no data files needed).

  * the Ausgrid zone-substation CSV is read in its own layout (one row
    per day, 96 quarter-hour MW columns labelled by interval end) and
    collapsed to the repo's half-hour kW convention;
  * an unpopulated day is refused with the populated span in the message;
  * the headroom envelope tightens below the baseline exactly where the
    substation exceeds its limit and never exceeds the feeder cap;
  * on a synthetic ensemble the DOE regime keeps the aggregate inside
    its envelope (up to reported shortfall), never exceeds the
    no-battery peak, and relieves the substation at least as well as
    the static regime.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

vc = pytest.importorskip("vpp.vpp_common")
base = pytest.importorskip("dispatch.osqp_daily")
sr = pytest.importorskip("studies.static_vs_doe_replay")
from test_vpp_methods import synthetic_household  # noqa: E402

T = vc.T
DT = vc.DT


def _write_jesmond_csv(path, rows, labels=sr.QUARTER_HOUR_LABELS):
    """rows: {ddMONyyyy: list of 96 MW values or None for an empty day}."""
    records = []
    for date, values in rows.items():
        rec = {"year": 2011, "Zone Substation": "Jesmond 132_11kV",
               "Date": date, "unit": "MW"}
        rec.update({lab: (np.nan if values is None else values[i])
                    for i, lab in enumerate(labels)})
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False)


def test_quarter_hour_labels_match_the_ausgrid_header():
    labels = sr.QUARTER_HOUR_LABELS
    assert len(labels) == 96
    assert labels[:4] == ("0:15", "0:30", "0:45", "1:00")
    assert labels[-2:] == ("23:45", "24:00")


def test_load_jesmond_day_averages_quarter_hours_into_kw(tmp_path):
    csv = tmp_path / "jesmond.csv"
    quarter_mw = [0.01 * (i + 1) for i in range(96)]      # 0.01 .. 0.96 MW
    _write_jesmond_csv(csv, {"04FEB2011": None, "05FEB2011": quarter_mw})

    kw = sr.load_jesmond_day(csv, "2011-02-05")

    expected = np.array(quarter_mw).reshape(T, 2).mean(axis=1) * 1000.0
    assert kw.shape == (T,)
    np.testing.assert_allclose(kw, expected)
    assert kw[0] == pytest.approx(15.0)                   # (0.01+0.02)/2 MW


def test_load_jesmond_day_selects_columns_by_label_not_position(tmp_path):
    csv = tmp_path / "jesmond_shuffled.csv"
    quarter_mw = [0.01 * (i + 1) for i in range(96)]
    reversed_labels = list(sr.QUARTER_HOUR_LABELS)[::-1]   # "24:00" first
    _write_jesmond_csv(csv, {"05FEB2011": quarter_mw[::-1]},
                       labels=reversed_labels)

    kw = sr.load_jesmond_day(csv, "2011-02-05")

    expected = np.array(quarter_mw).reshape(T, 2).mean(axis=1) * 1000.0
    np.testing.assert_allclose(kw, expected)


def test_load_jesmond_day_rejects_missing_columns(tmp_path):
    csv = tmp_path / "jesmond_short.csv"
    labels = list(sr.QUARTER_HOUR_LABELS)[:-1]           # drop "24:00"
    _write_jesmond_csv(csv, {"05FEB2011": [1.0] * 95}, labels=labels)

    with pytest.raises(ValueError, match="missing quarter-hour columns"):
        sr.load_jesmond_day(csv, "2011-02-05")


def test_load_jesmond_day_rejects_unpopulated_day(tmp_path):
    csv = tmp_path / "jesmond.csv"
    _write_jesmond_csv(csv, {"04FEB2011": None,
                             "05FEB2011": [1.0] * 96,
                             "06FEB2011": [1.0] * 96})

    with pytest.raises(ValueError, match="96 of 96 readings missing.*"
                                         "2011-02-05 -> 2011-02-06"):
        sr.load_jesmond_day(csv, "2011-02-04")
    with pytest.raises(ValueError, match="not a row"):
        sr.load_jesmond_day(csv, "2011-03-01")


def test_zone_headroom_envelope_tightens_only_in_the_stress_window():
    hrs = np.arange(T) * DT
    p_base = 100.0 + 300.0 * np.exp(-((hrs - 19.0) / 2.0) ** 2)
    jes = 3000.0 + 1200.0 * np.exp(-((hrs - 17.0) / 3.0) ** 2)
    zone_limit = 0.95 * jes.max()
    feeder_cap = p_base.max()

    d_max = sr.zone_headroom_envelope(jes, p_base, zone_limit, feeder_cap)

    stress = jes > zone_limit
    assert stress.any() and not stress.all()
    # inside the window the fleet must shed exactly the substation excess
    np.testing.assert_allclose(d_max[stress],
                               p_base[stress] - (jes[stress] - zone_limit))
    assert np.all(d_max[stress] < p_base[stress])
    # outside it the fleet may draw at least its baseline, up to the cap
    assert np.all(d_max[~stress] >= p_base[~stress] - 1e-9)
    assert np.all(d_max <= feeder_cap + 1e-9)


def test_zone_headroom_envelope_rejects_baseline_above_measurement():
    with pytest.raises(ValueError, match="slice"):
        sr.zone_headroom_envelope(np.full(T, 100.0), np.full(T, 200.0),
                                  90.0, 500.0)


@pytest.fixture(scope="module")
def synthetic_case():
    tariff = base.build_tariff()
    households = [synthetic_household(i, tariff) for i in range(4)]
    p_base = np.sum([hh.net for hh in households], axis=0)
    hrs = np.arange(T) * DT
    # substation stressed across the evening peak, ~10x the ensemble
    jes = 10.0 * p_base.max() * (0.6 + 0.4 * np.exp(-((hrs - 18.5) / 2.5) ** 2))
    zone_limit = 0.97 * jes.max()
    feeder_cap = p_base.max()
    d_max = sr.zone_headroom_envelope(jes, p_base, zone_limit, feeder_cap)
    cases = sr.solve_cases(households, export_limit_kw=1.5, d_max_doe=d_max)
    return households, tariff, jes, p_base, zone_limit, d_max, cases


def test_solve_cases_nobatt_is_the_baseline_and_doe_obeys_its_envelope(
        synthetic_case):
    households, _tariff, _jes, p_base, _zl, d_max, cases = synthetic_case

    np.testing.assert_allclose(
        vc.aggregate_pi(households, cases["nobatt"]["B"]), p_base)
    for name in ("static", "doe"):
        for hh, b in zip(households, cases[name]["B"]):
            assert vc.validate_dispatch(hh, b) == []

    for name in ("static", "doe"):
        agg = vc.aggregate_pi(households, cases[name]["B"])
        res = cases[name]["result"]
        assert np.all(agg <= cases[name]["d_max"] + res.slack_up + 1e-4)
        assert np.all(agg >= cases[name]["d_min"] - res.slack_lo - 1e-4)

    agg_doe = vc.aggregate_pi(households, cases["doe"]["B"])
    shortfall = cases["doe"]["result"].slack_up
    assert agg_doe.max() <= p_base.max() + shortfall.max() + 1e-4


def test_doe_relieves_the_substation_at_least_as_well_as_static(
        synthetic_case):
    households, tariff, jes, p_base, zone_limit, _d_max, cases = synthetic_case

    rows = {name: sr.case_metrics(name, households, cases[name], tariff,
                                  "fit", jes, p_base, zone_limit)
            for name in sr.CASES}

    assert rows["nobatt"]["zone_exceed_kwh"] > 0
    assert rows["doe"]["zone_exceed_kwh"] <= rows["static"]["zone_exceed_kwh"] + 1e-6
    assert rows["doe"]["zone_exceed_kwh"] < rows["nobatt"]["zone_exceed_kwh"]
    for name in sr.CASES:
        assert rows[name]["import_shortfall_kwh"] >= 0.0
        assert rows[name]["export_excess_kwh"] >= 0.0
    assert rows["nobatt"]["savings_per_day"] == pytest.approx(0.0)


def test_solve_cases_two_stage_hook_adds_a_valid_method_b_case(
        synthetic_case):
    households, tariff, jes, p_base, zone_limit, d_max, _cases = synthetic_case

    cases = sr.solve_cases(households, export_limit_kw=1.5, d_max_doe=d_max,
                           two_stage_rules=("maxmin",))

    assert list(cases) == ["nobatt", "static", "doe", "two_stage_maxmin"]
    case = cases["two_stage_maxmin"]
    assert case["B"].shape == (len(households), T)
    assert case["n_failed"] == 0
    for hh, b in zip(households, case["B"]):
        assert vc.validate_dispatch(hh, b, tol=1e-3) == []
    # slices sum to the envelope, so the aggregate is inside it up to the
    # reported per-household shortfall / export excess
    agg = vc.aggregate_pi(households, case["B"])
    assert np.all(agg <= d_max + case["result"].slack_up + 1e-3)
    assert np.all(agg >= case["d_min"] - case["result"].slack_lo - 1e-3)
    row = sr.case_metrics("two_stage_maxmin", households, case, tariff,
                          "fit", jes, p_base, zone_limit)
    assert row["import_shortfall_kwh"] >= 0.0
    assert row["n_failed"] == 0


def test_save_voltages_writes_monitor_ordered_matrix(tmp_path):
    monitored = ["load_b", "load_a", "load_c"]
    result = {"voltages": {m: np.full(T, 0.90 + 0.01 * i)
                           for i, m in enumerate(monitored)}}

    path = sr.save_voltages(tmp_path, "static", result, monitored)

    assert path.name == "voltages_static.npy"
    V = np.load(path)
    assert V.shape == (3, T)
    np.testing.assert_allclose(V[:, 0], [0.90, 0.91, 0.92])
