"""
Pure helpers of studies/network_doe_study.py on a synthetic ensemble --
no GLM sources, no data.csv, no power flow.

  * realised_flows: PV above the export cap is curtailed (never more than
    the PV available), import above the cap is reported as shortfall and
    left in the flow;
  * dispatch_loads: with caps that cannot bind every load reproduces its
    household's envelope-free dispatch, and a cap the battery cannot meet
    is a reported shortfall, never a failed solve;
  * load_profiles: one profile per network load, carrying the curtailed
    PV and the realised grid flow;
  * load_savings: curtailed energy costs its feed-in credit;
  * summary_row: per-load results scale back to the 152-household
    ensemble the other regimes are reported at.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

vc = pytest.importorskip("vpp.vpp_common")
base = pytest.importorskip("dispatch.osqp_daily")
nd = pytest.importorskip("vpp.network_doe")
study = pytest.importorskip("studies.network_doe_study")
from test_vpp_methods import synthetic_household  # noqa: E402

T = vc.T
DT = vc.DT


@pytest.fixture(scope="module")
def ensemble():
    tariff = base.build_tariff()
    return [synthetic_household(i, tariff) for i in range(3)], tariff


def targets_for(households, owner):
    net = np.vstack([households[i].net for i in owner])
    hi, lo = nd.envelope_targets(net, vc.P_MAX)
    return net, hi, lo


def test_realised_flows_curtail_pv_and_report_shortfall():
    net = np.array([[-4.0, 3.0, -4.0]])
    pv = np.array([[5.0, 0.0, 1.0]])
    B = np.zeros((1, 3))
    hi = np.array([[9.0, 2.0, 9.0]])
    lo = np.array([[-1.5, -5.0, -1.5]])

    grid, curtail, shortfall = study.realised_flows(net, pv, B, hi, lo)

    np.testing.assert_allclose(curtail, [[2.5, 0.0, 1.0]])   # capped by PV
    np.testing.assert_allclose(shortfall, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(grid, [[-1.5, 3.0, -3.0]])


def test_realised_flows_follow_the_battery():
    net = np.array([[-4.0, 3.0]])
    pv = np.array([[5.0, 0.0]])
    B = np.array([[-2.0, 1.5]])                  # charge 2 kW, discharge 1.5
    hi = np.array([[9.0, 1.0]])
    lo = np.array([[-1.5, -5.0]])

    grid, curtail, shortfall = study.realised_flows(net, pv, B, hi, lo)

    # charging absorbs 2 of the 4 kW surplus: 0.5 kW left to curtail
    np.testing.assert_allclose(curtail, [[0.5, 0.0]])
    np.testing.assert_allclose(shortfall, [[0.0, 0.5]])
    np.testing.assert_allclose(grid, [[-1.5, 1.5]])


def test_dispatch_with_non_binding_caps_equals_envelope_free_solve(ensemble):
    households, _ = ensemble
    owner = np.array([0, 1, 2, 0])
    _, hi, lo = targets_for(households, owner)

    B, n_failed = study.dispatch_loads(households, owner, hi, lo)

    assert n_failed == 0
    for j, i in enumerate(owner):
        free, status = vc.HouseholdSolver(households[i]).solve()
        assert "solved" in status
        np.testing.assert_allclose(B[j], free, atol=2e-3)


def test_cap_the_battery_cannot_meet_is_shortfall_not_failure(ensemble):
    households, _ = ensemble
    owner = np.array([0, 1])
    net, hi, lo = targets_for(households, owner)
    hi = np.zeros_like(hi)                       # zero import all day

    B, n_failed = study.dispatch_loads(households, owner, hi, lo)
    pv = np.vstack([households[i].pv for i in owner])
    grid, _, shortfall = study.realised_flows(net, pv, B, hi, lo)

    assert n_failed == 0
    assert shortfall.sum() > 0.0, "a 10 kWh battery cannot carry a full day"
    # a real dispatch, not the zero fallback of a failed solve
    assert np.abs(B).max() > 0.1
    np.testing.assert_allclose(grid, net - B)    # nothing to curtail


def test_load_profiles_carry_curtailed_pv_and_realised_grid(ensemble):
    households, _ = ensemble
    owner = np.array([0, 1])
    loads = ("load_a", "load_b")
    B = np.zeros((2, T))
    curtail = np.zeros((2, T))
    curtail[1, 24] = 0.7

    lc_map, profiles = study.load_profiles(households, owner, loads, B,
                                           curtail, "2011-02-05")

    assert lc_map == {"load_a": "load_a", "load_b": "load_b"}
    day = profiles["load_b"][0]
    hh = households[1]
    np.testing.assert_allclose(day["pv"], hh.pv - curtail[1])
    np.testing.assert_allclose(day["grid"], hh.load - hh.pv + curtail[1])
    assert day["date"] == "2011-02-05"


def test_curtailed_energy_costs_its_feed_in_credit(ensemble):
    households, tariff = ensemble
    owner = np.array([0])
    B = np.zeros((1, T))
    curtail = np.zeros((1, T))
    curtail[0, 24:28] = 0.5                      # 1 kWh over two hours

    kept = study.load_savings(households, owner, B, np.zeros((1, T)),
                              tariff, "fit")
    lost = study.load_savings(households, owner, B, curtail, tariff, "fit")

    assert kept[0] - lost[0] == pytest.approx(
        curtail.sum() * DT * base.FIT_RATE)


def test_summary_row_zone_flow_and_totals():
    agg = np.linspace(100.0, 200.0, T)
    jes = np.full(T, 1000.0)

    row = study.summary_row("a", "A", n_customers=10, agg_kw=agg,
                            base_kw=agg - 20.0, jes_kw=jes,
                            zone_limit_kw=950.0, savings=np.full(10, 2.0),
                            shortfall_kwh=4.0, excess_kwh=6.0,
                            curtailed_kwh=5.0, n_failed=0)

    assert row["savings_per_day"] == pytest.approx(20.0)
    assert row["jain_savings"] == pytest.approx(1.0)
    assert row["peak_kw"] == pytest.approx(200.0)
    assert row["peak_hour"] == pytest.approx((T - 1) * DT)
    # the substation sees the measurement plus the fleet's 20 kW change
    assert row["zone_peak_kw"] == pytest.approx(1020.0)
    assert row["zone_exceed_kwh"] == pytest.approx(70.0 * T * DT)
    assert row["curtailed_kwh"] == pytest.approx(5.0)


def synthetic_day(n_hh=2):
    hours = np.arange(T) * DT
    jes = 900.0 + 200.0 * np.exp(-((hours - 19.0) / 2.0) ** 2)
    return study.DayInputs(households=[None] * n_hh, date_iso="2011-02-05",
                           tariff=np.zeros(T), jes_kw=jes,
                           p_base=np.full(T, 10.0), zone_limit_kw=1000.0,
                           d_max=np.full(T, 12.0))


def test_profiles_frame_sums_caps_per_household_and_keeps_every_regime():
    owner = np.array([0, 1, 0])
    points = study.LoadPoints(loads=("a", "b", "c"), owner=owner, lc_map={},
                              monitored=(), scale=1.5,
                              weight=study.household_weights(owner))
    envelope = dict(hi=np.ones((3, T)), lo=-2.0 * np.ones((3, T)))
    aggs = {"nobatt": np.full(T, 10.0), study.NETWORK_DOE: np.full(T, 8.0)}
    results = {name: {"tx_p_kw": np.full(T, 100.0 + i)}
               for i, name in enumerate(aggs)}

    frame = study.profiles_frame(synthetic_day(), points, aggs, envelope,
                                 results)

    assert len(frame) == T and frame["hour"].iloc[-1] == (T - 1) * DT
    np.testing.assert_allclose(frame["import_cap_sum_kw"], 2.0)   # 2 hh
    np.testing.assert_allclose(frame["export_cap_sum_kw"], -4.0)
    np.testing.assert_allclose(frame["p_network_doe_kw"], 8.0)
    np.testing.assert_allclose(frame["tx_nobatt_kw"], 100.0)
    assert "tx_network_doe_kw" in frame and "d_max_zone_kw" in frame


def test_figures_are_written(tmp_path):
    owner = np.array([0, 1, 0])
    points = study.LoadPoints(loads=("a", "b", "c"), owner=owner, lc_map={},
                              monitored=(), scale=1.5,
                              weight=study.household_weights(owner))
    rng = np.random.default_rng(0)
    net = rng.normal(0.5, 1.0, (3, T))
    envelope = dict(hi=np.abs(net) + 1.0, lo=-np.abs(net) - 1.0)
    aggs = {"nobatt": np.full(T, 10.0), "static": np.full(T, 12.0),
            "doe": np.full(T, 9.0), study.NETWORK_DOE: np.full(T, 8.0)}
    results = {name: {"tx_p_kw": np.full(T, 100.0)} for name in aggs}
    day = synthetic_day()
    frame = study.profiles_frame(day, points, aggs, envelope, results)

    study.make_figures(tmp_path, frame, day, points, envelope, net)

    for name in ("feeder_profile", "envelope_profile",
                 "zone_transformer_power"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0


def test_refigure_redraws_profile_figures_from_a_run_folder(tmp_path):
    import json
    owner = np.array([0, 1, 0])
    points = study.LoadPoints(loads=("a", "b", "c"), owner=owner, lc_map={},
                              monitored=(), scale=1.5,
                              weight=study.household_weights(owner))
    envelope = dict(hi=np.ones((3, T)), lo=-np.ones((3, T)))
    aggs = {"nobatt": np.full(T, 10.0), "static": np.full(T, 12.0),
            "doe": np.full(T, 9.0), study.NETWORK_DOE_IMPORT: np.full(T, 8.0)}
    results = {name: {"tx_p_kw": np.full(T, 100.0)} for name in aggs}
    day = synthetic_day()
    study.profiles_frame(day, points, aggs, envelope, results).to_csv(
        tmp_path / "profiles.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "ensemble": {"date": day.date_iso, "n_households": 2},
        "jesmond": {"zone_limit_kw": day.zone_limit_kw},
        "network": {"n_loads": 3, "scale": 1.5}}), encoding="utf-8")

    study.refigure(tmp_path)

    for name in ("feeder_profile", "zone_transformer_power"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
    assert not (tmp_path / "figures" / "envelope_profile.png").exists()


def test_household_weights_count_every_household_once():
    owner = np.array([0, 1, 0, 1, 0])            # household 0 has 3 copies

    w = study.household_weights(owner)

    np.testing.assert_allclose(w, [1 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 3])
    np.testing.assert_allclose(np.bincount(owner, weights=w), [1.0, 1.0])


def test_load_case_row_is_at_ensemble_scale_with_unequal_copies():
    owner = np.array([0, 1, 0])                  # 2 copies of 0, 1 of 1
    points = study.LoadPoints(loads=("a", "b", "c"), owner=owner, lc_map={},
                              monitored=(), scale=1.5,
                              weight=study.household_weights(owner))
    ones = np.ones((3, T))
    case = dict(grid=ones * [[2.0], [4.0], [6.0]], net=ones * 3.0,
                savings=np.array([1.0, 5.0, 3.0]),
                shortfall=ones * [[0.0], [0.5], [0.0]],
                curtail=ones * [[0.2], [0.0], [0.4]],
                breach=np.zeros((3, T)), n_failed=0)

    row = study.load_case_row(study.NETWORK_DOE, 2, np.full(T, 100.0), 90.0,
                              points, case)

    # household 0 = mean of its copies (2, 6) -> 4; household 1 -> 4
    assert row["peak_kw"] == pytest.approx(8.0)
    assert row["zone_peak_kw"] == pytest.approx(100.0 - 6.0 + 8.0)
    assert row["savings_per_day"] == pytest.approx(2.0 + 5.0)
    assert row["import_shortfall_kwh"] == pytest.approx(0.5 * T * DT)
    assert row["curtailed_kwh"] == pytest.approx(0.3 * T * DT)
    assert row["export_excess_kwh"] == pytest.approx(0.3 * T * DT)
    assert row["n_customers"] == 3
