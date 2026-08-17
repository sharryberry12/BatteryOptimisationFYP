"""
Regression tests for the DOE rows in osqp_daily_with_DOE.py.

Background (2026-08-16): the DOE constraint had never been enforced. The
persistent OSQP workspace was set up with the 2T+1 base rows and each solve
then called ``solver.update(A=<extended matrix>, l=..., u=...)`` -- osqp 1.x
``update(**kwargs)`` silently ignores an ``A`` keyword, so the solver kept
solving the unconstrained problem and every DOE scenario produced the same
dispatch (and the same savings) as no DOE at all. The fix keeps the DOE
identity rows in the workspace from setup (inactive at +-inf) and only
updates their bounds per day. These tests use synthetic days only.
"""

import numpy as np
import pytest

D = pytest.importorskip("osqp_daily_with_DOE")
BASE = pytest.importorskip("osqp_daily")

T = D.T
E_MAX = 10.0


def peaked_pv_day():
    """0.5 kW flat load, 5 kW-peak sinusoidal PV over 10 h: the unconstrained
    QP exports ~3.06 kW at the peak, so an export cap between ~2.5 and 3 kW
    binds and is feasible, while <= 2 kW exceeds the 5 kWh SOC headroom."""
    load = np.full(T, 0.5)
    pv = np.zeros(T)
    pv[16:36] = 5.0 * np.sin(np.linspace(0.0, np.pi, 20))
    return load, pv


def flat_export_cap(cap_kw):
    return -cap_kw * np.ones(T), np.inf * np.ones(T)


def h0():
    return D.build_H0_diag(D.build_tariff())


def test_doe_rows_live_in_the_workspace_matrix():
    """A must carry the DOE identity block from setup: 3T+1 rows, and the
    last T rows are exactly I so per-day bound updates address b_k directly."""
    A, l, u, _ = D.build_constraints(E_MAX)
    assert A.shape == (3 * T + 1, T)
    assert D.N_BASE_ROWS == 2 * T + 1
    doe_block = A[D.N_BASE_ROWS:, :].toarray()
    assert np.array_equal(doe_block, np.eye(T))
    assert np.all(np.isneginf(l[D.N_BASE_ROWS:]))
    assert np.all(np.isposinf(u[D.N_BASE_ROWS:]))


def test_no_envelope_matches_base_script():
    """With no DOE the extension must reproduce osqp_daily's dispatch."""
    load, pv = peaked_pv_day()
    b_doe, feasible = D.solve_battery(load, pv, h0(), E_MAX)
    b_base = BASE.solve_battery(load, pv, h0(), E_MAX)
    assert feasible is True
    assert np.allclose(b_doe, b_base, atol=1e-5)


@pytest.mark.parametrize("cap_kw", [3.0, 2.5])
def test_feasible_export_cap_binds(cap_kw):
    load, pv = peaked_pv_day()
    b_free, _ = D.solve_battery(load, pv, h0(), E_MAX)
    b_cap, feasible = D.solve_battery(load, pv, h0(), E_MAX,
                                      *flat_export_cap(cap_kw))
    p_free, p_cap = load - pv - b_free, load - pv - b_cap
    assert p_free.min() < -cap_kw - 0.01, "test day must exceed the cap unconstrained"
    assert feasible is True
    assert p_cap.min() >= -cap_kw - 1e-4          # envelope respected ...
    assert p_cap.min() == pytest.approx(-cap_kw, abs=1e-3)   # ... and active
    assert not np.allclose(b_free, b_cap)
    soc = 0.5 * E_MAX - np.cumsum(b_cap) * D.DT
    assert soc.min() >= -1e-4 and soc.max() <= E_MAX + 1e-4
    assert abs(b_cap).max() <= D.P_MAX + 1e-6
    assert b_cap.sum() == pytest.approx(0.0, abs=1e-6)


def test_infeasible_export_cap_falls_back_to_unconstrained_dispatch():
    """A cap the battery cannot honour must not return OSQP's infeasibility
    iterate as if it were a dispatch: fall back to the unconstrained solve
    and flag it, so simulate_day() records the breach in doe_slack."""
    load, pv = peaked_pv_day()
    b_free, _ = D.solve_battery(load, pv, h0(), E_MAX)
    b_cap, feasible = D.solve_battery(load, pv, h0(), E_MAX,
                                      *flat_export_cap(1.5))
    assert feasible is False
    assert np.allclose(b_cap, b_free, atol=1e-6)


def test_simulate_day_reports_compliance_and_slack():
    load, pv = peaked_pv_day()
    tariff = D.build_tariff()
    _, _, p_ok, _, ok, slack_ok = D.simulate_day(
        load, pv, tariff, "fit", E_MAX, *flat_export_cap(2.5))
    assert ok is True and slack_ok.max() == pytest.approx(0.0, abs=1e-4)
    assert p_ok.min() >= -2.5 - 1e-4
    _, _, p_bad, _, bad, slack_bad = D.simulate_day(
        load, pv, tariff, "fit", E_MAX, *flat_export_cap(1.5))
    assert bad is False and slack_bad.max() > 0.5


def test_generated_scenarios_give_different_dispatch_on_a_binding_day():
    """The symptom that exposed the bug was identical dispatch (hence
    identical annual savings) under every scenario. The shipped scenarios
    must now produce different grid profiles on a day where they bind."""
    load, pv = peaked_pv_day()
    tariff = D.build_tariff()
    profiles = {}
    for scenario in ("none", "conservative", "tight"):
        doe_min, doe_max = D.generate_doe_envelope(scenario, base_export_limit=3.5)
        profiles[scenario] = D.simulate_day(load, pv, tariff, "fit", E_MAX,
                                            doe_min, doe_max)[2]
    assert not np.allclose(profiles["none"], profiles["conservative"])
    assert not np.allclose(profiles["conservative"], profiles["tight"])
