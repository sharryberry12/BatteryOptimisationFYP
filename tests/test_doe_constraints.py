"""
Regression tests for the DOE rows in osqp_daily_with_DOE.py.

Background (2026-08-16): the DOE constraint had never been enforced. The
persistent OSQP workspace was set up with the 2T+1 base rows and each solve
then called ``solver.update(A=<extended matrix>, l=..., u=...)`` -- osqp 1.x
``update(**kwargs)`` silently ignores an ``A`` keyword, so the solver kept
solving the unconstrained problem and every DOE scenario produced the same
dispatch (and the same savings) as no DOE at all.

Formulation under test (2026-08-19): x = [b, c, s] with curtailed PV c and
import shortfall s; p = load - pv + c - b; the export cap is hard (always
feasible via c), the import cap is soft (s reports what load the battery
could not cover). These tests use synthetic days only.
"""

import numpy as np
import pytest

D = pytest.importorskip("osqp_daily_with_DOE")
BASE = pytest.importorskip("osqp_daily")

T = D.T
E_MAX = 10.0
INF = np.inf * np.ones(T)


def peaked_pv_day():
    """0.5 kW flat load, 5 kW-peak sinusoidal PV over 10 h: the unconstrained
    QP exports ~3.06 kW at the peak, so an export cap between ~2.5 and 3 kW
    binds without curtailment, while <= 2 kW exceeds the 5 kWh SOC headroom
    and needs curtailment."""
    load = np.full(T, 0.5)
    pv = np.zeros(T)
    pv[16:36] = 5.0 * np.sin(np.linspace(0.0, np.pi, 20))
    return load, pv


def evening_load_day(evening_kw=4.0):
    """0.4 kW base load, `evening_kw` from 18:00 to 24:00, 2 kW PV midday:
    an import cap below evening_kw needs battery discharge; below
    evening_kw - P_MAX it cannot be met at all."""
    load = np.full(T, 0.4)
    load[36:48] = evening_kw
    pv = np.zeros(T)
    pv[16:34] = 2.0
    return load, pv


def flat_export_cap(cap_kw):
    return -cap_kw * np.ones(T), INF.copy()


def flat_import_cap(cap_kw):
    return -INF.copy(), cap_kw * np.ones(T)


def h0():
    return D.build_H0_diag(D.build_tariff())


def soc(b):
    return 0.5 * E_MAX - D.DT * np.cumsum(b)


def assert_battery_valid(b):
    assert abs(b).max() <= D.P_MAX + 1e-6
    assert soc(b).min() >= -1e-4 and soc(b).max() <= E_MAX + 1e-4
    assert b.sum() == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# workspace structure
# ----------------------------------------------------------------------

def test_workspace_has_relief_variables_and_fixed_rows():
    """x = [b, c, s]; rows: rate, SOC, neutrality, c box, s box, export,
    import -- all present from setup so only bounds change per day."""
    A, l, u, _ = D.build_constraints(E_MAX)
    assert A.shape == (D.N_ROWS, D.N_VAR) == (6 * T + 1, 3 * T)
    A = A.toarray()
    r0 = D.N_BASE_ROWS
    assert np.array_equal(A[r0:r0 + T, T:2 * T], np.eye(T))          # c box
    assert np.array_equal(A[r0 + T:r0 + 2 * T, 2 * T:], np.eye(T))   # s box
    exp = A[r0 + 2 * T:r0 + 3 * T]
    assert np.array_equal(exp[:, :T], -np.eye(T)) and np.array_equal(exp[:, T:2 * T], np.eye(T))
    imp = A[r0 + 3 * T:]
    assert np.array_equal(imp[:, 2 * T:], -np.eye(T))
    # inactive by default: c pinned to 0, export/import rows +-inf
    assert (u[r0:r0 + T] == 0).all()
    assert np.isneginf(l[r0 + 2 * T:]).all() and np.isposinf(u[r0 + 2 * T:]).all()


# ----------------------------------------------------------------------
# no envelope
# ----------------------------------------------------------------------

def test_no_envelope_matches_base_script_and_uses_no_relief():
    load, pv = peaked_pv_day()
    res = D.solve_battery(load, pv, h0(), E_MAX)
    b_base = BASE.solve_battery(load, pv, h0(), E_MAX)
    assert res.doe_feasible
    assert np.allclose(res.b, b_base, atol=1e-5)
    assert res.curtail.max() < 1e-6 and res.import_slack.max() < 1e-6


# ----------------------------------------------------------------------
# export side
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cap_kw", [3.0, 2.5])
def test_feasible_export_cap_binds_without_curtailment(cap_kw):
    load, pv = peaked_pv_day()
    free = D.solve_battery(load, pv, h0(), E_MAX)
    res = D.solve_battery(load, pv, h0(), E_MAX, *flat_export_cap(cap_kw))
    p_free = load - pv - free.b
    p = load - pv + res.curtail - res.b
    assert p_free.min() < -cap_kw - 0.01, "test day must exceed the cap unconstrained"
    assert res.doe_feasible
    assert p.min() >= -cap_kw - 1e-4                  # respected ...
    assert p.min() == pytest.approx(-cap_kw, abs=1e-3)   # ... and active
    assert res.curtail.max() < 1e-4                   # the battery alone did it
    assert not np.allclose(free.b, res.b)
    assert_battery_valid(res.b)


@pytest.mark.parametrize("cap_kw, soc_reaches", [
    (1.5, 10.0),    # headroom exhausted, then curtail
    (0.5, 10.0),
    (0.0, 8.0),     # p >= 0 also throttles DISCHARGE to the load, so the battery
                    # can only cycle what it can push out later under the cap
])
def test_tight_export_cap_is_met_by_curtailing_only_after_the_battery_is_full(cap_kw, soc_reaches):
    """Below the SOC headroom the cap cannot be met by charging alone: the
    QP must curtail. Curtailment is a last resort (penalised above any
    flattening gain), so the battery is used as far as the cap lets it and
    no more PV than necessary is spilled; the export cap is still met exactly."""
    load, pv = peaked_pv_day()
    res = D.solve_battery(load, pv, h0(), E_MAX, *flat_export_cap(cap_kw))
    p = load - pv + res.curtail - res.b
    assert "solved" in res.status
    assert res.doe_feasible
    assert p.min() >= -cap_kw - 1e-4
    assert res.curtail.max() > 0.1
    assert (res.curtail <= pv + 1e-6).all()           # only real PV
    assert soc(res.b).max() >= soc_reaches - 0.02     # battery used first
    assert_battery_valid(res.b)
    # curtailment never exceeds what the cap requires beyond the battery:
    # wherever c > 0 the export row is tight
    tight = np.abs(p + cap_kw) < 1e-3
    assert tight[res.curtail > 1e-3].all()


# ----------------------------------------------------------------------
# import side
# ----------------------------------------------------------------------

def test_import_cap_the_battery_can_meet_binds_with_no_shortfall():
    load, pv = evening_load_day(evening_kw=2.5)      # 2 kW cap needs 0.5 kW x 6 h = 3 kWh
    free = D.solve_battery(load, pv, h0(), E_MAX)
    res = D.solve_battery(load, pv, h0(), E_MAX, *flat_import_cap(2.0))
    p_free = load - pv - free.b
    p = load - pv + res.curtail - res.b
    assert p_free.max() > 2.0 + 0.1
    assert res.doe_feasible and res.import_slack.max() < 1e-4
    assert p.max() <= 2.0 + 1e-4
    assert res.curtail.max() < 1e-6                   # no export cap -> no curtailment
    assert_battery_valid(res.b)


def test_impossible_import_cap_reports_the_shortfall():
    """4 kW of evening load against a 1 kW cap needs 3 kW of discharge for
    6 h = 18 kWh from a 10 kWh battery: the shortfall must be positive,
    equal to the breach of p, and the dispatch must still be valid."""
    load, pv = evening_load_day(evening_kw=4.0)
    res = D.solve_battery(load, pv, h0(), E_MAX, *flat_import_cap(1.0))
    p = load - pv + res.curtail - res.b
    assert not res.doe_feasible
    assert res.import_slack.sum() * D.DT > 5.0
    breach = np.maximum(p - 1.0, 0.0)
    assert np.allclose(breach, res.import_slack, atol=1e-3)
    assert_battery_valid(res.b)


def test_both_sides_at_once():
    load, pv = peaked_pv_day()
    res = D.solve_battery(load, pv, h0(), E_MAX, -1.0 * np.ones(T), 1.0 * np.ones(T))
    p = load - pv + res.curtail - res.b
    assert p.min() >= -1.0 - 1e-4 and p.max() <= 1.0 + res.import_slack.max() + 1e-4
    assert res.curtail.sum() > 0
    assert_battery_valid(res.b)


# ----------------------------------------------------------------------
# envelope generation, simulate_day, scenarios
# ----------------------------------------------------------------------

def test_generate_doe_envelope_import_limit():
    dmin, dmax = D.generate_doe_envelope("conservative", 3.0)
    assert np.isposinf(dmax).all()
    dmin2, dmax2 = D.generate_doe_envelope("conservative", 3.0, base_import_limit=2.0)
    assert np.allclose(dmin2, dmin) and np.allclose(dmax2, 2.0)
    dmin3, dmax3 = D.generate_doe_envelope("none", 3.0, base_import_limit=2.0)
    assert np.isneginf(dmin3).all() and np.allclose(dmax3, 2.0)


def test_simulate_day_reports_compliance_curtailment_and_shortfall():
    load, pv = peaked_pv_day()
    tariff = D.build_tariff()
    ok = D.simulate_day(load, pv, tariff, "fit", E_MAX, *flat_export_cap(2.5))
    assert ok.doe_compliant and ok.doe_slack.max() < 1e-4 and ok.curtail.max() < 1e-4
    cut = D.simulate_day(load, pv, tariff, "fit", E_MAX, *flat_export_cap(1.5))
    assert cut.doe_compliant                          # met, via curtailment
    assert cut.curtail.sum() * D.DT > 1.0
    assert cut.p.min() >= -1.5 - 1e-4
    # curtailed PV is not credited: savings must be lower than the
    # feasible-cap case (same day, less FiT / self-consumption)
    assert cut.savings < ok.savings
    load2, pv2 = evening_load_day(4.0)
    short = D.simulate_day(load2, pv2, tariff, "fit", E_MAX, *flat_import_cap(1.0))
    assert not short.doe_compliant and short.doe_slack.max() > 0.5
    assert np.allclose(short.doe_slack, short.import_slack, atol=1e-3)


def test_generated_scenarios_give_different_dispatch_on_a_binding_day():
    """The symptom that exposed the 2026-08-16 bug was identical dispatch
    (hence identical annual savings) under every scenario."""
    load, pv = peaked_pv_day()
    tariff = D.build_tariff()
    profiles = {}
    for scenario in ("none", "conservative", "tight"):
        doe_min, doe_max = D.generate_doe_envelope(scenario, base_export_limit=3.5)
        profiles[scenario] = D.simulate_day(load, pv, tariff, "fit", E_MAX,
                                            doe_min, doe_max).p
    assert not np.allclose(profiles["none"], profiles["conservative"])
    assert not np.allclose(profiles["conservative"], profiles["tight"])
