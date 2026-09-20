"""
Cross-method consistency tests for the VPP coupling layer (vpp/).

The two methods make claims about each other that can be checked directly on
a small synthetic ensemble (no data.csv needed):

  * Method A hard == Method A soft when the envelope is feasible;
  * Method A's coupling duals are non-zero exactly where the cap binds, with
    the sign OSQP's convention dictates on each side of the envelope;
  * two-stage allocation (Method B) is feasible when every slice is, and
    can never beat A;
  * vpp_common.feeder_envelope('tight_tou') is osqp_daily_with_DOE's
    'tight' envelope times N (a documented cross-script claim);
  * a HouseholdSolver with a per-household DOE reproduces
    osqp_daily_with_DOE.solve_battery -- Part B's local problem IS Part A's QP.

Two fixtures cover both sides of the coupling: `ensemble` binds the IMPORT
cap (winter evening peak), `ensemble_export` binds the EXPORT cap (summer
midday PV). The export fixture exists because a sign error in the coupling
duals or the two-stage curtailment credit would be invisible to import-only
tests (audit finding, 2026-09-01).

Every dispatch is also run through validate_dispatch (SOC, rate,
neutrality). Verified 2026-08-18; export side added 2026-09-02; trimmed to
Methods A and B on 2026-09-11.
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
D = pytest.importorskip("dispatch.osqp_daily_with_DOE")
ts = pytest.importorskip("vpp.two_stage_doe_allocation.two_stage_doe_allocation")

T = vc.T
E_MAX = 10.0


def synthetic_household(i, tariff):
    """A winter-like day: morning/evening load bumps, small PV, scaled per i."""
    hrs = np.arange(T) * vc.DT
    load = 0.4 + 0.9 * np.exp(-((hrs - 7.5) / 1.5) ** 2) \
        + (1.6 + 0.3 * i) * np.exp(-((hrs - 18.5) / 2.0) ** 2)
    pv = (0.6 + 0.2 * i) * np.maximum(np.sin(np.pi * (hrs - 7.0) / 10.0), 0.0)
    pv[(hrs < 7.0) | (hrs > 17.0)] = 0.0
    h, b_unc, sav = base.optimise_H(load, pv, tariff, E_MAX, "fit")
    return vc.HouseholdDay(name=f"synth{i}", customer=i, date="2010-07-01",
                           load=load, pv=pv, net=load - pv, h=h, e_max=E_MAX,
                           b_uncoupled=b_unc, savings_uncoupled=sav)


@pytest.fixture(scope="module")
def ensemble():
    tariff = base.build_tariff()
    households = [synthetic_household(i, tariff) for i in range(4)]
    agg_unc = vc.aggregate_pi(households,
                              np.vstack([hh.b_uncoupled for hh in households]))
    # import cap at 75 % of the uncoupled aggregate peak: binds, stays feasible
    cap = 0.75 * agg_unc.max()
    d_min = -np.inf * np.ones(T)
    d_max = cap * np.ones(T)
    assert vc.envelope_violation(agg_unc, d_min, d_max)["max_kw"] > 0.5
    return households, tariff, d_min, d_max


@pytest.fixture(scope="module")
def centralised(ensemble):
    households, _tariff, d_min, d_max = ensemble
    res = vc.solve_centralised(households, d_min, d_max)
    assert res.status == "solved"
    return res


def synthetic_pv_household(i, tariff):
    """A summer-like day: big midday PV over a light load -- the export
    regime. Per-household surplus stays below P_MAX so no envelope slice
    ever needs the curtailment relaxation."""
    hrs = np.arange(T) * vc.DT
    load = 0.35 + 0.5 * np.exp(-((hrs - 19.0) / 2.0) ** 2)
    pv = (3.2 + 0.4 * i) * np.maximum(np.sin(np.pi * (hrs - 7.0) / 10.0), 0.0)
    pv[(hrs < 7.0) | (hrs > 17.0)] = 0.0
    h, b_unc, sav = base.optimise_H(load, pv, tariff, E_MAX, "fit")
    return vc.HouseholdDay(name=f"pv{i}", customer=i, date="2010-12-21",
                           load=load, pv=pv, net=load - pv, h=h, e_max=E_MAX,
                           b_uncoupled=b_unc, savings_uncoupled=sav)


@pytest.fixture(scope="module")
def ensemble_export():
    tariff = base.build_tariff()
    households = [synthetic_pv_household(i, tariff) for i in range(4)]
    agg_unc = vc.aggregate_pi(households,
                              np.vstack([hh.b_uncoupled for hh in households]))
    # export cap at 75 % of the uncoupled aggregate export peak: binds,
    # stays feasible (agg_unc is most negative at midday)
    cap = 0.75 * (-agg_unc.min())
    assert cap > 1.0, "fixture must actually export"
    d_min = -cap * np.ones(T)
    d_max = np.inf * np.ones(T)
    assert vc.envelope_violation(agg_unc, d_min, d_max)["max_kw"] > 0.5
    return households, tariff, d_min, d_max


@pytest.fixture(scope="module")
def centralised_export(ensemble_export):
    households, _tariff, d_min, d_max = ensemble_export
    res = vc.solve_centralised(households, d_min, d_max)
    assert res.status == "solved"
    return res


def _valid(households, B):
    return all(not vc.validate_dispatch(hh, b) for hh, b in zip(households, B))


def _viol(households, B, d_min, d_max):
    return vc.envelope_violation(vc.aggregate_pi(households, B), d_min, d_max)["max_kw"]


def test_centralised_is_feasible_valid_and_binding(ensemble, centralised):
    households, _t, d_min, d_max = ensemble
    assert _valid(households, centralised.B)
    assert _viol(households, centralised.B, d_min, d_max) < 1e-4
    assert int((np.abs(centralised.y_couple) > 1e-6).sum()) >= 1   # the cap binds


def test_soft_equals_hard_when_feasible(ensemble, centralised):
    households, _t, d_min, d_max = ensemble
    soft = vc.solve_centralised(households, d_min, d_max, soft=True)
    assert soft.status == "solved"
    assert (soft.slack_up + soft.slack_lo).sum() * vc.DT < 1e-4
    assert soft.objective == pytest.approx(centralised.objective, rel=1e-6)


@pytest.mark.parametrize("rule", ts.RULES)
def test_two_stage_is_feasible_and_never_beats_centralised(ensemble, centralised, rule):
    households, _t, d_min, d_max = ensemble
    B, curtail_kw, shortfall_kw, n_failed = ts.run_rule(
        rule, households, d_min, d_max)
    assert n_failed == 0
    assert curtail_kw.max() < 1e-6 and shortfall_kw.max() < 1e-6
    assert _valid(households, B)
    assert _viol(households, B, d_min, d_max) < 1e-3
    assert vc.objective_surrogate(households, B) >= centralised.objective - 1e-6


# ==========================================================
# EXPORT-CAP SIDE (ensemble_export) -- the paths an import-only
# suite cannot see: the upper coupling bound's dual sign and the
# two-stage curtailment credit
# ==========================================================

def test_centralised_export_cap_binds_with_positive_duals(
        ensemble_export, centralised_export):
    households, _t, d_min, d_max = ensemble_export
    assert _valid(households, centralised_export.B)
    assert _viol(households, centralised_export.B, d_min, d_max) < 1e-4
    # export cap active -> upper coupling bound -> y strictly positive
    assert (centralised_export.y_couple > 1e-6).any()
    assert centralised_export.y_couple.min() > -1e-6


@pytest.mark.parametrize("rule", ts.RULES)
def test_two_stage_export_side_is_feasible_after_curtailment(
        ensemble_export, centralised_export, rule):
    households, _t, d_min, d_max = ensemble_export
    B, curtail_kw, shortfall_kw, n_failed = ts.run_rule(
        rule, households, d_min, d_max)
    assert n_failed == 0
    assert shortfall_kw.max() < 1e-6           # no import cap here
    agg_delivered = vc.aggregate_pi(households, B) + curtail_kw
    assert vc.envelope_violation(agg_delivered, d_min, d_max)["max_kw"] < 1e-3
    assert vc.objective_surrogate(households, B) \
        >= centralised_export.objective - 1e-6


def test_tight_tou_envelope_matches_doe_script_times_n():
    N = 7
    d_min, d_max = vc.feeder_envelope("tight_tou", N, export_limit_kw=1.5)
    doe_min, doe_max = D.generate_doe_envelope("tight", base_export_limit=3.0)
    assert np.allclose(d_min, N * doe_min)
    assert np.isposinf(d_max).all() and np.isposinf(doe_max).all()


def test_household_solver_reproduces_part_a_doe_solve(ensemble):
    """Part B's local subproblem with a per-household envelope is Part A's
    DOE-constrained QP: same weights, same net, same bounds -> same b."""
    households, _t, _dmin, _dmax = ensemble
    hh = households[0]
    doe_min, doe_max = D.generate_doe_envelope("conservative", base_export_limit=3.0)
    res = D.solve_battery(hh.load, hh.pv, hh.h, hh.e_max, doe_min, doe_max)
    assert res.doe_feasible and res.curtail.max() < 1e-4   # battery alone meets it
    solver = vc.HouseholdSolver(hh, d_min=doe_min, d_max=doe_max)
    b_b, status = solver.solve()
    assert "solved" in status
    assert np.allclose(res.b, b_b, atol=1e-4)
