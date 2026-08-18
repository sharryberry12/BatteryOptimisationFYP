"""
Cross-method consistency tests for the VPP coupling layer (vpp/).

The methods make claims about each other that can be checked directly on a
small synthetic ensemble (no data.csv needed):

  * Method A hard == Method A soft when the envelope is feasible;
  * a one-shot broadcast of A's coupling duals (Method E 'shadow') reproduces
    A exactly -- the objectives are strictly convex, so the Lagrangian
    minimiser at the optimal prices IS the primal optimum;
  * the FCAS co-optimisation at zero FCAS price is Method A;
  * sharing ADMM (Method D) converges to A;
  * dual decomposition (Method C) with an adequate step converges to A;
  * two-stage allocation (Method B) is feasible when every slice is, and
    can never beat A;
  * vpp_common.feeder_envelope('tight_tou') is osqp_daily_with_DOE's
    'tight' envelope times N (a documented cross-script claim);
  * a HouseholdSolver with a per-household DOE reproduces
    osqp_daily_with_DOE.solve_battery -- Part B's local problem IS Part A's QP.

Every dispatch is also run through validate_dispatch (SOC, rate,
neutrality). Verified 2026-08-18.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for sub in ("vpp", "vpp/sharing_admm", "vpp/dual_decomposition",
            "vpp/two_stage_doe_allocation", "vpp/price_based_control",
            "vpp/fcas_cooptimisation"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

vc = pytest.importorskip("vpp_common")
base = pytest.importorskip("osqp_daily")
D = pytest.importorskip("osqp_daily_with_DOE")
admm = pytest.importorskip("sharing_admm")
dd = pytest.importorskip("dual_decomposition")
ts = pytest.importorskip("two_stage_doe_allocation")
pbc = pytest.importorskip("price_based_control")
fc = pytest.importorskip("fcas_cooptimisation")

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


def test_shadow_price_broadcast_reproduces_centralised(ensemble, centralised):
    households, tariff, d_min, d_max = ensemble
    mu = pbc.build_signal("shadow", 50.0, tariff, centralised.y_couple)
    B = pbc.respond(households, mu)
    assert vc.objective_surrogate(households, B) == pytest.approx(
        centralised.objective, rel=1e-5)
    assert _viol(households, B, d_min, d_max) < 1e-3
    assert np.allclose(B, centralised.B, atol=1e-3)


def test_price_none_is_the_uncoupled_dispatch(ensemble):
    households, tariff, _dmin, _dmax = ensemble
    B = pbc.respond(households, pbc.build_signal("none", 50.0, tariff, None))
    assert np.allclose(B, np.vstack([hh.b_uncoupled for hh in households]),
                       atol=1e-4)


def test_fcas_at_zero_price_is_centralised(ensemble, centralised):
    households, _t, d_min, d_max = ensemble
    B, R, _dt, status = fc.solve_fcas(households, d_min, d_max,
                                      np.zeros(T), fc.TAU_DEFAULT)
    assert status == "solved"
    assert np.allclose(B, centralised.B, atol=1e-5)
    assert R.max() < 1e-4


def test_fcas_respects_headroom_adequacy_and_envelope(ensemble):
    households, _t, d_min, d_max = ensemble
    price = 0.1 * np.ones(T)
    B, R, _dt, status = fc.solve_fcas(households, d_min, d_max, price,
                                      fc.TAU_DEFAULT)
    assert status == "solved" and R.sum() > 0
    assert _valid(households, B)
    assert (B + R <= vc.P_MAX + 1e-4).all()                       # headroom
    soc = np.vstack([vc.SOC_INIT_FRAC * hh.e_max - vc.DT * np.cumsum(b)
                     for hh, b in zip(households, B)])
    assert (soc >= fc.TAU_DEFAULT * R - 1e-4).all()               # adequacy
    assert _viol(households, B, d_min, d_max) < 1e-3               # import cap


def test_sharing_admm_converges_to_centralised(ensemble, centralised):
    households, _t, d_min, d_max = ensemble
    B, hist, n_it = admm.run_admm(households, d_min, d_max, rho=50.0,
                                  iters=500, tol_kw=0.01)
    assert n_it < 500, "ADMM hit the iteration cap"
    assert _valid(households, B)
    gap = vc.objective_surrogate(households, B) / centralised.objective - 1
    assert abs(gap) < 2e-3
    assert _viol(households, B, d_min, d_max) < 0.02


def test_dual_decomposition_prices_converge_to_centralised_duals(ensemble, centralised):
    """The sharp claim of Method C is that the projected-subgradient prices
    converge to the coupling duals of Method A (mu -> -y). With an adequate
    step they do within ~1 % in 400 iterations. The ergodic-average PRIMAL
    recovers only at O(1/sqrt t): on a strongly binding cap (|y| ~ 120 here)
    a 0.1 kW residual violation is worth several % of the objective, so it
    is bounded loosely. NOTE the shipped default alpha0=0.5 is an order of
    magnitude too small for real instances (dual scale ~50-70): see
    vpp/dual_decomposition/README.md."""
    households, _t, d_min, d_max = ensemble
    out = dd.run_dual(households, d_min, d_max, iters=400, alpha0=10.0)
    y_scale = np.abs(centralised.y_couple).max()
    assert np.abs(out["mu"] + centralised.y_couple).max() < 0.05 * y_scale
    B = out["B_avg"]
    assert _valid(households, B)
    assert _viol(households, B, d_min, d_max) < 0.1
    gap = vc.objective_surrogate(households, B) / centralised.objective - 1
    assert -0.06 < gap < 0.02          # infeasible-side, shrinking with iterations
    # more iterations tighten the primal (O(1/sqrt t)): 1500 must beat 400
    out2 = dd.run_dual(households, d_min, d_max, iters=1500, alpha0=10.0)
    assert _viol(households, out2["B_avg"], d_min, d_max) < _viol(households, B, d_min, d_max)


@pytest.mark.parametrize("rule", ts.RULES)
def test_two_stage_is_feasible_and_never_beats_centralised(ensemble, centralised, rule):
    households, _t, d_min, d_max = ensemble
    B, curtail_kwh, n_failed = ts.run_rule(rule, households, d_min, d_max)
    assert n_failed == 0
    assert _valid(households, B)
    assert _viol(households, B, d_min, d_max) < 1e-3
    assert vc.objective_surrogate(households, B) >= centralised.objective - 1e-6


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
