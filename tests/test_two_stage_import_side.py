"""
Import-side allocation and soft household envelopes for the two-stage
method (vpp/two_stage_doe_allocation) -- no data.csv needed.

  * floor_fill is the import-side mirror of water_fill: equal allowance
    above per-household floors, sums to the budget, passes floors through
    when the budget cannot cover them, and leaves unbounded intervals
    unbounded;
  * the maxmin rule's import slices sum to the feeder cap and never drop
    below a household's unavoidable import (net - P_MAX)+;
  * HouseholdSolver(soft=True) equals the hard solve when the envelope is
    feasible and reports the shortfall (never a failed solve) when it is
    not, with pi - s_up inside the cap;
  * run_rule(soft=True) never falls back to zero dispatch under a cap that
    makes the hard solve fail, and reports the same excess post hoc.
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
ts = pytest.importorskip("vpp.two_stage_doe_allocation.two_stage_doe_allocation")
from test_vpp_methods import synthetic_household  # noqa: E402

T = vc.T
DT = vc.DT


def test_floor_fill_equalises_above_floors():
    alloc = ts.floor_fill(12.0, [0.0, 0.0, 3.0, 6.0])
    np.testing.assert_allclose(alloc, [1.5, 1.5, 3.0, 6.0])
    assert alloc.sum() == pytest.approx(12.0)


def test_floor_fill_is_equal_split_without_binding_floors():
    np.testing.assert_allclose(ts.floor_fill(8.0, [0.0, 1.0, 1.5]),
                               [8 / 3] * 3)


def test_floor_fill_passes_floors_through_when_budget_is_short():
    floors = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(ts.floor_fill(10.0, floors), floors)


def test_floor_fill_keeps_unbounded_budget_unbounded():
    assert np.isposinf(ts.floor_fill(np.inf, [0.0, 2.0])).all()


@pytest.fixture(scope="module")
def households():
    tariff = base.build_tariff()
    return [synthetic_household(i, tariff) for i in range(4)], tariff


@pytest.fixture(scope="module")
def harsh_import_cap(households):
    """A flat cap at 20 % of the no-battery aggregate peak: as equal
    slices it is energy-infeasible for every synthetic household (each
    imports 14-17 kWh/day against a 10 kWh battery)."""
    hh, _t = households
    p0 = np.sum([h.net for h in hh], axis=0)
    d_min = -np.inf * np.ones(T)
    d_max = 0.2 * p0.max() * np.ones(T)
    return d_min, d_max


def test_maxmin_import_slices_sum_to_cap_and_respect_floors(
        households, harsh_import_cap):
    hh, _t = households
    d_min, d_max = harsh_import_cap

    _d_min_i, d_max_i = ts.allocate("maxmin", hh, d_min, d_max)

    floors = ts.import_floors(hh)
    assert d_max_i.shape == (len(hh), T)
    np.testing.assert_allclose(d_max_i.sum(axis=0), d_max, rtol=1e-9)
    assert np.all(d_max_i >= floors - 1e-9)
    # the rule differs from an equal split exactly where a floor binds
    equal = np.tile(d_max / len(hh), (len(hh), 1))
    binding = floors > equal
    assert np.all(d_max_i[binding] == floors[binding])
    assert np.all(d_max_i[~binding] <= equal[~binding] + 1e-9)


def test_prorata_surplus_import_slices_follow_need(households, harsh_import_cap):
    hh, _t = households
    d_min, d_max = harsh_import_cap
    _d_min_i, d_max_i = ts.allocate("prorata_surplus", hh, d_min, d_max)
    need = np.vstack([np.maximum(h.net, 0.0) for h in hh])
    k = int(np.argmax(need.sum(axis=0)))
    np.testing.assert_allclose(d_max_i[:, k],
                               d_max[k] * need[:, k] / need[:, k].sum())


def test_household_soft_equals_hard_when_feasible(households):
    hh, _t = households
    h0 = hh[0]
    d_max = h0.net.max() + 1.0                 # never binds
    hard = vc.HouseholdSolver(h0, d_max=np.full(T, d_max))
    soft = vc.HouseholdSolver(h0, d_max=np.full(T, d_max), soft=True)

    b_hard, s_hard = hard.solve()
    b_soft, s_soft = soft.solve()

    assert s_hard == "solved" and s_soft == "solved"
    np.testing.assert_allclose(b_soft, b_hard, atol=1e-3)
    assert soft.slack_up.max() < 1e-4 and soft.slack_lo.max() < 1e-4


def test_household_soft_reports_shortfall_instead_of_failing(households):
    hh, _t = households
    h0 = hh[0]
    zero_cap = np.zeros(T)                     # no import allowed all day

    hard = vc.HouseholdSolver(h0, d_max=zero_cap)
    _b, status_hard = hard.solve()
    assert "solved" not in status_hard         # energy-infeasible as hard rows

    soft = vc.HouseholdSolver(h0, d_max=zero_cap, soft=True)
    b, status_soft = soft.solve()
    assert status_soft == "solved"
    assert vc.validate_dispatch(h0, b) == []
    pi = h0.net - b
    assert soft.slack_up.sum() * DT > 1.0     # it needed the slack
    assert np.all(pi - soft.slack_up <= zero_cap + 1e-4)
    # slack is used only where the cap is actually breached
    assert np.all(soft.slack_up[pi <= 1e-6] < 1e-4)


@pytest.mark.parametrize("rule", ts.RULES)
def test_run_rule_soft_never_falls_back_to_zero_dispatch(
        households, harsh_import_cap, rule):
    hh, _t = households
    d_min, d_max = harsh_import_cap

    _B, _c, _s, n_failed_hard = ts.run_rule(rule, hh, d_min, d_max)
    B, curtail_kw, shortfall_kw, n_failed_soft = ts.run_rule(
        rule, hh, d_min, d_max, soft=True)

    assert n_failed_hard > 0                   # documents what soft fixes
    assert n_failed_soft == 0
    assert all(vc.validate_dispatch(h, b) == [] for h, b in zip(hh, B))
    assert all(b.any() for b in B)             # nobody dropped out
    assert curtail_kw.max() < 1e-6             # no export cap here
    assert shortfall_kw.sum() * DT > 0.0
    # post-hoc excess equals the solver slack summed over households
    d_min_i, d_max_i = ts.allocate(rule, hh, d_min, d_max)
    excess = sum(np.maximum(h.net - b - d_max_i[i], 0.0)
                 for i, (h, b) in enumerate(zip(hh, B)))
    np.testing.assert_allclose(shortfall_kw, excess, atol=1e-6)
