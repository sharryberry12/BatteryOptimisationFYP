"""
Network-aware operating envelopes (vpp/network_doe.py) on small synthetic
networks -- no GLM sources, no data.csv.

  * every operating point inside the per-customer caps satisfies every
    voltage row of the linear model (the property that makes the caps
    safe without coordination), including with mixed-sign sensitivities;
  * a network that cannot bind returns the widest useful envelope;
  * with only the substation row and equal targets the caps are equal
    slices of it;
  * a binding row cuts customers in proportion to their sensitivity;
  * a limit that zero injection cannot meet is dropped and counted, not
    used to squeeze customers to zero;
  * the envelope always contains zero exchange with the grid.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

nd = pytest.importorskip("vpp.network_doe")

V_LO, V_HI = 0.94, 1.10
P_MAX = 5.0


def linear_voltage(v0, dv_dp, p0, p):
    return v0 + dv_dp @ (np.asarray(p, dtype=float) - p0)


@pytest.fixture
def mixed_network():
    """Two monitors, three customers, cross-phase (positive) terms."""
    dv_dp = np.array([[-4e-3, -1e-3, 5e-4],
                      [3e-4, -2e-3, -3e-3]])
    v0 = np.array([0.955, 1.085])
    p0 = np.array([1.0, 0.5, -2.0])
    return dv_dp, v0, p0


def test_targets_are_own_flow_plus_full_battery_rate():
    hi, lo = nd.envelope_targets(np.array([2.0, -3.0, 0.0]), P_MAX)

    np.testing.assert_allclose(hi, [7.0, 5.0, 5.0])
    np.testing.assert_allclose(lo, [-5.0, -8.0, -5.0])


def test_every_point_inside_the_caps_satisfies_every_voltage_row(
        mixed_network):
    dv_dp, v0, p0 = mixed_network
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI,
                               cross_phase=True)

    assert env.n_binding >= 1, "the example is built so the rows bind"
    for corner in itertools.product(*zip(env.lo, env.hi)):
        v = linear_voltage(v0, dv_dp, p0, corner)
        assert v.min() >= V_LO - 1e-9 and v.max() <= V_HI + 1e-9, corner


def test_default_is_safe_when_the_fleet_moves_in_one_direction(
        mixed_network):
    dv_dp, v0, p0 = mixed_network
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)
    strict = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI,
                                  cross_phase=True)

    all_importing = linear_voltage(v0, dv_dp, p0, env.hi)
    all_exporting = linear_voltage(v0, dv_dp, p0, env.lo)
    assert env.n_binding >= 1, "caps of zero would be safe but useless"
    assert all_importing.min() >= V_LO - 1e-9
    assert all_exporting.max() <= V_HI + 1e-9
    # leaving out the cross-phase worst case can only widen the envelope:
    # it is at least as close to the targets as the strict one

    def distance(e):
        return np.hypot(np.linalg.norm(e.hi - hi_t),
                        np.linalg.norm(e.lo - lo_t))
    assert distance(env) <= distance(strict) + 1e-6


def test_envelope_contains_zero_and_stays_within_targets(mixed_network):
    dv_dp, v0, p0 = mixed_network
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)

    assert np.all(env.hi >= 0.0) and np.all(env.lo <= 0.0)
    assert np.all(env.hi <= hi_t + 1e-9) and np.all(env.lo >= lo_t - 1e-9)


def test_network_that_cannot_bind_returns_the_targets():
    dv_dp = np.full((2, 3), -1e-7)
    v0, p0 = np.array([1.0, 1.0]), np.zeros(3)
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)

    np.testing.assert_allclose(env.hi, hi_t)
    np.testing.assert_allclose(env.lo, lo_t)
    assert env.n_binding == 0 and env.n_unfixable == 0


def test_substation_row_alone_gives_equal_slices_for_equal_targets():
    dv_dp = np.zeros((1, 4))
    v0, p0 = np.array([1.0]), np.zeros(4)
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI,
                               zone_cap_kw=10.0)

    np.testing.assert_allclose(env.hi, [2.5] * 4, atol=1e-4)
    np.testing.assert_allclose(env.lo, lo_t)
    assert env.hi.sum() <= 10.0 + 1e-9


def test_binding_row_cuts_customers_in_proportion_to_sensitivity():
    dv_dp = np.array([[-3e-3, -1e-3]])
    v0, p0 = np.array([0.95]), np.zeros(2)
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)

    # 3e-3 h0 + 1e-3 h1 <= 0.01, cuts (3t, t) from (5, 5) -> t = 1
    np.testing.assert_allclose(env.hi, [2.0, 4.0], atol=1e-3)


def test_binding_over_voltage_row_cuts_export_caps_in_proportion():
    dv_dp = np.array([[-3e-3, -1e-3]])
    v0, p0 = np.array([1.09]), np.zeros(2)
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)

    # 3e-3 |lo0| + 1e-3 |lo1| <= 0.01, cuts (3t, t) from (5, 5) -> t = 1
    np.testing.assert_allclose(env.lo, [-2.0, -4.0], atol=1e-3)
    np.testing.assert_allclose(env.hi, hi_t)        # imports unaffected


def test_zero_substation_cap_pins_imports_and_leaves_exports():
    dv_dp = np.array([[-3e-3, -1e-3]])
    v0, p0 = np.array([1.09]), np.zeros(2)          # over-voltage row binds
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI,
                               zone_cap_kw=0.0)

    np.testing.assert_array_equal(env.hi, [0.0, 0.0])
    np.testing.assert_allclose(env.lo, [-2.0, -4.0], atol=1e-3)


def test_limit_zero_injection_cannot_meet_is_dropped_and_counted():
    dv_dp = np.array([[-1e-3, -1e-3]])
    v0, p0 = np.array([1.12]), np.zeros(2)        # above V_HI at no load
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)

    env = nd.interval_envelope(dv_dp, v0, p0, hi_t, lo_t, V_LO, V_HI)

    assert env.n_unfixable == 1
    np.testing.assert_allclose(env.hi, hi_t)
    np.testing.assert_allclose(env.lo, lo_t)


def test_shape_mismatch_is_rejected(mixed_network):
    dv_dp, v0, p0 = mixed_network
    hi_t, lo_t = nd.envelope_targets(p0, P_MAX)
    with pytest.raises(ValueError, match="shape"):
        nd.interval_envelope(dv_dp, v0[:1], p0, hi_t, lo_t, V_LO, V_HI)
