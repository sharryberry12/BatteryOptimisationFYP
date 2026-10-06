"""
Numerical voltage sensitivities of the Elermore Vale model
(network/voltage_sensitivity.py) -- needs the GLM sources, no data.csv.

  * the matrix has one row per monitored load and one column per
    perturbed load, and a load's own voltage falls when it imports more;
  * for a small simultaneous change at several loads the linear model
    v0 + S @ delta reproduces the full power flow;
  * the operating point is restored after the perturbations, so repeated
    calls and later solves see the same circuit;
  * unknown load names are rejected instead of silently skipped.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conftest import COMMON_DIR, GLM_DIR, requires_glm_sources  # noqa: E402

vs = pytest.importorskip("network.voltage_sensitivity")

pytestmark = requires_glm_sources

BASE_KW = 1.0
N_MONITORS = 20


@pytest.fixture(scope="module")
def circuit(ev):
    """Built circuit with every load at BASE_KW; (ev, loads, monitored)."""
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR), skip_generators=True)
    loads = ev.get_network_load_names()
    step = len(loads) // N_MONITORS
    monitored = [loads[i * step] for i in range(N_MONITORS)]
    return ev, loads, monitored


def base_point(loads):
    return {name: BASE_KW for name in loads}


def test_matrix_shape_and_own_sensitivity_is_negative(circuit):
    ev, loads, monitored = circuit
    perturb = monitored[:3]

    sens = vs.voltage_sensitivity(ev, base_point(loads), monitored,
                                  perturb=perturb)

    assert sens.dv_dp.shape == (len(monitored), len(perturb))
    assert sens.v0_pu.shape == (len(monitored),)
    assert sens.loads == tuple(perturb)
    for col, name in enumerate(perturb):
        own = sens.dv_dp[monitored.index(name), col]
        assert own < 0.0, f"{name}: importing more raised its own voltage"


def test_linear_model_matches_power_flow_for_a_small_change(circuit):
    ev, loads, monitored = circuit
    perturb = monitored[:3]
    delta_kw = np.array([2.0, -1.5, 1.0])
    sens = vs.voltage_sensitivity(ev, base_point(loads), monitored,
                                  perturb=perturb)

    moved = {**base_point(loads),
             **{n: BASE_KW + d for n, d in zip(perturb, delta_kw)}}
    vs.apply_operating_point(ev, moved)
    actual = vs.solve_voltages(ev, vs.monitor_node_index(ev, monitored))
    predicted = sens.v0_pu + sens.dv_dp @ delta_kw

    moved_by = np.abs(actual - sens.v0_pu).max()
    assert moved_by > 2e-3, "the change must be far above the tolerance"
    np.testing.assert_allclose(predicted, actual, atol=2e-4)


def test_operating_point_and_tolerance_are_restored(circuit):
    ev, loads, monitored = circuit
    solution = ev.dss.ActiveCircuit.Solution
    tolerance_before = solution.Tolerance
    first = vs.voltage_sensitivity(ev, base_point(loads), monitored,
                                   perturb=monitored[:2])

    after = vs.solve_voltages(ev, vs.monitor_node_index(ev, monitored))

    # a load left 1 kW high would move its own voltage by ~1e-3
    np.testing.assert_allclose(after, first.v0_pu, atol=1e-5)
    assert solution.Tolerance == tolerance_before
    assert vs.SOLVE_TOLERANCE < tolerance_before


def test_three_phase_load_is_read_at_its_first_conductor(circuit):
    ev, loads, _ = circuit
    dss_circuit = ev.dss.ActiveCircuit
    three_phase = None
    for name in loads:
        dss_circuit.SetActiveElement(f"Load.{name}")
        if dss_circuit.ActiveCktElement.NumPhases == 3:
            three_phase = name
            break
    assert three_phase is not None, "the model has three-phase loads"

    index = vs.monitor_node_index(ev, [three_phase])

    bus = dss_circuit.ActiveCktElement.BusNames[0].lower().split(".")[0]
    node = dss_circuit.AllNodeNames[int(index[0])].lower()
    assert node == f"{bus}.1"


def test_unknown_monitored_load_is_rejected(circuit):
    ev, loads, _ = circuit
    with pytest.raises(KeyError, match="no_such_load"):
        vs.monitor_node_index(ev, ["no_such_load"])


def test_unknown_load_is_rejected(circuit):
    ev, loads, monitored = circuit
    with pytest.raises(KeyError, match="no_such_load"):
        vs.voltage_sensitivity(ev, base_point(loads), monitored,
                               perturb=["no_such_load"])
