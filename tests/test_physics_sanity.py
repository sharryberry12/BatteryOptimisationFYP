"""
Level 3 -- physics sanity tests: known-answer power flows on the Elermore
Vale model (see MODEL_VERIFICATION.md).

Unlike Levels 1-2 these tests SOLVE the circuit, so they assert physical
identities (flat no-load profile, energy conservation, quadratic-ish loss
growth, transformer voltage-drop consistency) rather than translation
bookkeeping. Runtime is a few seconds -- each test rebuilds the circuit
because it mutates load setpoints on the singleton DSS engine.

All tests run the load-only model (skip_generators=True) at a realistic
diversified demand of ~1 kW/household: the builder's 3 kW default is a
placeholder that overloads the LV network (mean voltage ~0.74 pu), pushing
constant-P loads below their 0.85 pu model floor. 1 kW keeps the network in
its normal operating region. The final test covers the full model
(PV + generator-modelled batteries).
"""

import numpy as np
import pytest

from conftest import COMMON_DIR, GLM_DIR, requires_glm_sources

pytestmark = requires_glm_sources

REALISTIC_KW = 1.0      # diversified per-household demand for assertions
# Zone-substation transformer impedance, pu on 50 MVA (from the GLM:
# 0.0075 + 0.358j -- mirrored in build_elermorevale's TXZoneSub command).
ZONE_TX_R_PU = 0.0075
ZONE_TX_X_PU = 0.358
ZONE_TX_MVA_BASE = 50.0


def build_load_only(ev, kw=None):
    """Fresh load-only circuit; optionally set every load to kw."""
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR), skip_generators=True)
    if kw is not None:
        ev.dss.Text.Command = f"BatchEdit Load..* kW={kw}"


def solved(ev):
    assert ev.solve_snapshot(), "snapshot power flow did not converge"


def live_voltages_pu(ev):
    v = np.array(ev.dss.ActiveCircuit.AllBusVmagPu)
    v = v[v > 0.01]                     # drop de-energised bookkeeping nodes
    assert v.size > 0, "no energised buses -- dead circuit"
    return v


def source_pq_kw(ev):
    """(P, Q) delivered by the 132 kV source, positive = into the circuit."""
    p, q = ev.dss.ActiveCircuit.TotalPower
    return -p, -q


def losses_kw(ev):
    return ev.dss.ActiveCircuit.Losses[0] / 1000.0


def total_load_power_kw(ev):
    """Sum of real power actually consumed by every load element."""
    ckt = ev.dss.ActiveCircuit
    total, idx = 0.0, ckt.Loads.First
    while idx:
        powers = ckt.ActiveCktElement.Powers   # P1,Q1,P2,Q2,... per conductor
        total += sum(powers[0::2])
        idx = ckt.Loads.Next
    return total


def node_mean_pu(ev, bus_prefix):
    ckt = ev.dss.ActiveCircuit
    vpu = np.array(ckt.AllBusVmagPu)
    vals = [vpu[i] for i, name in enumerate(ckt.AllNodeNames)
            if name.startswith(bus_prefix)]
    assert vals, f"no nodes match bus prefix {bus_prefix!r}"
    return float(np.mean(vals))


def test_zero_load_gives_flat_profile_and_no_losses(ev):
    """With (near-)zero load everywhere, every bus must sit at ~1.0 pu and
    losses must vanish -- the single best detector of impedance-unit bugs:
    a mile/km mix-up (61%) would not survive this."""
    build_load_only(ev)
    ev.dss.Text.Command = "BatchEdit Load..* kW=0.0001 kvar=0"
    solved(ev)
    v = live_voltages_pu(ev)
    assert v.min() > 0.97
    assert v.max() < 1.03
    assert losses_kw(ev) < 0.5


def test_energy_conservation(ev):
    """Source injection must equal the power the loads actually consumed
    plus total losses -- catches dangling or double-counted elements."""
    build_load_only(ev, kw=REALISTIC_KW)
    solved(ev)
    p_source, _ = source_pq_kw(ev)
    balance = total_load_power_kw(ev) + losses_kw(ev)
    assert p_source == pytest.approx(balance, rel=1e-3)


def test_transformer_voltage_drop_matches_hand_calc(ev):
    """Hand-check: the 11 kV bus dip below 1.0 pu must match the classic
    approximation dV ~ P*R + Q*X (pu) across the zone transformer, using
    the impedance stated in the GLM. Verifies the impedance translation
    against the solver through an independent formula (measured agreement
    is ~3e-4 pu; tolerance leaves room for the formula's own error)."""
    build_load_only(ev, kw=REALISTIC_KW)
    solved(ev)
    p_kw, q_kvar = source_pq_kw(ev)
    p_pu = p_kw / (ZONE_TX_MVA_BASE * 1000.0)
    q_pu = q_kvar / (ZONE_TX_MVA_BASE * 1000.0)
    predicted = 1.0 - (p_pu * ZONE_TX_R_PU + q_pu * ZONE_TX_X_PU)
    solved_11kv = node_mean_pu(ev, "buszonesuboltc")
    assert solved_11kv == pytest.approx(predicted, abs=0.005)


def test_losses_scale_superlinearly_with_load(ev):
    """Scaling all loads x1.2 must depress the minimum voltage and grow
    losses faster than linearly (I^2R ~ quadratic; voltage droop makes it
    slightly worse). Catches sign/direction errors cheaply."""
    build_load_only(ev, kw=REALISTIC_KW)
    solved(ev)
    losses_base, vmin_base = losses_kw(ev), live_voltages_pu(ev).min()

    build_load_only(ev, kw=REALISTIC_KW * 1.2)
    solved(ev)
    losses_up, vmin_up = losses_kw(ev), live_voltages_pu(ev).min()

    assert vmin_up < vmin_base
    ratio = losses_up / losses_base
    assert 1.2 < ratio < 1.2 ** 3, f"loss ratio {ratio:.3f} not quadratic-ish"


def test_golden_snapshot_regression(ev):
    """Frozen reference solve (load-only, 1 kW/household), measured 2026-08
    with dss-python 0.15.7 / DSS C-API 0.14.5. Guards the whole translation
    against silent regressions; re-derive deliberately if the model or the
    engine version changes (see MODEL_VERIFICATION.md)."""
    build_load_only(ev, kw=REALISTIC_KW)
    solved(ev)
    v = live_voltages_pu(ev)
    p_source, _ = source_pq_kw(ev)
    assert p_source == pytest.approx(1945.0, rel=0.01)
    assert losses_kw(ev) == pytest.approx(133.74, rel=0.02)
    assert v.min() == pytest.approx(0.7898, abs=0.01)
    assert v.mean() == pytest.approx(0.9064, abs=0.01)
    assert v.max() == pytest.approx(0.9997, abs=0.005)


def test_full_model_snapshot_energises_network(ev):
    """The documented smoke command (python elermorevale_openDSS.py) builds
    the full model with PV + batteries and must energise the whole network.
    Guards the fix for the Storage-element engine defect: batteries are
    modelled as dispatchable generators because 2+ active Storage elements
    made DSS C-API 0.14.5 collapse to a dead all-zero solution that still
    reported Converged=True (MODEL_VERIFICATION.md 'Known defects')."""
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR), skip_generators=False)
    assert ev.solve_snapshot()
    v = live_voltages_pu(ev)
    assert v.size > 6000                # essentially all nodes energised
    assert v.min() > 0.5                # and at sane magnitudes
