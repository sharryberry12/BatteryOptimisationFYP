"""
voltage_sensitivity.py
======================

Numerical voltage sensitivities of the Elermore Vale OpenDSS model: how
much the voltage at each monitored load moves per kW of extra import at
each load, around a chosen operating point.

    dv_dp[m, j] = d V_m / d p_j      (p.u. per kW, p > 0 = import)

Each column is one perturbed snapshot power flow (forward difference),
so the matrix is the engine's own linearisation -- unbalanced, with the
real impedances and load model -- rather than an analytic approximation.
It feeds the network-aware operating envelopes (vpp/network_doe.py),
where the voltage limits become linear rows in the customers' powers:

    V_m  ~=  v0_m + sum_j dv_dp[m, j] * (p_j - p0_j)

The functions act on the circuit currently loaded in the engine: build
it first with elermorevale_openDSS.build_elermorevale(). Loads are driven
the way attach_shapes() drives them in profile mode (signed kW, unity
power factor), in snapshot mode, so an operating point here equals one
half-hour of a daily run.
"""

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DELTA_KW_DEFAULT = 1.0       # forward-difference step per load
DEAD_NODE_PU = 0.01          # a monitored node at ~0 V is a model defect
MAX_ITERATIONS = 100
# Each column is the difference of two solves, and most entries are tiny
# (median ~1e-5 p.u./kW). At the engine's default tolerance (1e-4) their
# sum over the fleet is off by up to ~17 % at the worst monitor; at 1e-6
# the error is gone for ~10 % more solve time.
SOLVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sensitivity:
    """Linearised voltages at one operating point."""
    monitored: tuple          # monitored load names, row order
    loads: tuple              # perturbed load names, column order
    v0_pu: np.ndarray         # (M,) voltages at the operating point
    dv_dp: np.ndarray         # (M, J) p.u. per kW of extra import
    delta_kw: float


def monitor_node_index(ev, monitored: Sequence[str]) -> np.ndarray:
    """
    Index into the engine's node-voltage vector of each load's first
    conductor -- the quantity the per-load voltage monitors record.
    """
    circuit = ev.dss.ActiveCircuit
    position = {name.lower(): i
                for i, name in enumerate(circuit.AllNodeNames)}
    index = []
    for name in monitored:
        if circuit.SetActiveElement(f"Load.{name}") < 0:
            raise KeyError(f"load {name!r} is not in the circuit")
        bus = circuit.ActiveCktElement.BusNames[0].lower()
        base, _, nodes = bus.partition(".")
        first = nodes.split(".")[0] if nodes else "1"
        node = f"{base}.{first}"
        if node not in position:
            raise KeyError(f"load {name!r}: node {node!r} is not solved")
        index.append(position[node])
    return np.asarray(index, dtype=int)


def apply_operating_point(ev, kw_by_load: Mapping[str, float]) -> None:
    """
    Set every listed load to its signed kW at unity power factor and
    select snapshot mode with the controls of the daily runs.
    """
    cmd = ev.dss.Text
    for name, kw in kw_by_load.items():
        cmd.Command = f"Load.{name}.kw={float(kw):.6f}"
        cmd.Command = f"Load.{name}.pf=1"
    cmd.Command = "Set mode=snapshot"
    cmd.Command = f"Set controlmode={'static' if ev.OLTC_ACTIVE else 'off'}"
    cmd.Command = f"Set maxiterations={MAX_ITERATIONS}"
    cmd.Command = "Calcvoltagebases"


def solve_voltages(ev, node_index: np.ndarray) -> np.ndarray:
    """
    Solve the current operating point; per-unit voltages (V_NOM base) at
    the given nodes. Converged=True is not trusted alone: a monitored
    node at ~0 V raises DeadMonitorError like the daily runs do.
    """
    circuit = ev.dss.ActiveCircuit
    circuit.Solution.Solve()
    if not circuit.Solution.Converged:
        raise RuntimeError("snapshot power flow did not converge")
    v_pu = np.asarray(circuit.AllBusVmag, dtype=float)[node_index] / ev.V_NOM
    if np.any(v_pu <= DEAD_NODE_PU):
        raise ev.DeadMonitorError(
            f"{int(np.sum(v_pu <= DEAD_NODE_PU))} monitored node(s) read "
            f"<= {DEAD_NODE_PU} pu in a snapshot solve")
    return v_pu


def voltage_sensitivity(ev, kw_by_load: Mapping[str, float],
                        monitored: Sequence[str],
                        perturb: Optional[Sequence[str]] = None,
                        delta_kw: float = DELTA_KW_DEFAULT) -> Sensitivity:
    """
    Sensitivity of the monitored voltages to each load in `perturb`
    (default: every load of the operating point) around kw_by_load.
    The operating point is restored before returning.
    """
    perturb = list(kw_by_load) if perturb is None else list(perturb)
    unknown = [name for name in perturb if name not in kw_by_load]
    if unknown:
        raise KeyError(f"loads not in the operating point: {unknown[:5]}")
    if delta_kw == 0.0:
        raise ValueError("delta_kw must be non-zero")

    apply_operating_point(ev, kw_by_load)
    node_index = monitor_node_index(ev, monitored)
    cmd = ev.dss.Text
    solution = ev.dss.ActiveCircuit.Solution
    tolerance_before = solution.Tolerance
    solution.Tolerance = SOLVE_TOLERANCE
    try:
        v0 = solve_voltages(ev, node_index)
        dv_dp = np.empty((len(monitored), len(perturb)))
        for col, name in enumerate(perturb):
            base_kw = float(kw_by_load[name])
            cmd.Command = f"Load.{name}.kw={base_kw + delta_kw:.6f}"
            dv_dp[:, col] = (solve_voltages(ev, node_index) - v0) / delta_kw
            cmd.Command = f"Load.{name}.kw={base_kw:.6f}"
        solve_voltages(ev, node_index)      # leave the engine at the base
    finally:
        solution.Tolerance = tolerance_before

    logger.info("voltage sensitivities: %d monitors x %d loads, "
                "delta %.2f kW", len(monitored), len(perturb), delta_kw)
    return Sensitivity(monitored=tuple(monitored), loads=tuple(perturb),
                       v0_pu=v0, dv_dp=dv_dp, delta_kw=float(delta_kw))
