"""
network_doe.py
==============

Network-aware dynamic operating envelopes: a per-customer import cap and
export cap for one interval, chosen so that the customers can each move
anywhere inside their caps, without coordination, and the monitored
voltages (linear model) and the substation headroom still hold.

The idea is the right-hand-side decomposition of Mahmoodi et al., "DER
capacity assessment of active distribution systems using dynamic
operating envelopes", IEEE Trans. Smart Grid 15(2), 2024: each network
limit, written as  sum_j f_j(p_j) <= b,  is split into per-customer
shares that add up to b, and a customer's envelope is whatever keeps it
inside its shares. Reduced to real power an envelope is the interval
lo_j <= p_j <= hi_j (p > 0 import), and the worst case of a row over the
box is linear in the caps, so the shares need not be explicit variables:

    V_m(p) = v_nl,m + sum_j D_mj p_j          (D = dV/dp, mixed sign)
    D = D+ - D-,   D+ = max(D, 0),   D- = max(-D, 0)

    under-voltage:   D- hi - D+ lo  <=  v_nl - V_lo
    over-voltage:    D+ hi - D- lo  <=  V_hi - v_nl
    substation:      sum_j hi_j     <=  zone cap

Customer j's share of a row is its own term on the left.

Three deliberate choices:
  * D+ (a load raising the voltage on ANOTHER phase) is left out of the
    rows unless cross_phase=True. With it the caps are safe for every
    combination, including one phase importing at its cap while another
    exports at its cap; on Elermore Vale that worst case leaves most
    customers unable to import their own load. Without it the import caps
    keep the under-voltage limit when everyone imports at its cap, and
    the export caps keep the over-voltage limit when everyone exports at
    its cap -- the two ways a tariff-driven fleet causes violations. A
    cross-phase effect in the other direction (exports on one phase
    lowering the voltage on another) is not guarded. The two sides then
    decouple: under-voltage rows set the import caps, over-voltage rows
    the export caps.
  * the caps start from the widest envelope a customer could use (own
    flow plus the full battery rate, envelope_targets) and are pulled
    back as little as possible in the least-squares sense. A binding row
    then cuts customers in proportion to their sensitivity; the paper's
    throughput objective would give the least sensitive customers
    everything and the most sensitive ones nothing.
  * an envelope always contains zero exchange (hi >= 0 >= lo), the usual
    meaning of an import/export limit. A limit that zero exchange cannot
    meet (e.g. a transformer tap holding the no-load voltage above V_hi)
    is not the customers' to fix: that row is dropped and counted.

The projection has thousands of variables but only a few hundred rows,
all dense, so it is solved through its dual (one multiplier per row,
L-BFGS-B), where each step is a clip and two matrix products.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

DUAL_OPTIONS = dict(maxiter=10000, maxfun=40000, ftol=1e-15, gtol=1e-7)
BINDING_REL_TOL = 1e-3       # a row within 0.1 % of its limit is binding
SHRINK_WARN = 0.99           # repair moved a row's caps >1 %: worth a warning


@dataclass(frozen=True)
class IntervalEnvelope:
    """Per-customer caps for one interval (kW, p > 0 import)."""
    hi: np.ndarray            # (J,) import cap, >= 0
    lo: np.ndarray            # (J,) export cap, <= 0
    n_rows: int               # voltage rows that could bind
    n_binding: int            # voltage rows at their limit
    n_unfixable: int          # rows zero exchange cannot meet (dropped)
    shrink: float             # smallest row scaling of the final repair
    status: str


def envelope_targets(net_kw: np.ndarray,
                     p_max_kw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Widest envelope a customer can use: its own import plus full-rate
    charging, and its own export plus full-rate discharge. Returns
    (hi_target >= 0, lo_target <= 0).
    """
    net = np.asarray(net_kw, dtype=float)
    return np.maximum(net, 0.0) + p_max_kw, np.minimum(net, 0.0) - p_max_kw


def voltage_rows(dv_dp: np.ndarray, v0_pu: np.ndarray, p0_kw: np.ndarray,
                 v_lo: float, v_hi: float, cross_phase: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Worst-case voltage rows over the box, A @ [hi; lo] <= rhs, with the
    rows zero exchange cannot meet removed. Returns (A, rhs, n_removed).
    """
    v_nl = v0_pu - dv_dp @ p0_kw            # linear model at zero exchange
    neg = np.maximum(-dv_dp, 0.0)
    pos = np.maximum(dv_dp, 0.0) if cross_phase else np.zeros_like(dv_dp)
    A = np.vstack([np.hstack([neg, -pos]),              # under-voltage
                   np.hstack([pos, -neg])])             # over-voltage
    rhs = np.concatenate([v_nl - v_lo, v_hi - v_nl])
    fixable = rhs >= 0.0
    return A[fixable], rhs[fixable], int(np.sum(~fixable))


def _check_shapes(dv_dp, v0_pu, p0_kw, hi_target, lo_target):
    n_mon, n_cust = dv_dp.shape
    if v0_pu.shape != (n_mon,):
        raise ValueError(f"v0_pu shape {v0_pu.shape} != ({n_mon},)")
    for name, arr in (("p0_kw", p0_kw), ("hi_target", hi_target),
                      ("lo_target", lo_target)):
        if arr.shape != (n_cust,):
            raise ValueError(f"{name} shape {arr.shape} != ({n_cust},)")
    if np.any(hi_target < 0.0) or np.any(lo_target > 0.0):
        raise ValueError("targets must satisfy hi_target >= 0 >= lo_target")


def _project(z_target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             A: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    min ||z - z_target||^2  s.t.  lower <= z <= upper,  A z <= rhs,
    through the dual: for multipliers lam >= 0 the minimiser over the box
    is clip(z_target - A' lam), and the dual gradient is the row residual.
    """
    scale = np.abs(A).max(axis=1)           # rows are screened: scale > 0
    A_s, rhs_s = A / scale[:, None], rhs / scale

    def primal(lam: np.ndarray) -> np.ndarray:
        return np.clip(z_target - A_s.T @ lam, lower, upper)

    def negative_dual(lam: np.ndarray) -> Tuple[float, np.ndarray]:
        z = primal(lam)
        residual = A_s @ z - rhs_s
        value = 0.5 * np.sum((z - z_target) ** 2) + lam @ residual
        return -value, -residual

    out = minimize(negative_dual, np.zeros(len(rhs)), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, None)] * len(rhs),
                   options=DUAL_OPTIONS)
    return primal(out.x), ("solved" if out.success else "inaccurate")


def _repair_rows(z: np.ndarray, A: np.ndarray,
                 rhs: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Make every row hold exactly. The solver meets rows only to tolerance,
    so each row still over its limit has the caps of the customers in it
    scaled toward zero. Every term of every row is non-negative, so a
    scaling can only lower the other rows: one pass is enough. A row whose
    limit is zero pins its customers to zero exchange.
    Returns (caps, smallest factor applied to a row with a positive limit).
    """
    z, worst = z.copy(), 1.0
    for row in np.flatnonzero(A @ z > rhs):
        lhs = float(A[row] @ z)
        if lhs <= rhs[row]:
            continue                        # an earlier scaling fixed it
        factor = rhs[row] / lhs
        z[A[row] != 0.0] *= factor
        if rhs[row] > 0.0:
            worst = min(worst, factor)
    return z, worst


def interval_envelope(dv_dp: np.ndarray, v0_pu: np.ndarray,
                      p0_kw: np.ndarray, hi_target: np.ndarray,
                      lo_target: np.ndarray, v_lo: float, v_hi: float,
                      zone_cap_kw: Optional[float] = None,
                      cross_phase: bool = False) -> IntervalEnvelope:
    """
    Caps for one interval from the linearisation (v0_pu, dv_dp) taken at
    the operating point p0_kw. zone_cap_kw, when given, bounds the sum of
    the import caps (the substation row).
    """
    dv_dp = np.asarray(dv_dp, dtype=float)
    v0_pu, p0_kw = np.asarray(v0_pu, float), np.asarray(p0_kw, float)
    hi_target = np.asarray(hi_target, dtype=float)
    lo_target = np.asarray(lo_target, dtype=float)
    _check_shapes(dv_dp, v0_pu, p0_kw, hi_target, lo_target)
    if zone_cap_kw is not None and zone_cap_kw < 0.0:
        raise ValueError(f"zone_cap_kw must be >= 0, got {zone_cap_kw}")

    n_cust = dv_dp.shape[1]
    z_target = np.concatenate([hi_target, lo_target])
    A, rhs, n_unfixable = voltage_rows(dv_dp, v0_pu, p0_kw, v_lo, v_hi,
                                       cross_phase)
    can_bind = A @ z_target > rhs
    A, rhs = A[can_bind], rhs[can_bind]
    n_voltage = len(rhs)
    if zone_cap_kw is not None and hi_target.sum() > zone_cap_kw:
        zone = np.concatenate([np.ones(n_cust), np.zeros(n_cust)])
        A, rhs = np.vstack([A, zone]), np.append(rhs, zone_cap_kw)

    if len(rhs) == 0:
        return IntervalEnvelope(hi=hi_target.copy(), lo=lo_target.copy(),
                                n_rows=0, n_binding=0,
                                n_unfixable=n_unfixable, shrink=1.0,
                                status="unconstrained")

    lower = np.concatenate([np.zeros(n_cust), lo_target])
    upper = np.concatenate([hi_target, np.zeros(n_cust)])
    z, status = _project(z_target, lower, upper, A, rhs)
    z, shrink = _repair_rows(z, A, rhs)
    if shrink < SHRINK_WARN:
        logger.warning("envelope projection left a row over its limit; "
                       "its customers' caps were scaled by %.3f", shrink)
    lhs = (A @ z)[:n_voltage]
    n_binding = int(np.sum(lhs >= rhs[:n_voltage] * (1.0 - BINDING_REL_TOL)))
    return IntervalEnvelope(hi=z[:n_cust], lo=z[n_cust:],
                            n_rows=n_voltage, n_binding=n_binding,
                            n_unfixable=n_unfixable, shrink=shrink,
                            status=status)
