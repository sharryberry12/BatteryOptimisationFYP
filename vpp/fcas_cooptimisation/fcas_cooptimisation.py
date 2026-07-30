"""
fcas_cooptimisation.py -- FCAS co-optimisation under DOEs
=========================================================

Extends the centralised VPP QP (Method A) with contingency-raise FCAS
enablement, and quantifies how much aggregate FCAS capacity a static vs
a dynamic feeder envelope allows -- the DOE-FCAS interaction flagged as
the strongest available result in VPP_EXTENSION.md Section 9 (compare
SAPN's finding that dynamic limits more than double export capacity at
key intervals).

Per household i, add r_i >= 0 (enabled raise, kW). Still a QP:

  headroom        b_ik + r_ik <= B_MAX          (can physically ramp up)
  energy adequacy soc_ik >= tau * r_ik          (can sustain for tau h)
  DOE interaction sum_i (pi_ik - r_ik) >= D_min_k
                  (a delivered raise response must still fit inside the
                   feeder export envelope)

  objective       sum h (net-b)^2  -  sum_k p_k * R_k,   R_k = sum_i r_ik

P gains zero diagonal entries for r (still diagonal); the FCAS price
enters through q only. Raise-only keeps the story clean; lower is
symmetric.

Run from the repo root, e.g.:
    python vpp/fcas_cooptimisation/fcas_cooptimisation.py --save
    python vpp/fcas_cooptimisation/fcas_cooptimisation.py \
        --compare static,dynamic_solar,tight_tou --fcas-price 0.2
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import osqp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vpp_common as vc  # noqa: E402
import osqp_daily as base  # noqa: E402  (path set by vpp_common)

logger = logging.getLogger("vpp.fcas")

TAU_DEFAULT = 1.0 / 12.0  # 5-minute contingency sustain, in hours


def solve_fcas(households, d_min, d_max, fcas_price, tau):
    """
    Centralised co-optimised solve. Variables x = [b_1..b_N, r_1..r_N].
    Returns (B, R, solve_time, status).
    """
    N = len(households)
    n = N * vc.T
    T_ = vc.T
    agg_net = np.sum([hh.net for hh in households], axis=0)
    A_soc = sp.csc_matrix(np.tril(np.ones((T_, T_))) * vc.DT)
    I_T = sp.eye(T_, format="csc")

    blocks, ls, us = [], [], []
    for hh in households:
        A_i, l_i, u_i = base.build_constraints(hh.e_max)
        blocks.append(A_i)
        ls.append(l_i)
        us.append(u_i)
    A_local_b = sp.block_diag(blocks, format="csc")          # rows x n
    Z_r = sp.csc_matrix((A_local_b.shape[0], n))

    blk_I = sp.block_diag([I_T] * N, format="csc")
    blk_soc = sp.block_diag([A_soc] * N, format="csc")
    soc_init = vc.SOC_INIT_FRAC * households[0].e_max

    rows = [
        # local battery rows (b only)
        sp.hstack([A_local_b, Z_r]),
        # 0 <= r <= P_MAX
        sp.hstack([sp.csc_matrix((n, n)), blk_I]),
        # headroom: b + r <= B_MAX
        sp.hstack([blk_I, blk_I]),
        # adequacy: A_soc b + tau r <= soc_init  (soc_k >= tau r_k)
        sp.hstack([blk_soc, tau * blk_I]),
        # import side: sum b >= agg_net - D_max
        sp.hstack([sp.hstack([I_T] * N), sp.csc_matrix((T_, n))]),
        # export + FCAS: sum b + sum r <= agg_net - D_min
        sp.hstack([sp.hstack([I_T] * N), sp.hstack([I_T] * N)]),
    ]
    l = np.concatenate([
        np.concatenate(ls),
        np.zeros(n),
        -np.inf * np.ones(n),
        -np.inf * np.ones(n),
        agg_net - d_max,
        -np.inf * np.ones(T_),
    ])
    u = np.concatenate([
        np.concatenate(us),
        vc.P_MAX * np.ones(n),
        vc.P_MAX * np.ones(n),
        soc_init * np.ones(n),
        np.inf * np.ones(T_),
        agg_net - d_min,
    ])
    A = sp.vstack(rows, format="csc")

    P = sp.diags(np.concatenate(
        [np.concatenate([2.0 * hh.h for hh in households]),
         np.zeros(n)]), format="csc")
    q = np.concatenate(
        [np.concatenate([-2.0 * hh.h * hh.net for hh in households]),
         np.tile(-np.asarray(fcas_price, dtype=float), N)])

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, **vc.OSQP_SETTINGS)
    t0 = time.perf_counter()
    res = prob.solve()
    dt = time.perf_counter() - t0
    if res.info.status_val not in (1, 2):
        logger.warning("FCAS OSQP status: %s", res.info.status)
        return np.zeros((N, T_)), np.zeros((N, T_)), dt, res.info.status
    B = res.x[:n].reshape(N, T_)
    R = res.x[n:].reshape(N, T_)
    return B, R, dt, res.info.status


def main():
    parser = vc.standard_argparser(
        "FCAS contingency-raise co-optimisation under feeder DOEs")
    parser.add_argument("--compare", default="static,dynamic_solar",
                        help="Comma-separated envelope scenarios to compare")
    parser.add_argument("--fcas-price", type=float, default=0.1,
                        help="Raise enablement price, $/kW per interval "
                             "(flat; spiky series are future work)")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT,
                        help="Required sustain duration in hours "
                             "(default 5 min)")
    args = parser.parse_args()
    scenarios = [s.strip() for s in args.compare.split(",") if s.strip()]

    day_arrays = vc.load_day_arrays(args.data)
    households, date_iso, tariff = vc.assemble_ensemble(
        day_arrays, args.n_households, args.date,
        mode=args.mode, e_max=args.e_max)
    logger.info("Day %s, comparing envelopes: %s", date_iso, scenarios)

    price = args.fcas_price * np.ones(vc.T)
    hours = vc.hours_axis()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    logger.info("=== FCAS co-optimisation (raise-only) ===")
    logger.info("  %-14s %12s %10s %10s %10s",
                "envelope", "R total kWh", "R mid kW", "revenue $",
                "save $/d")
    midday = slice(20, 34)  # 10:00-17:00
    for scen in scenarios:
        d_min, d_max = vc.feeder_envelope(
            scen, args.n_households,
            export_limit_kw=args.export_limit,
            import_limit_kw=args.import_limit)
        B, R, dt, status = solve_fcas(households, d_min, d_max,
                                      price, args.tau)
        if "solved" not in status:
            logger.warning("scenario %s failed (%s), skipping", scen, status)
            continue
        R_agg = R.sum(axis=0)
        revenue = float(np.sum(price * R_agg))
        savings = vc.savings_vector(households, B, tariff, args.mode)
        vc.validate_ensemble(households, B)
        logger.info("  %-14s %12.2f %10.2f %10.2f %10.2f",
                    scen, R_agg.sum() * vc.DT,
                    R_agg[midday].mean(), revenue, savings.sum())

        axes[0].plot(hours, R_agg, label=scen)
        if np.isfinite(d_min).any():
            axes[1].plot(hours, d_min, "--", label=f"{scen} envelope")

    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("enabled raise R_k (kW)")
    axes[0].set_title("Aggregate FCAS raise enablement")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("feeder export limit D_min (kW)")
    axes[1].set_title("Envelope shapes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"FCAS capacity: static vs dynamic envelopes, "
                 f"N={args.n_households}")
    vc.finish_figure(fig, args, "fcas_enablement.png", __file__)


if __name__ == "__main__":
    main()
