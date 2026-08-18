"""
diag_violation_attribution.py -- where do the Elermore Vale voltage
violations come from, and what does the QP change?

Re-runs baseline (no battery) and QP dispatch for a spread of days and
splits every violation-point (monitored load x half-hour outside
0.94-1.10 pu) three ways:

  * over vs under voltage,
  * by hour of day (which tariff block / PV window),
  * over-voltage points on the LV feeder of a chosen transformer
    (default HP00007159, the +2.56 % boost-tap unit in the GLM) vs the
    rest, and under-voltage points in the 22:00-24:00 off-peak block
    (synchronised battery charging) vs the rest.

Companion to NETWORK_AWARE_DISPATCH.md ("91 % of the QP's residual
over-voltage is one transformer; 82 % of its under-voltage is 22:00-24:00").

Usage (repo root):
    python diag_violation_attribution.py [--profiles profiles/fit_profiles.csv]
                                         [--every 15] [--feeder 7159]
"""

import argparse
import logging

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from paths import GLM_COMMON, GLM_DIR, PROFILES  # noqa: E402
from network import elermorevale_openDSS as ev  # noqa: E402

logging.disable(logging.INFO)


def load_bus(name):
    ev.dss.ActiveCircuit.SetActiveElement("Load." + name)
    return ev.dss.ActiveCircuit.ActiveCktElement.BusNames[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profiles", default=str(PROFILES / "fit_profiles.csv"))
    ap.add_argument("--every", type=int, default=15,
                    help="simulate every N-th day (default 15 -> 25 days)")
    ap.add_argument("--feeder", default="7159",
                    help="LV feeder id whose over-voltage share to report "
                         "(bus names fdr_<id>_lv_...; default 7159 = HP00007159)")
    ap.add_argument("--glm-dir", default=str(GLM_DIR))
    ap.add_argument("--common-dir", default=str(GLM_COMMON))
    args = ap.parse_args()

    profiles = ev.load_profiles_from_csv(args.profiles)
    ev.build_elermorevale(args.glm_dir, args.common_dir, skip_generators=True)
    lcm = ev.map_customers_to_network_loads(sorted(profiles.keys()),
                                            ev.get_network_load_names())
    mon = ev.select_monitored_loads(lcm, n_monitors=100)
    on_feeder = np.array([f"fdr_{args.feeder}_" in load_bus(n) for n in mon])
    n_days = max(len(d) for d in profiles.values())
    days = list(range(0, n_days, args.every))
    print(f"{len(mon)} monitors, {int(on_feeder.sum())} on feeder {args.feeder}; "
          f"{len(days)} days (every {args.every}th)")

    tot = {k: np.zeros(ev.T, dtype=int) for k in
           ("base_over", "base_under", "qp_over", "qp_under")}
    feeder = {"base_over": 0, "qp_over": 0}
    for d in days:
        for tag, base in (("base", True), ("qp", False)):
            r = ev.simulate_scenario(args.glm_dir, args.common_dir, lcm, mon,
                                     profiles, d, use_baseline=base)
            V = np.array([r["voltages"][n] for n in mon])
            over, under = V > ev.V_UPPER_PU, V < ev.V_LOWER_PU
            tot[f"{tag}_over"] += over.sum(axis=0)
            tot[f"{tag}_under"] += under.sum(axis=0)
            feeder[f"{tag}_over"] += int(over[on_feeder].sum())

    def by_hour(v):
        return ", ".join(f"{i / 2:04.1f}h:{n}" for i, n in enumerate(v) if n)

    for tag in ("base", "qp"):
        o, u = tot[f"{tag}_over"], tot[f"{tag}_under"]
        late = u[44:].sum()
        print(f"\n[{tag}] over={o.sum()}  under={u.sum()}")
        print(f"  over on feeder {args.feeder}: {feeder[f'{tag}_over']} "
              f"({100 * feeder[f'{tag}_over'] / max(1, o.sum()):.0f} %)")
        print(f"  under in 22:00-24:00: {late} ({100 * late / max(1, u.sum()):.0f} %)")
        print("  over by hour :", by_hour(o))
        print("  under by hour:", by_hour(u))


if __name__ == "__main__":
    main()
