"""
replay_peak_event.py
====================

Replay the worst firm-capacity exceedance event through the Elermore Vale
OpenDSS model — the physical companion to peak_duty_analysis.py.

Where peak_duty_analysis.py answers "how often / how large" in aggregate
kW-bookkeeping, this script shows the network actually being relieved:

  1. Find the worst exceedance event at the focus threshold (default 70 %
     of the 3-year peak) from the cached aggregate series.
  2. Rebuild that day's per-customer load/PV profiles from the raw CSV.
  3. Construct an explicit *peaker dispatch*: the smallest battery fleet
     (5 kW / 10 kWh households, sized exactly as in the duty analysis)
     discharges pro-rata so aggregate net demand never exceeds the
     threshold. No tariff optimisation — this is VPP-as-power-plant.
  4. Export no-battery and VPP scenario CSVs in the schema
     elermorevale_openDSS.load_profiles_from_csv() reads.
  5. Simulate both scenarios on the Elermore Vale feeder and report the
     measured feeder-head shave, voltage envelope, and losses.

Artifacts land in runs/peak_replay_<date>_<timestamp>/ (dispatch CSVs,
summary CSV, manifest.json, figures/).

Usage:
    python replay_peak_event.py                       # worst event, 70 %
    python replay_peak_event.py --focus 0.8
    python replay_peak_event.py --date 2013-01-08     # replay another day
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import osqp_daily as base  # noqa: E402
from peak_duty_analysis import (  # noqa: E402
    build_aggregate, demand_series, find_events,
    C_BLUE, C_ORANGE, C_RED, INK_2, MUTED, CACHE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DT = base.DT
T = base.T


# ==========================================================
# STAGE 1 -- worst event at the focus threshold
# ==========================================================

def locate_event(agg, focus, date_override=None):
    """
    Threshold from the FULL series peak (firm capacity is set by
    history); event selection either the max-energy event overall or the
    max-energy event on an explicitly requested day.
    """
    demand = demand_series(agg)
    peak_kw = float(demand.max())
    thr = focus * peak_kw
    ts = agg["ts"].reset_index(drop=True)
    events = find_events(ts, demand, thr)
    if events.empty:
        raise SystemExit(f"no exceedance events at {focus:.0%} of peak")

    if date_override:
        day = pd.Timestamp(date_override).date()
        on_day = events[events["start"].dt.date == day]
        if on_day.empty:
            raise SystemExit(
                f"no event on {day} at {focus:.0%} of peak; event days: "
                f"{sorted(events['start'].dt.date.unique())}")
        ev = on_day.loc[on_day["energy_kwh"].idxmax()]
    else:
        ev = events.loc[events["energy_kwh"].idxmax()]

    logger.info("Event: %s -> %s | %.1f h | max deficit %.1f kW | %.1f kWh",
                ev["start"], ev["end"], ev["duration_h"],
                ev["max_deficit_kw"], ev["energy_kwh"])
    return ev, thr, peak_kw


# ==========================================================
# STAGE 2 -- per-customer profiles for the event day
# ==========================================================

def day_profiles(data_path, day):
    """
    {customer_id: (load_kw[T], pv_kw[T])} for one date. Cached per
    (file, date) because loading the 3-year CSV takes minutes.
    """
    stem = os.path.splitext(os.path.basename(data_path))[0]
    cache = os.path.join(CACHE_DIR, f"peak_day_{stem}_{day}.csv")
    if os.path.exists(cache):
        logger.info("Loading cached day profiles: %s", cache)
        df = pd.read_csv(cache)
    else:
        full = base.load_dataset(data_path)
        sel = full[full["date_parsed"] == pd.Timestamp(day)]
        if sel.empty:
            raise SystemExit(f"{day} not present in {data_path}")
        df = sel[["Customer", "interval", "load", "pv"]].rename(
            columns={"Customer": "customer", "load": "load_kw",
                     "pv": "pv_kw"})
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache, index=False)
        logger.info("Cached day profiles -> %s", cache)

    out = {}
    for cid, grp in df.groupby("customer"):
        grp = grp.sort_values("interval")
        if len(grp) != T:
            logger.warning("customer %s has %d intervals on %s; skipped",
                           cid, len(grp), day)
            continue
        out[int(cid)] = (grp["load_kw"].to_numpy(float),
                         grp["pv_kw"].to_numpy(float))
    logger.info("Day %s: %d customers with complete profiles", day, len(out))
    return out


# ==========================================================
# STAGE 3 -- peaker fleet dispatch
# ==========================================================

def build_dispatch(profiles, thr, batt_kw, batt_kwh, usable_frac):
    """
    Size the fleet exactly as the duty analysis does (worst event on this
    day, power- vs energy-limited), spread it evenly across the customer
    list, and split the deficit pro-rata. Returns (fleet_ids, B) with B a
    {cid: b[T]} dict, b positive on discharge.
    """
    ids = sorted(profiles)
    net = np.sum([profiles[c][0] - profiles[c][1] for c in ids], axis=0)
    deficit = np.maximum(net - thr, 0.0)
    if deficit.max() <= 0:
        raise SystemExit("day aggregate never exceeds the threshold — "
                         "nothing to replay")

    # events within the day (indices are contiguous half-hours)
    above = deficit > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.view(np.int8),
                                                   [0]))))
    segs = list(zip(edges[::2], edges[1::2]))
    seg_energy = [deficit[a:b].sum() * DT for a, b in segs]

    e_usable = batt_kwh * usable_frac
    n_power = math.ceil(deficit.max() / batt_kw)
    n_energy = math.ceil(max(seg_energy) / e_usable)
    n_fleet = max(n_power, n_energy)
    binding = "power" if n_power >= n_energy else "energy"

    fleet_idx = np.unique(np.round(
        np.linspace(0, len(ids) - 1, n_fleet)).astype(int))
    fleet = [ids[i] for i in fleet_idx]
    if len(fleet) < n_fleet:
        raise SystemExit("could not select a unique fleet of size "
                         f"{n_fleet} from {len(ids)} customers")

    b = deficit / n_fleet
    assert b.max() <= batt_kw + 1e-9, "per-household power limit violated"
    for (a, z), e in zip(segs, seg_energy):
        if e / n_fleet > e_usable + 1e-9:
            logger.warning("event intervals %d..%d need %.2f kWh/household "
                           "(> %.1f usable) — recharge-between-events "
                           "assumption violated", a, z, e / n_fleet,
                           e_usable)

    logger.info("Fleet: %d households (%s-limited; power needs %d, energy "
                "needs %d) | per-household peak %.2f kW, worst-event "
                "%.2f kWh", n_fleet, binding, n_power, n_energy,
                b.max(), max(seg_energy) / n_fleet)
    B = {cid: (b.copy() if cid in set(fleet) else np.zeros(T))
         for cid in ids}
    return fleet, B, net, deficit, dict(
        n_fleet=n_fleet, binding=binding, n_power=n_power,
        n_energy=n_energy, per_hh_peak_kw=float(b.max()),
        per_hh_energy_kwh=float(max(seg_energy) / n_fleet))


# ==========================================================
# STAGE 4 -- scenario CSV export
# ==========================================================

def export_scenario(path, profiles, B, day, batt_kwh):
    """Long-format profiles CSV in the exact schema the network reads."""
    rows = []
    for cid in sorted(profiles):
        load, pv = profiles[cid]
        b = B[cid]
        grid = load - pv - b
        soc = batt_kwh - DT * np.cumsum(b)
        for j in range(T):
            rows.append((cid, day, j + 1, load[j], pv[j], b[j], grid[j],
                         soc[j], 0.0))
    df = pd.DataFrame(rows, columns=[
        "customer", "date", "interval", "load_kw", "pv_kw", "battery_kw",
        "grid_kw", "soc_kwh", "daily_savings"])
    assert np.allclose(df["grid_kw"],
                       df["load_kw"] - df["pv_kw"] - df["battery_kw"])
    assert not df.isna().any().any(), "NaN in exported dispatch"
    df.to_csv(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(df))


# ==========================================================
# STAGE 5 -- network simulation + reporting
# ==========================================================

def injected_aggregate(lc_map, profiles):
    agg = np.zeros(T)
    for _lname, cid in lc_map.items():
        agg += profiles[cid][0]["grid"]
    return agg


def simulate(run_dir, csvs, args):
    import elermorevale_openDSS as ev
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ev.OUTPUT_DIR = fig_dir

    profiles = {s: ev.load_profiles_from_csv(p) for s, p in csvs.items()}
    ids = sorted(profiles["vpp"].keys())

    logger.info("building network to enumerate loads ...")
    ev.build_elermorevale(args.glm_dir, args.common_dir,
                          skip_generators=True)
    load_names = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads(ids, load_names)
    monitored = ev.select_monitored_loads(lc_map,
                                          n_monitors=args.n_monitors)
    scale = len(lc_map) / len(ids)
    logger.info("replication: %d loads / %d customers -> x%.2f feeder "
                "scale", len(lc_map), len(ids), scale)

    results, injected = {}, {}
    for scen in ("nobatt", "vpp"):
        logger.info("simulating scenario %r ...", scen)
        results[scen] = ev.simulate_scenario(
            args.glm_dir, args.common_dir, lc_map, monitored,
            profiles[scen], day_idx=0)
        injected[scen] = injected_aggregate(lc_map, profiles[scen])
    return results, injected, scale, len(lc_map), ev


def fig_feeder_head(results, injected, thr_feeder, day, fig_dir):
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, results["nobatt"]["tx_p_kw"], color=C_ORANGE, lw=2,
            label="no battery (measured)")
    ax.plot(hours, results["vpp"]["tx_p_kw"], color=C_BLUE, lw=2,
            label="VPP peaker fleet (measured)")
    ax.plot(hours, injected["nobatt"], color=C_ORANGE, lw=1, ls="--",
            alpha=0.6, label="no battery (injected)")
    ax.plot(hours, injected["vpp"], color=C_BLUE, lw=1, ls="--",
            alpha=0.6, label="VPP (injected)")
    ax.axhline(thr_feeder, color=MUTED, lw=1.2, ls="--")
    ax.annotate("firm-capacity threshold (feeder scale)",
                xy=(0.2, thr_feeder), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("zone substation P (kW, +import)")
    ax.set_title(f"Feeder-head relief from the VPP peaker fleet — {day}",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = os.path.join(fig_dir, "feeder_head_relief.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


def fig_voltages(results, day, fig_dir, ev):
    hours = np.arange(T) * DT
    fig, ax = plt.subplots(figsize=(10, 5))
    for scen, color, label in (("nobatt", C_ORANGE, "no battery"),
                               ("vpp", C_BLUE, "VPP peaker fleet")):
        V = np.array(list(results[scen]["voltages"].values()))
        ax.fill_between(hours, V.min(axis=0), V.max(axis=0),
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(hours, V.min(axis=0), color=color, lw=1.2, label=label)
        ax.plot(hours, V.max(axis=0), color=color, lw=1.2)
    ax.axhline(ev.V_LOWER_PU, color=C_RED, lw=1, ls="--")
    ax.annotate(f"statutory limits ({ev.V_LOWER_PU:.2f} / "
                f"{ev.V_UPPER_PU:.2f} p.u.)",
                xy=(0.2, ev.V_LOWER_PU), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.axhline(ev.V_UPPER_PU, color=C_RED, lw=1, ls="--")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("voltage (p.u.)")
    ax.set_title(f"Voltage envelope across monitored loads — {day}",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = os.path.join(fig_dir, "voltage_envelope.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay the worst peak event on the Elermore Vale "
                    "feeder with a peaker-mode battery fleet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data_3_years.csv")
    p.add_argument("--focus", type=float, default=0.70,
                   help="Firm-capacity threshold as fraction of peak")
    p.add_argument("--date", default=None,
                   help="Replay this day's worst event instead of the "
                        "global worst (ISO yyyy-mm-dd)")
    p.add_argument("--battery-kw", type=float, default=base.P_MAX)
    p.add_argument("--battery-kwh", type=float, default=base.E_MAX_DEFAULT)
    p.add_argument("--usable-frac", type=float, default=1.0)
    p.add_argument("--glm-dir", default="Elermorevale")
    p.add_argument("--common-dir", default="common")
    p.add_argument("--n-monitors", type=int, default=100)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--skip-network", action="store_true",
                   help="Stop after exporting the dispatch CSVs")
    return p.parse_args()


def main():
    args = parse_args()

    agg = build_aggregate(args.data)
    ev_row, thr, peak_kw = locate_event(agg, args.focus, args.date)
    day = ev_row["start"].date().isoformat()

    profiles = day_profiles(args.data, day)
    fleet, B, net, deficit, sizing = build_dispatch(
        profiles, thr, args.battery_kw, args.battery_kwh,
        args.usable_frac)

    run_dir = os.path.join(
        args.runs_root,
        f"peak_replay_{day}_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    csvs = {"nobatt": os.path.join(run_dir, "dispatch_nobatt.csv"),
            "vpp": os.path.join(run_dir, "dispatch_vpp.csv")}
    zeroB = {cid: np.zeros(T) for cid in B}
    export_scenario(csvs["nobatt"], profiles, zeroB, day, args.battery_kwh)
    export_scenario(csvs["vpp"], profiles, B, day, args.battery_kwh)

    manifest = {
        "kind": "peak_replay",
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "peak_kw_3yr": peak_kw,
        "threshold_kw": thr,
        "event": {k: str(ev_row[k]) for k in
                  ("start", "end", "duration_h", "max_deficit_kw",
                   "energy_kwh")},
        "sizing": sizing,
        "fleet_ids": fleet,
    }

    if args.skip_network:
        with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info("--skip-network set; dispatch exported to %s", run_dir)
        return

    results, injected, scale, n_loads, ev = simulate(run_dir, csvs, args)
    thr_feeder = thr * scale
    fig_dir = os.path.join(run_dir, "figures")
    fig_feeder_head(results, injected, thr_feeder, day, fig_dir)
    fig_voltages(results, day, fig_dir, ev)

    rows = []
    for scen, label in (("nobatt", "no battery"),
                        ("vpp", "VPP peaker fleet")):
        r = results[scen]
        tx = np.asarray(r["tx_p_kw"], float)
        rows.append({
            "scenario": label,
            "peak_tx_kw": float(tx.max()),
            "tx_over_threshold_kw": float(max(tx.max() - thr_feeder, 0.0)),
            "intervals_over_threshold": int((tx > thr_feeder).sum()),
            "v_min_pu": r["v_min_pu"],
            "v_max_pu": r["v_max_pu"],
            "n_voltage_violations": r["n_violations"],
            "loss_kw_final_step": r["loss_kw"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "replay_summary.csv"), index=False)

    shave = (np.asarray(results["nobatt"]["tx_p_kw"], float)
             - np.asarray(results["vpp"]["tx_p_kw"], float))
    manifest["network"] = {
        "n_loads": n_loads, "scale": scale,
        "threshold_feeder_kw": thr_feeder,
        "max_shave_kw": float(shave.max()),
        "summary": rows,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\n=== Worst-event replay: {day}, threshold "
          f"{args.focus:.0%} of peak ===")
    print(f"  Fleet: {sizing['n_fleet']} of {len(profiles)} households "
          f"({sizing['binding']}-limited), per-household peak "
          f"{sizing['per_hh_peak_kw']:.2f} kW / "
          f"{sizing['per_hh_energy_kwh']:.2f} kWh")
    print(f"  Feeder scale x{scale:.2f} -> threshold "
          f"{thr_feeder:.0f} kW at the zone substation")
    print(df.to_string(index=False,
                       float_format=lambda v: f"{v:.3f}"))
    print(f"  Max measured shave: {shave.max():.0f} kW")
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
