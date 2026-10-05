"""
battery_location_study.py
=========================

Does it matter WHERE the flexibility sits? The same battery dispatch is
injected into the Elermore Vale model in two ways:

  distributed  every household's battery sits behind its own meter: the
               load carries p = load - pv - b (the pipeline's default)
  aggregate    the loads carry the no-battery baseline and ONE Generator
               at the 11 kV feeder-head bus injects sum_i b_i at feeder
               scale -- the "VPP as a plant" picture
  nobatt       reference

The dispatch comes from an existing run directory (default: the latest
static_vs_doe_* run; --dispatch picks which dispatch_<case>.csv), so
both representations carry exactly the same sum_i b_i. Each day is
solved step by step so total circuit losses are recorded per interval
(elermorevale_openDSS.simulate_scenario only keeps the last interval's).

What to expect: the zone-transformer flow is nearly identical in both
representations, because both inject the same power upstream of the LV
network. Everything below the 11 kV bus -- LV line currents, losses,
customer voltages -- only changes when the batteries are distributed.
The gap between the two representations is the location value.

Artifacts land in outputs/runs/battery_location_<date>_<timestamp>/:
summary.csv, manifest.json and figures/ (transformer_and_losses,
voltage_envelope, location_value).

Usage:
    python studies/battery_location_study.py                 # latest static_vs_doe run, DOE dispatch
    python studies/battery_location_study.py --dispatch static
    python studies/battery_location_study.py --run-dir outputs/runs/centralised_qp_static_... --dispatch coupled
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GLM_COMMON, GLM_DIR, RUNS  # noqa: E402
from vpp import vpp_export as vexport  # noqa: E402
from studies.peak_duty_analysis import (  # noqa: E402
    C_BLUE, C_ORANGE, C_RED, INK_2, MUTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

T = 48
DT = 0.5
HOURS = np.arange(T) * DT
MW_TO_KW = 1000.0

SOURCE_RUN_PREFIX = "static_vs_doe_"
AGG_BUS = "BusZoneSub11kV"          # feeder-head 11 kV bus (OLTC secondary)
AGG_KV = 11.0
AGG_NAME = "VPP_agg"
AGG_SHAPE = "shape_vpp_agg"

CASES = ("nobatt", "distributed", "aggregate")
CASE_STYLE = {
    "nobatt": ("no battery", MUTED),
    "distributed": ("distributed (behind the meter)", C_BLUE),
    "aggregate": ("aggregate generator at 11 kV", C_ORANGE),
}


# ==========================================================
# INPUTS
# ==========================================================

def latest_run_dir(runs_root, prefix=SOURCE_RUN_PREFIX):
    """Most recent run directory whose name starts with prefix."""
    runs_root = Path(runs_root)
    candidates = sorted(p for p in runs_root.glob(f"{prefix}*")
                        if (p / "manifest.json").is_file())
    if not candidates:
        raise FileNotFoundError(
            f"no {prefix}* run with a manifest under {runs_root}; run "
            "studies/static_vs_doe_replay.py first or pass --run-dir")
    return candidates[-1]


def aggregate_injection(load_customer_map, profiles, day_idx):
    """
    Battery power the distributed representation injects, summed over
    the loads actually attached to the network (kW, +discharge). This is
    what the single 11 kV generator must reproduce.
    """
    inject = np.zeros(T)
    for _lname, cid in load_customer_map.items():
        days = profiles.get(cid, [])
        if day_idx < len(days):
            inject += np.asarray(days[day_idx]["battery"], dtype=float)
    return inject


# ==========================================================
# NETWORK
# ==========================================================

def add_aggregate_generator(ev, inject_kw, bus=AGG_BUS, kv=AGG_KV):
    """
    One three-phase constant-P Generator at the feeder-head bus following
    a daily loadshape: kw=1 base, so the multipliers carry signed kW
    directly (negative = the fleet charging, drawn from the grid).
    """
    inject_kw = np.asarray(inject_kw, dtype=float)
    mult = ",".join(f"{v:.6f}" for v in inject_kw)
    rating = max(float(np.abs(inject_kw).max()), 1.0)
    cmd = ev.dss.Text
    cmd.Command = (f"New Loadshape.{AGG_SHAPE} npts={T} minterval=30 "
                   f"mult=({mult})")
    cmd.Command = (f"New Generator.{AGG_NAME} bus1={bus} phases=3 kv={kv} "
                   f"kw=1 pf=1 model=1 kva={rating:.1f} daily={AGG_SHAPE}")


def run_daily_stepwise(ev):
    """
    run_daily() one interval at a time, reading total circuit losses
    after each power flow. Same solver settings and time sequence as
    run_daily(), so the monitors record identical samples.
    Returns (loss_kw, loss_kvar), each shape (T,).
    """
    cmd = ev.dss.Text
    sol = ev.dss.ActiveCircuit.Solution
    cmd.Command = "Set mode=daily stepsize=30m number=1"
    cmd.Command = f"Set controlmode={'static' if ev.OLTC_ACTIVE else 'off'}"
    cmd.Command = "Set maxcontroliter=50"
    cmd.Command = "Set maxiterations=100"
    cmd.Command = "Calcvoltagebases"
    sol.Hour = 0
    sol.Seconds = 0.0
    loss_kw, loss_kvar = np.zeros(T), np.zeros(T)
    for k in range(T):
        try:
            sol.Solve()
        except Exception as exc:
            if "485" not in str(exc):
                raise
            logger.warning("interval %d: control loop didn't settle (#485)", k)
        if not sol.Converged:
            raise RuntimeError(f"interval {k}: daily power flow did not converge")
        watts, vars_ = ev.dss.ActiveCircuit.Losses
        loss_kw[k], loss_kvar[k] = watts / 1000.0, vars_ / 1000.0
    ev.dss.ActiveCircuit.Monitors.SaveAll()
    return loss_kw, loss_kvar


def simulate_case(ev, args, lc_map, monitored, profiles, day_idx,
                  inject_kw=None):
    """One day under one representation; returns the metrics dict."""
    ev.build_elermorevale(args.glm_dir, args.common_dir, skip_generators=True,
                          oltc=ev.OLTC_ACTIVE)
    ev.add_monitors(monitored)
    date_str = ev.attach_shapes(lc_map, profiles, day_idx, series="grid")
    if inject_kw is not None:
        add_aggregate_generator(ev, inject_kw)
    loss_kw, loss_kvar = run_daily_stepwise(ev)

    voltages = ev.collect_voltages(monitored)
    ev.assert_monitors_energised(voltages)
    tx_p, tx_q = ev.collect_tx_power()
    all_v = np.array(list(voltages.values()))
    return {
        "date": date_str,
        "voltages": voltages,
        "tx_p_kw": tx_p,
        "tx_q_kvar": tx_q,
        "loss_kw": loss_kw,
        "loss_kvar": loss_kvar,
        "v_min_pu": float(all_v.min()),
        "v_max_pu": float(all_v.max()),
        "n_over": int(np.sum(all_v > ev.V_UPPER_PU)),
        "n_under": int(np.sum(all_v < ev.V_LOWER_PU)),
        "total_points": int(all_v.size),
    }


# ==========================================================
# METRICS
# ==========================================================

def summary_rows(results):
    base = results["nobatt"]
    loss0 = float(base["loss_kw"].sum() * DT)
    rows = []
    for name in CASES:
        r = results[name]
        loss_kwh = float(r["loss_kw"].sum() * DT)
        tx = np.asarray(r["tx_p_kw"], dtype=float)
        rows.append({
            "case": name,
            "label": CASE_STYLE[name][0],
            "tx_peak_kw": float(tx.max()),
            "tx_peak_hour": float(HOURS[int(np.argmax(tx))]),
            "tx_import_kwh": float(np.maximum(tx, 0).sum() * DT),
            "loss_kwh": loss_kwh,
            "loss_peak_kw": float(r["loss_kw"].max()),
            "loss_saving_kwh": loss0 - loss_kwh,
            "loss_saving_pct": (100.0 * (loss0 - loss_kwh) / loss0
                                if loss0 else 0.0),
            "v_min_pu": r["v_min_pu"],
            "v_max_pu": r["v_max_pu"],
            "n_under": r["n_under"],
            "n_over": r["n_over"],
            "n_violations": r["n_under"] + r["n_over"],
            "total_points": r["total_points"],
        })
    return rows


# ==========================================================
# FIGURES
# ==========================================================

def _save(fig, fig_dir, name):
    path = Path(fig_dir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _hour_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.grid(axis="x", visible=False)


def _label_line_ends(ax, series, x=HOURS[-1], min_gap=0.0):
    items = sorted(series, key=lambda s: s[1])
    placed = []
    for _label, y in items:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    for (label, _y), y in zip(items, placed):
        ax.annotate(label, xy=(x, y), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK_2, annotation_clip=False)


def fig_transformer_and_losses(results, date_iso, fig_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    ends1, ends2 = [], []
    for name in CASES:
        label, color = CASE_STYLE[name]
        ls = "--" if name == "aggregate" else "-"
        tx = np.asarray(results[name]["tx_p_kw"], dtype=float) / MW_TO_KW
        ax1.plot(HOURS, tx, color=color, lw=2, ls=ls, label=label)
        ends1.append((label, tx[-1]))
        ax2.plot(HOURS, results[name]["loss_kw"], color=color, lw=2, ls=ls,
                 label=label)
        ends2.append((label, results[name]["loss_kw"][-1]))
    ax1.set_ylabel("zone transformer P (MW, +import)")
    ax1.set_title(f"Same dispatch, two locations — {date_iso}",
                  loc="left", fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="x", visible=False)
    _label_line_ends(ax1, ends1, min_gap=0.25)

    ax2.set_ylabel("total circuit losses (kW)")
    ax2.legend(fontsize=9, loc="upper left")
    _hour_axis(ax2)
    span = max(r["loss_kw"].max() for r in results.values())
    _label_line_ends(ax2, ends2, min_gap=0.05 * span)
    _save(fig, fig_dir, "transformer_and_losses")


def fig_voltage_envelope(results, ev, date_iso, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for name in CASES:
        label, color = CASE_STYLE[name]
        V = np.array(list(results[name]["voltages"].values()))
        ls = "--" if name == "aggregate" else "-"
        ax.fill_between(HOURS, V.min(axis=0), V.max(axis=0),
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(HOURS, V.min(axis=0), color=color, lw=1.4, ls=ls, label=label)
        ax.plot(HOURS, V.max(axis=0), color=color, lw=1.4, ls=ls)
    ax.axhline(ev.V_LOWER_PU, color=C_RED, lw=1, ls="--")
    ax.axhline(ev.V_UPPER_PU, color=C_RED, lw=1, ls="--")
    ax.annotate(f"statutory limits ({ev.V_LOWER_PU:.2f} / "
                f"{ev.V_UPPER_PU:.2f} p.u.)",
                xy=(0.2, ev.V_LOWER_PU), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylabel("voltage across monitored loads (p.u.)")
    ax.set_title(f"Customer voltage envelope by battery location — "
                 f"{date_iso}", loc="left", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    _hour_axis(ax)
    _save(fig, fig_dir, "voltage_envelope")


def fig_location_value(rows, date_iso, fig_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [CASE_STYLE[r["case"]][0].split(" (")[0] for r in rows]
    colors = [CASE_STYLE[r["case"]][1] for r in rows]
    x = np.arange(len(rows))
    panels = ((ax1, "loss_kwh", "daily circuit losses (kWh)"),
              (ax2, "n_under", f"under-voltage points (of "
                               f"{rows[0]['total_points']})"))
    for ax, key, ylabel in panels:
        vals = [r[key] for r in rows]
        bars = ax.bar(x, vals, color=colors, width=0.6)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
    fig.suptitle(f"Location value of the same dispatch — {date_iso}",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, fig_dir, "location_value")


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Same battery dispatch as distributed behind-the-meter "
                    "batteries vs one aggregate Generator at the 11 kV "
                    "feeder-head bus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="Run directory holding dispatch_<case>.csv files "
                        f"(default: latest {SOURCE_RUN_PREFIX}* run)")
    p.add_argument("--dispatch", default="doe",
                   help="Which dispatch_<case>.csv supplies the batteries")
    p.add_argument("--baseline", default="nobatt",
                   help="Which dispatch_<case>.csv is the no-battery "
                        "reference")
    p.add_argument("--glm-dir", default=str(GLM_DIR))
    p.add_argument("--common-dir", default=str(GLM_COMMON))
    p.add_argument("--n-monitors", type=int, default=100)
    p.add_argument("--runs-root", default=str(RUNS))
    return p.parse_args()


def load_inputs(ev, args):
    """Source run, per-customer profiles for baseline and dispatch, date."""
    src = (Path(args.run_dir) if args.run_dir
           else latest_run_dir(args.runs_root))
    csvs = {"nobatt": src / f"dispatch_{args.baseline}.csv",
            "dispatch": src / f"dispatch_{args.dispatch}.csv"}
    for name, path in csvs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path} not found")
    logger.info("source run: %s (baseline=%s, dispatch=%s)", src.name,
                args.baseline, args.dispatch)

    profiles = {name: ev.load_profiles_from_csv(str(path))
                for name, path in csvs.items()}
    ids = sorted(profiles["dispatch"])
    if ids != sorted(profiles["nobatt"]):
        raise ValueError("baseline and dispatch CSVs cover different "
                         "customers")
    n_days = {len(days) for days in profiles["dispatch"].values()}
    if n_days != {1}:
        raise ValueError(f"expected single-day dispatch CSVs, found "
                         f"{n_days} days per customer")
    date_iso = str(profiles["dispatch"][ids[0]][0]["date"])[:10]
    return src, profiles, ids, date_iso


def main():
    args = parse_args()
    from network import elermorevale_openDSS as ev

    src, profiles, ids, date_iso = load_inputs(ev, args)
    day_idx = 0

    logger.info("building network to enumerate loads ...")
    ev.build_elermorevale(args.glm_dir, args.common_dir, skip_generators=True)
    load_names = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads(ids, load_names)
    monitored = ev.select_monitored_loads(lc_map, n_monitors=args.n_monitors)
    scale = len(lc_map) / len(ids)
    inject = aggregate_injection(lc_map, profiles["dispatch"], day_idx)
    logger.info("replication x%.2f; fleet injection %.0f kW peak discharge, "
                "%.0f kW peak charge, %.0f kWh discharged", scale,
                inject.max(), -inject.min(),
                np.maximum(inject, 0).sum() * DT)

    results = {}
    logger.info("simulating 'nobatt' ...")
    results["nobatt"] = simulate_case(ev, args, lc_map, monitored,
                                      profiles["nobatt"], day_idx)
    logger.info("simulating 'distributed' ...")
    results["distributed"] = simulate_case(ev, args, lc_map, monitored,
                                           profiles["dispatch"], day_idx)
    logger.info("simulating 'aggregate' (Generator.%s at %s) ...",
                AGG_NAME, AGG_BUS)
    results["aggregate"] = simulate_case(ev, args, lc_map, monitored,
                                         profiles["nobatt"], day_idx,
                                         inject_kw=inject)

    rows = summary_rows(results)
    run_dir = Path(args.runs_root) / (
        f"battery_location_{date_iso}_{datetime.now():%Y%m%d-%H%M%S}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_transformer_and_losses(results, date_iso, fig_dir)
    fig_voltage_envelope(results, ev, date_iso, fig_dir)
    fig_location_value(rows, date_iso, fig_dir)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "summary.csv", index=False)
    tx_gap = (np.asarray(results["aggregate"]["tx_p_kw"], dtype=float)
              - np.asarray(results["distributed"]["tx_p_kw"], dtype=float))
    manifest = {
        "kind": "battery_location_study",
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "source_run": src.name,
        "date": date_iso,
        "n_households": len(ids),
        "n_loads": len(lc_map),
        "scale": scale,
        "aggregate_generator": {"name": AGG_NAME, "bus": AGG_BUS,
                                "kv": AGG_KV, "inject_kw": inject},
        "series": {name: {"tx_p_kw": results[name]["tx_p_kw"],
                          "loss_kw": results[name]["loss_kw"]}
                   for name in CASES},
        "tx_gap_aggregate_minus_distributed_kw": tx_gap,
        "summary": rows,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(vexport.json_safe(manifest), indent=2, allow_nan=False),
        encoding="utf-8")

    print(f"\n=== Battery location study on {date_iso}: dispatch "
          f"'{args.dispatch}' from {src.name}, {len(ids)} households "
          f"x{scale:.1f} ===")
    cols = ["label", "tx_peak_kw", "tx_import_kwh", "loss_kwh",
            "loss_saving_kwh", "loss_saving_pct", "v_min_pu", "n_under",
            "n_over"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"  max |tx aggregate - tx distributed| = "
          f"{np.abs(tx_gap).max():.0f} kW")
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
