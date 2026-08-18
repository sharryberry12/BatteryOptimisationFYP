"""
peak_duty_analysis.py
=====================

Peak-finding and VPP duty-cycle analysis over the full Ausgrid solar-home
dataset.

Motivation (Part B reframe): a VPP's value as a *virtual peaker plant* is set
by how rarely it must run. If the feeder must never exceed a firm-capacity
threshold (e.g. 70 % of historical peak), the battery fleet only discharges
during the few intervals per year where aggregate demand exceeds that
threshold. This script quantifies that duty cycle:

  1. Build the aggregate half-hourly demand series over the whole dataset
     (default: net demand = GC + CL - GG, i.e. what the grid actually sees).
  2. Find the peak demand condition (when, how large, top-10 table).
  3. Sweep firm-capacity thresholds f x peak: hours/year above, exceedance
     events (contiguous runs), worst-event power deficit and energy.
  4. Size the battery fleet per threshold: households required, limited by
     either power (5 kW each) or energy (10 kWh each), and the resulting
     per-battery duty (hours and cycle-equivalents per year).

Sizing assumptions (deliberately peaker-plant-flavoured):
  * The fleet is pre-charged before an event: the full --battery-kwh is
    available (the QP world starts at SOC_INIT_FRAC = 0.5; use
    --usable-frac 0.5 to model "no pre-charge notice").
  * Events are sized independently -- consecutive events are far enough
    apart to recharge between them (checked and reported if violated).
  * No round-trip losses / degradation, consistent with paper_context.md §9.

Usage:
    python peak_duty_analysis.py --data data_3_years.csv --save
    python peak_duty_analysis.py --data data_3_years.csv --clean --save
    python peak_duty_analysis.py --data data1.csv --focus 0.8 --save

The first run aggregates the raw CSV (~minutes for the 3-year file) and
caches the half-hourly aggregate under outputs/cache/; later runs start in
seconds. Figures land in --output-dir (default figures/peak_duty/), along
with duty_cycle_summary.csv and events_focus.csv.
"""

import argparse
import logging
import math
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# repo root on sys.path so `paths` and `dispatch.*` import from any cwd
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CACHE as _CACHE, DATA_3Y_CSV, FIGURES  # noqa: E402
from dispatch import osqp_daily as base  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DT = base.DT           # 0.5 h per interval
T = base.T             # 48 intervals per day
HOURS_PER_YEAR = 8766.0

CACHE_DIR = str(_CACHE)

# Palette (validated reference palette, light mode)
C_BLUE = "#2a78d6"     # primary series (net demand)
C_ORANGE = "#eb6834"   # secondary series (gross load / energy-limited)
C_AQUA = "#1baf7a"     # tertiary series (PV)
C_RED = "#d03b3b"      # status-critical: exceedance above threshold
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 10,
    "legend.frameon": False,
})


# ==========================================================
# STAGE 1 -- aggregate half-hourly series (cached)
# ==========================================================

def build_aggregate(data_path, clean=False, rebuild=False):
    """
    Aggregate the per-customer long-format data into one half-hourly series:

        ts, load_kw (sum GC+CL), pv_kw (sum GG), n (customers reporting)

    ts marks the START of each half-hour interval. Sums are rescaled to a
    constant panel of n_ref households (the max simultaneously reporting)
    via the per-household mean, so customers dropping in/out of the dataset
    can't fake demand dips. Cached as CSV keyed by file stem + cleaning.
    """
    stem = os.path.splitext(os.path.basename(data_path))[0]
    cache = os.path.join(CACHE_DIR, f"peak_agg_{stem}{'_clean' if clean else ''}.csv")
    if os.path.exists(cache) and not rebuild:
        logger.info("Loading cached aggregate: %s", cache)
        agg = pd.read_csv(cache, parse_dates=["ts"])
        return agg

    df = base.load_dataset(data_path)
    if clean:
        df = base.clean_dataset(df)

    df["net_load"] = df["load"]          # GC + CL, set in load_dataset
    grp = df.groupby(["date_parsed", "interval"], sort=True)
    agg = grp.agg(
        load_kw=("net_load", "sum"),
        pv_kw=("pv", "sum"),
        n=("Customer", "nunique"),
    ).reset_index()

    agg["ts"] = agg["date_parsed"] + pd.to_timedelta((agg["interval"] - 1) * 30, unit="m")
    n_ref = int(agg["n"].max())
    scale = n_ref / agg["n"]
    agg["load_kw"] *= scale
    agg["pv_kw"] *= scale
    agg["n_ref"] = n_ref
    agg = agg[["ts", "load_kw", "pv_kw", "n", "n_ref"]].sort_values("ts").reset_index(drop=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    agg.to_csv(cache, index=False)
    logger.info("Cached aggregate series -> %s (%d intervals, n_ref=%d, "
                "reporting range %d..%d)", cache, len(agg), n_ref,
                int(agg["n"].min()), int(agg["n"].max()))
    return agg


# ==========================================================
# STAGE 2 -- peak identification
# ==========================================================

def demand_series(agg, gross=False):
    """Demand the grid must serve (kW, positive = consumption)."""
    if gross:
        return agg["load_kw"].to_numpy()
    return (agg["load_kw"] - agg["pv_kw"]).to_numpy()


def peak_table(agg, demand, top=10):
    """Top-N distinct-day peak intervals."""
    df = agg.assign(demand_kw=demand, day=agg["ts"].dt.date)
    df = df.sort_values("demand_kw", ascending=False)
    df = df.drop_duplicates(subset="day").head(top)
    n_ref = int(agg["n_ref"].iloc[0])
    out = pd.DataFrame({
        "when": df["ts"].dt.strftime("%Y-%m-%d %H:%M"),
        "weekday": df["ts"].dt.strftime("%a"),
        "agg_kw": df["demand_kw"].round(1),
        "per_household_kw": (df["demand_kw"] / n_ref).round(3),
    })
    return out.reset_index(drop=True)


# ==========================================================
# STAGE 3 -- exceedance events + duty-cycle sweep
# ==========================================================

def find_events(ts, demand, thr):
    """
    Contiguous runs of demand > thr. A gap in the timestamps (dropped DST
    days etc.) breaks contiguity even if both sides exceed the threshold.
    Returns a DataFrame, one row per event.
    """
    above = demand > thr
    step = pd.Timedelta(minutes=30)
    events = []
    i, n = 0, len(demand)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1] and (ts[j + 1] - ts[j]) == step:
            j += 1
        seg = demand[i:j + 1]
        events.append({
            "start": ts[i],
            "end": ts[j] + step,
            "duration_h": (j - i + 1) * DT,
            "peak_kw": seg.max(),
            "max_deficit_kw": (seg - thr).max(),
            "energy_kwh": (seg - thr).sum() * DT,
        })
        i = j + 1
    return pd.DataFrame(events)


def sweep_thresholds(ts, demand, peak_kw, fractions, years,
                     batt_kw, batt_kwh, usable_frac, n_ref):
    """One row of duty-cycle + fleet-sizing metrics per threshold fraction."""
    e_usable = batt_kwh * usable_frac
    rows = []
    for f in fractions:
        thr = f * peak_kw
        ev = find_events(ts, demand, thr)
        if ev.empty:
            rows.append({
                "fraction": f, "threshold_kw": thr, "hours_per_year": 0.0,
                "events_per_year": 0.0, "n_events": 0, "max_event_h": 0.0,
                "max_deficit_kw": 0.0, "max_event_kwh": 0.0,
                "energy_per_year_kwh": 0.0, "n_households_power": 0,
                "n_households_energy": 0, "n_households": 0, "binding": "-",
                "cycles_per_year": 0.0,
            })
            continue
        n_power = math.ceil(ev["max_deficit_kw"].max() / batt_kw)
        n_energy = math.ceil(ev["energy_kwh"].max() / e_usable)
        n_req = max(n_power, n_energy)
        energy_yr = ev["energy_kwh"].sum() / years
        rows.append({
            "fraction": f,
            "threshold_kw": thr,
            "hours_per_year": ev["duration_h"].sum() / years,
            "events_per_year": len(ev) / years,
            "n_events": len(ev),
            "max_event_h": ev["duration_h"].max(),
            "max_deficit_kw": ev["max_deficit_kw"].max(),
            "max_event_kwh": ev["energy_kwh"].max(),
            "energy_per_year_kwh": energy_yr,
            "n_households_power": n_power,
            "n_households_energy": n_energy,
            "n_households": n_req,
            "binding": "power" if n_power >= n_energy else "energy",
            "cycles_per_year": energy_yr / (n_req * e_usable) if n_req else 0.0,
        })
    out = pd.DataFrame(rows)
    out["pct_of_panel"] = 100.0 * out["n_households"] / n_ref
    return out


def check_recharge_gaps(events, batt_kwh, usable_frac, batt_kw):
    """
    The per-event sizing assumes full recharge between events. Flag pairs
    of events closer than the time needed to recharge the worst-event
    energy at full power (a conservative bound: charging is also limited
    to batt_kw per household).
    """
    if len(events) < 2:
        return 0
    recharge_h = (usable_frac * batt_kwh) / batt_kw
    gaps = (events["start"].iloc[1:].to_numpy()
            - events["end"].iloc[:-1].to_numpy()) / np.timedelta64(1, "h")
    return int((gaps < recharge_h).sum())


# ==========================================================
# STAGE 4 -- figures
# ==========================================================

def fig_duration_curve(demand, peak_kw, fractions_marked, years, outdir, save):
    """Load-duration curve, full + top-of-curve zoom on a log time axis."""
    srt = np.sort(demand)[::-1]
    hours_at_or_above = np.arange(1, len(srt) + 1) * DT / years

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle("Aggregate demand duration curve (dataset panel, scaled)",
                 x=0.02, ha="left", fontweight="bold")

    ax1.plot(100.0 * np.arange(1, len(srt) + 1) / len(srt), srt,
             color=C_BLUE, lw=2)
    ax1.set_xlabel("% of time demand is at or above")
    ax1.set_ylabel("aggregate demand (kW)")
    ax1.set_title("full curve", loc="left", fontsize=10, color=INK_2)

    top = srt[srt >= 0.5 * peak_kw]
    ax2.plot(hours_at_or_above[:len(top)], top, color=C_BLUE, lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("hours per year at or above (log)")
    ax2.set_title("top of curve, thresholds marked", loc="left",
                  fontsize=10, color=INK_2)
    for f in fractions_marked:
        thr = f * peak_kw
        ax2.axhline(thr, color=MUTED, lw=1, ls="--")
        ax2.annotate(f"{f:.0%} of peak", xy=(ax2.get_xlim()[1], thr),
                     xytext=(-4, 3), textcoords="offset points",
                     ha="right", fontsize=8, color=INK_2)
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save:
        p = os.path.join(outdir, "duration_curve.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", p)
    return fig


def fig_peak_day(agg, demand, focus_thr, outdir, save):
    """Gross load / net demand / PV on the day of the global demand peak."""
    i_peak = int(np.argmax(demand))
    day = agg["ts"].dt.date.iloc[i_peak]
    m = (agg["ts"].dt.date == day).to_numpy()
    ts = agg.loc[m, "ts"]
    hours = ts.dt.hour + ts.dt.minute / 60.0
    gross = agg.loc[m, "load_kw"].to_numpy()
    pv = agg.loc[m, "pv_kw"].to_numpy()
    net = demand[m]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(hours, gross, color=C_ORANGE, lw=2, label="gross load (GC+CL)")
    ax.plot(hours, net, color=C_BLUE, lw=2, label="net demand (load − PV)")
    ax.plot(hours, pv, color=C_AQUA, lw=2, label="PV generation")
    ax.axhline(focus_thr, color=MUTED, lw=1, ls="--")
    ax.annotate("firm-capacity threshold", xy=(0.2, focus_thr),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK_2)
    ax.fill_between(hours, net, focus_thr, where=net > focus_thr,
                    color=C_RED, alpha=0.25, linewidth=0,
                    label="VPP must cover")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("kW (aggregate)")
    ax.set_title(f"Peak demand day — {day}", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save:
        p = os.path.join(outdir, "peak_day.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", p)
    return fig


def fig_duty_sweep(sweep, focus, outdir, save):
    """2x2 small multiples over the threshold fraction."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    fig.suptitle("VPP duty cycle vs firm-capacity threshold",
                 x=0.02, ha="left", fontweight="bold")
    x = 100.0 * sweep["fraction"]

    ax = axes[0, 0]
    ax.plot(x, sweep["hours_per_year"], color=C_BLUE, lw=2)
    ax.set_yscale("log")
    ax.set_ylabel("hours per year above threshold")

    ax = axes[0, 1]
    ax.plot(x, sweep["events_per_year"], color=C_BLUE, lw=2)
    ax.set_ylabel("events per year")

    ax = axes[1, 0]
    ax.plot(x, sweep["max_event_h"], color=C_BLUE, lw=2)
    ax.set_ylabel("longest single event (h)")
    ax.set_xlabel("threshold (% of peak)")

    ax = axes[1, 1]
    ax.plot(x, sweep["n_households_power"], color=C_BLUE, lw=2,
            label="power-limited (5 kW each)")
    ax.plot(x, sweep["n_households_energy"], color=C_ORANGE, lw=2,
            label="energy-limited (10 kWh each)")
    ax.set_ylabel("households required")
    ax.set_xlabel("threshold (% of peak)")
    ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.axvline(100.0 * focus, color=MUTED, lw=1, ls=":")
        ax.grid(axis="x", visible=False)
    axes[0, 0].annotate(f"focus {focus:.0%}", xy=(100.0 * focus, 1),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points",
                        fontsize=8, color=INK_2)

    fig.tight_layout()
    if save:
        p = os.path.join(outdir, "duty_cycle_sweep.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", p)
    return fig


def fig_event_calendar(agg, demand, thr, focus, outdir, save):
    """Every half-hour above the focus threshold: date vs time of day."""
    mask = demand > thr
    ts = agg.loc[mask, "ts"]
    depth = demand[mask] - thr

    fig, ax = plt.subplots(figsize=(10, 4.2))
    size = 20 + 180 * depth / max(depth.max(), 1e-9)
    ax.scatter(ts.dt.to_pydatetime(), ts.dt.hour + ts.dt.minute / 60.0,
               s=size, color=C_BLUE, alpha=0.6, edgecolors="none")
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 6))
    ax.set_ylabel("time of day")
    ax.set_title(f"When the VPP must run — intervals above {focus:.0%} of peak "
                 f"(marker size = exceedance depth)",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save:
        p = os.path.join(outdir, "event_calendar.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", p)
    return fig


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=str(DATA_3Y_CSV),
                   help="Ausgrid solar-home CSV")
    p.add_argument("--clean", action="store_true",
                   help="Apply the Ratnam et al. 2017 cleaning rules first")
    p.add_argument("--gross", action="store_true",
                   help="Analyse gross load (GC+CL) instead of net (load - PV)")
    p.add_argument("--fractions", type=float, nargs="+",
                   default=list(np.round(np.arange(0.50, 0.951, 0.025), 3)),
                   help="Threshold fractions of peak to sweep")
    p.add_argument("--focus", type=float, default=0.70,
                   help="Focus threshold fraction for events CSV + figures")
    p.add_argument("--battery-kw", type=float, default=base.P_MAX,
                   help="Discharge limit per household battery (kW)")
    p.add_argument("--battery-kwh", type=float, default=base.E_MAX_DEFAULT,
                   help="Capacity per household battery (kWh)")
    p.add_argument("--usable-frac", type=float, default=1.0,
                   help="Fraction of capacity available at event start "
                        "(1.0 = pre-charged peaker; 0.5 = QP resting SOC)")
    p.add_argument("--save", action="store_true", help="Write figures + CSVs")
    p.add_argument("--output-dir", default=str(FIGURES / "peak_duty"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() (e.g. headless run)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    agg = build_aggregate(args.data, clean=args.clean, rebuild=args.rebuild_cache)
    demand = demand_series(agg, gross=args.gross)
    ts = agg["ts"].reset_index(drop=True)
    n_ref = int(agg["n_ref"].iloc[0])
    years = (ts.iloc[-1] - ts.iloc[0]).days / 365.25
    kind = "gross load" if args.gross else "net demand"

    # ---- peak ------------------------------------------------------------
    peak_kw = float(demand.max())
    i_peak = int(np.argmax(demand))
    logger.info("Series: %s | %.2f years | %d households (scaled panel)",
                kind, years, n_ref)
    logger.info("PEAK: %.1f kW aggregate (%.2f kW/household) at %s",
                peak_kw, peak_kw / n_ref, ts.iloc[i_peak])
    top10 = peak_table(agg, demand)
    print("\nTop-10 peak days (distinct days, aggregate %s):" % kind)
    print(top10.to_string(index=False))

    # ---- sweep -----------------------------------------------------------
    fractions = sorted(set(args.fractions) | {args.focus})
    sweep = sweep_thresholds(ts, demand, peak_kw, fractions, years,
                             args.battery_kw, args.battery_kwh,
                             args.usable_frac, n_ref)
    cols = ["fraction", "threshold_kw", "hours_per_year", "events_per_year",
            "max_event_h", "max_deficit_kw", "max_event_kwh",
            "n_households", "binding", "pct_of_panel", "cycles_per_year"]
    print("\nDuty-cycle sweep (thresholds as fraction of peak):")
    print(sweep[cols].round(2).to_string(index=False))

    # ---- focus threshold -------------------------------------------------
    thr = args.focus * peak_kw
    events = find_events(ts, demand, thr)
    row = sweep.loc[np.isclose(sweep["fraction"], args.focus)].iloc[0]
    tight_gaps = check_recharge_gaps(events, args.battery_kwh,
                                     args.usable_frac, args.battery_kw)
    print(f"\n=== Focus threshold: {args.focus:.0%} of peak "
          f"= {thr:.0f} kW aggregate ===")
    print(f"  VPP must run        : {row['hours_per_year']:.1f} h/year "
          f"in {row['events_per_year']:.1f} events/year "
          f"({100 * row['hours_per_year'] / HOURS_PER_YEAR:.3f} % of the year)")
    print(f"  Worst event         : {row['max_event_h']:.1f} h, "
          f"{row['max_deficit_kw']:.1f} kW max deficit, "
          f"{row['max_event_kwh']:.1f} kWh")
    print(f"  Fleet required      : {int(row['n_households'])} households "
          f"({row['pct_of_panel']:.1f} % of the {n_ref}-household panel), "
          f"{row['binding']}-limited")
    print(f"  Per-battery duty    : {row['cycles_per_year']:.2f} "
          f"full-cycle-equivalents/year")
    if tight_gaps:
        print(f"  WARNING: {tight_gaps} event pair(s) closer than the "
              f"full-recharge time — per-event sizing is optimistic there")

    if args.save:
        sweep.to_csv(os.path.join(args.output_dir, "duty_cycle_summary.csv"),
                     index=False)
        events.to_csv(os.path.join(args.output_dir, "events_focus.csv"),
                      index=False)
        top10.to_csv(os.path.join(args.output_dir, "peak_top10.csv"),
                     index=False)
        logger.info("Saved summary CSVs -> %s", args.output_dir)

    # ---- figures ---------------------------------------------------------
    fig_duration_curve(demand, peak_kw, [0.7, 0.8, 0.9], years,
                       args.output_dir, args.save)
    fig_peak_day(agg, demand, thr, args.output_dir, args.save)
    fig_duty_sweep(sweep, args.focus, args.output_dir, args.save)
    fig_event_calendar(agg, demand, thr, args.focus,
                       args.output_dir, args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
