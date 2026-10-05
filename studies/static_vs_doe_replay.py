"""
static_vs_doe_replay.py
=======================

Replay one day (default Saturday 2011-02-05, the February 2011 heatwave
peak) through the Elermore Vale OpenDSS model under three fleet regimes
and compare the feeder power profile  P_k = sum_i p_ik  and the flow
through the 132/11 kV zone transformer:

  nobatt   no batteries: p = load - pv
  static   today's practice: the battery fleet optimises bills under a
           fixed export limit (1.5 kW/household, SAPN-style) and no
           import limit. Nothing coordinates the import side, so the
           tariff herds every battery into charging at 22:00.
  doe      the same fleet coupled by a dynamic operating envelope: a
           time-varying feeder import limit derived from the zone
           substation's measured headroom on that day (Method A, soft).

The DOE comes from real data. data/Jesmond-132_11kV-FY2011.csv holds
Ausgrid's 15-min MW loading of the Jesmond 132/11 kV zone substation
that feeds Elermore Vale. Treating the household ensemble as one slice
of that load and everything else as inflexible,

    other_k    = L_jesmond,k - P_base,k
    headroom_k = C_zone - other_k
    D_max,k    = min(headroom_k, C_feeder)

C_zone is the level the DNSP wants to hold the substation at (default
95 % of the day's measured peak); C_feeder is the feeder's own planning
limit (default: the no-battery peak, so the fleet may never create a new
feeder peak). Inside the substation's stress window the DOE drops below
the baseline (the fleet must shed); outside it relaxes to C_feeder.

The export side is the same flat cap in both battery regimes, so the two
differ only through the import-side DOE.

Both battery regimes are solved with vpp_common.solve_centralised in
soft mode: an envelope the 5 kW / 10 kWh fleet cannot meet shows up as
import shortfall in the summary, not as an error.

Artifacts land in outputs/runs/static_vs_doe_<date>_<timestamp>/:
dispatch_<case>.csv (network schema), summary.csv, manifest.json and
figures/ (doe_derivation, feeder_profile, zone_transformer_power). With
the network stage, voltages_<case>.npy (monitored loads x T, p.u., rows
in voltage_monitors.txt order) is saved per case for the paper's
voltage-envelope figure. --two-stage-rules maxmin,equal adds Method B
cases (two_stage_<rule>: the same envelope pre-allocated into soft
per-household slices, each household solving alone) to every artifact.

Usage:
    python studies/static_vs_doe_replay.py                     # 2011-02-05
    python studies/static_vs_doe_replay.py --zone-limit-frac 0.9
    python studies/static_vs_doe_replay.py --skip-network      # no OpenDSS
    python studies/static_vs_doe_replay.py --two-stage-rules maxmin  # + Method B
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_CSV, GLM_COMMON, GLM_DIR, JESMOND_CSV, RUNS  # noqa: E402
from vpp import vpp_common as vc  # noqa: E402
from vpp import vpp_export as vexport  # noqa: E402
from vpp.two_stage_doe_allocation import two_stage_doe_allocation as ts  # noqa: E402
from studies.peak_duty_analysis import (  # noqa: E402
    C_BLUE, C_ORANGE, C_RED, INK, INK_2, MUTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

T = vc.T
DT = vc.DT
HOURS = np.arange(T) * DT              # interval start, 0.0 .. 23.5

DEFAULT_DATE = "2011-02-05"
JESMOND_DATE_FORMAT = "%d%b%Y"         # "05FEB2011"
N_QUARTER_HOURS = 96
# value columns are labelled by interval END: "0:15", "0:30", ... "24:00"
QUARTER_HOUR_LABELS = tuple(f"{(i + 1) // 4}:{15 * ((i + 1) % 4):02d}"
                            for i in range(N_QUARTER_HOURS))
MW_TO_KW = 1000.0

CASES = ("nobatt", "static", "doe")        # always solved, in this order
TWO_STAGE_PREFIX = "two_stage_"            # optional Method B cases
C_PURPLE = "#a23fa6"                       # two-stage series (paper convention)
CASE_STYLE = {
    "nobatt": ("no battery", MUTED),
    "static": ("static limits", C_ORANGE),
    "doe": ("DOE (zone-sub headroom)", C_BLUE),
    "two_stage_maxmin": ("two-stage DOE (max-min slices)", C_PURPLE),
    "two_stage_equal": ("two-stage DOE (equal slices)", "#c98bcb"),
}
SOLVED_STATUSES = ("solved", "solved inaccurate")


# ==========================================================
# STAGE 1 -- zone substation measurement
# ==========================================================

def load_jesmond_day(csv_path, date_iso):
    """
    Half-hourly zone-substation loading (kW, length T) for one day.

    The Ausgrid zone-substation file has one row per day and 96
    quarter-hour MW columns labelled by interval END ("0:15" .. "24:00").
    Quarter hours are averaged in pairs so interval k covers
    [k/2, (k+1)/2) h -- the same convention as the customer data.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Zone substation file not found: {csv_path}. It is local-only "
            "(data/ is gitignored) -- see data/README.md.")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Date", "unit"}.issubset(df.columns):
        raise ValueError(f"{csv_path.name}: expected 'Date' and 'unit' columns")
    missing = [c for c in QUARTER_HOUR_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing quarter-hour columns "
                         f"{missing[:4]}{'...' if len(missing) > 4 else ''} "
                         f"(expected '0:15' .. '24:00')")

    dates = pd.to_datetime(df["Date"].astype(str).str.strip().str.upper(),
                           format=JESMOND_DATE_FORMAT)
    # select BY LABEL in chronological order, whatever the file's column order
    values = df[list(QUARTER_HOUR_LABELS)].apply(pd.to_numeric,
                                                 errors="coerce")
    sel = np.flatnonzero(dates == pd.Timestamp(date_iso))
    if len(sel) == 0:
        raise ValueError(f"{date_iso} is not a row of {csv_path.name}")
    row = sel[0]
    unit = str(df.loc[row, "unit"]).strip().upper()
    if unit != "MW":
        raise ValueError(f"{csv_path.name}: unit {unit!r} on {date_iso}, "
                         "expected MW")
    q = values.iloc[row].to_numpy(dtype=float)
    n_missing = int(np.sum(~np.isfinite(q)))
    if n_missing:
        populated = dates[values.notna().all(axis=1)]
        span = (f"{populated.min().date()} -> {populated.max().date()}"
                if len(populated) else "none")
        raise ValueError(f"{date_iso}: {n_missing} of {N_QUARTER_HOURS} "
                         f"readings missing in {csv_path.name} (fully "
                         f"populated days: {span})")
    return q.reshape(T, 2).mean(axis=1) * MW_TO_KW


def zone_headroom_envelope(jes_kw, p_base_kw, zone_limit_kw, feeder_cap_kw):
    """
    Feeder import limit D_max,k (kW) from the substation's measured
    headroom. The ensemble is one slice of the substation load and the
    rest is inflexible:

        other_k    = jes_k - p_base_k
        headroom_k = zone_limit - other_k
        D_max,k    = min(headroom_k, feeder_cap)

    Below the baseline where the substation exceeds its limit (the fleet
    must shed), above it -- up to the feeder's own cap -- where the
    substation has spare capacity (the fleet may charge).
    """
    jes = np.asarray(jes_kw, dtype=float)
    base = np.asarray(p_base_kw, dtype=float)
    if jes.shape != (T,) or base.shape != (T,):
        raise ValueError(f"expected two length-{T} series, got "
                         f"{jes.shape} and {base.shape}")
    if np.any(jes < base):
        k = int(np.argmax(base - jes))
        raise ValueError(
            "ensemble baseline exceeds the measured substation load "
            f"(by {base[k] - jes[k]:.0f} kW at {k * DT:.1f} h) -- the "
            "ensemble cannot be a slice of it; check scale and time base")
    return np.minimum(zone_limit_kw - (jes - base), feeder_cap_kw)


# ==========================================================
# STAGE 2 -- fleet dispatch under each regime
# ==========================================================

def solve_cases(households, export_limit_kw, d_max_doe, penalty=1e3,
                two_stage_rules=()):
    """
    {case: {B, d_min, d_max, result}} for the three regimes. The export
    side (flat -export_limit_kw x N) is shared; only D_max differs.

    two_stage_rules adds one Method B case per allocation rule
    ("two_stage_<rule>"): the same envelope pre-allocated into soft
    per-household slices, each household solving alone
    (two_stage_doe_allocation.run_rule, as in doe_day_sweep). Its
    aggregate out-of-slice power is stored like the centralised slacks.
    """
    n = len(households)
    d_min = -export_limit_kw * n * np.ones(T)
    unbounded = np.inf * np.ones(T)
    cases = {"nobatt": dict(B=np.zeros((n, T)), d_min=d_min,
                            d_max=unbounded, result=None)}
    for name, d_max in (("static", unbounded),
                        ("doe", np.asarray(d_max_doe, dtype=float))):
        res = vc.solve_centralised(households, d_min, d_max,
                                   soft=True, penalty=penalty)
        if res.status not in SOLVED_STATUSES:
            raise RuntimeError(f"{name}: centralised solve failed "
                               f"({res.status})")
        logger.info("%s: solved in %.2f s (%s)", name, res.solve_time,
                    res.status)
        cases[name] = dict(B=res.B, d_min=d_min, d_max=d_max, result=res)
    d_max_doe = np.asarray(d_max_doe, dtype=float)
    for rule in two_stage_rules:
        B, curtail_kw, shortfall_kw, n_failed = ts.run_rule(
            rule, households, d_min, d_max_doe, soft=True)
        logger.info("two_stage_%s: %d household solves, %d failed", rule,
                    n, n_failed)
        cases[TWO_STAGE_PREFIX + rule] = dict(
            B=B, d_min=d_min, d_max=d_max_doe, n_failed=n_failed,
            result=SimpleNamespace(slack_up=shortfall_kw,
                                   slack_lo=curtail_kw))
    return cases


def case_metrics(name, households, case, tariff, mode, jes_kw, p_base,
                 zone_limit_kw):
    """Ensemble-scale bookkeeping for one regime (pre power flow)."""
    agg = vc.aggregate_pi(households, case["B"])
    res = case["result"]
    # soft mode returns one slack per envelope side; OSQP satisfies s >= 0
    # only to tolerance, so clip the noise
    shortfall = (np.maximum(res.slack_up, 0.0) if res is not None
                 else np.zeros(T))                       # import cap breached
    export_excess = (np.maximum(res.slack_lo, 0.0) if res is not None
                     else np.zeros(T))                   # export cap breached
    zone = jes_kw - p_base + agg            # what the substation would see
    zone_excess = np.maximum(zone - zone_limit_kw, 0.0)
    savings = vc.savings_vector(households, case["B"], tariff, mode)
    k_peak = int(np.argmax(agg))
    return {
        "case": name,
        "label": CASE_STYLE.get(name, (name,))[0],   # sweeps add cases
        "peak_kw": float(agg.max()),
        "peak_hour": float(HOURS[k_peak]),
        "min_kw": float(agg.min()),
        "mean_kw": float(agg.mean()),
        "std_kw": float(agg.std()),
        "max_ramp_kw": float(np.abs(np.diff(agg)).max()),
        "zone_peak_kw": float(zone.max()),
        "zone_exceed_kwh": float(zone_excess.sum() * DT),
        "zone_exceed_intervals": int((zone_excess > 1e-6).sum()),
        "import_shortfall_kwh": float(np.sum(shortfall) * DT),
        "export_excess_kwh": float(np.sum(export_excess) * DT),
        "savings_per_day": float(savings.sum()),
        "objective": vc.objective_surrogate(households, case["B"]),
        "n_failed": int(case.get("n_failed", 0)),
    }


def export_cases(run_dir, households, cases, date_iso, tariff, mode):
    """One network-schema CSV per regime; returns {case: path}."""
    paths = {}
    for name in cases:
        df = vexport.profile_frame(households, cases[name]["B"],
                                   date_iso, tariff, mode)
        path = Path(run_dir) / f"dispatch_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = str(path)
        logger.info("wrote %s (%d rows)", path.name, len(df))
    return paths


# ==========================================================
# STAGE 3 -- network simulation
# ==========================================================

def save_voltages(run_dir, name, result, monitored):
    """
    voltages_<case>.npy: per-monitor p.u. voltage, shape (n_monitors, T),
    rows in `monitored` order (listed in voltage_monitors.txt). Data hook
    for the paper's customer voltage-envelope figure.
    """
    V = np.array([np.asarray(result["voltages"][m], dtype=float)
                  for m in monitored])
    path = Path(run_dir) / f"voltages_{name}.npy"
    np.save(path, V)
    logger.info("wrote %s %s", path.name, V.shape)
    return path


def simulate(run_dir, csvs, args):
    from network import elermorevale_openDSS as ev
    fig_dir = Path(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev.OUTPUT_DIR = str(fig_dir)

    profiles = {c: ev.load_profiles_from_csv(p) for c, p in csvs.items()}
    ids = sorted(profiles["doe"].keys())

    logger.info("building network to enumerate loads ...")
    ev.build_elermorevale(args.glm_dir, args.common_dir,
                          skip_generators=True)
    load_names = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads(ids, load_names)
    monitored = ev.select_monitored_loads(lc_map,
                                          n_monitors=args.n_monitors)
    scale = len(lc_map) / len(ids)
    logger.info("replication: %d loads / %d households -> x%.2f feeder "
                "scale", len(lc_map), len(ids), scale)

    results = {}
    monitored = list(monitored)
    for name in csvs:
        logger.info("simulating %r ...", name)
        results[name] = ev.simulate_scenario(
            args.glm_dir, args.common_dir, lc_map, monitored,
            profiles[name], day_idx=0)
        save_voltages(run_dir, name, results[name], monitored)
    (Path(run_dir) / "voltage_monitors.txt").write_text(
        "\n".join(monitored) + "\n", encoding="utf-8")
    return results, scale, len(lc_map)


def network_rows(results):
    rows = []
    for name in results:
        r = results[name]
        tx = np.asarray(r["tx_p_kw"], dtype=float)
        rows.append({
            "case": name,
            "tx_peak_kw": float(tx.max()),
            "tx_peak_hour": float(HOURS[int(np.argmax(tx))]),
            "v_min_pu": float(r["v_min_pu"]),
            "v_max_pu": float(r["v_max_pu"]),
            "n_voltage_violations": int(r["n_violations"]),
            "n_under": int(r["n_under"]),
            "n_over": int(r["n_over"]),
            "total_points": int(r["total_points"]),
        })
    return rows


# ==========================================================
# STAGE 4 -- figures
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
    """Direct-label line ends at x; labels are nudged apart vertically."""
    items = sorted(series, key=lambda s: s[1])          # (label, y_end)
    placed = []
    for _label, y in items:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    for (label, _y), y in zip(items, placed):
        ax.annotate(label, xy=(x, y), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK_2, annotation_clip=False)


def _shade_stress(ax, mask):
    """Shade the intervals where the substation exceeds its limit."""
    ax.fill_between(HOURS, 0, 1, where=mask, step="post",
                    transform=ax.get_xaxis_transform(),
                    color=C_RED, alpha=0.07, linewidth=0)


def fig_doe_derivation(jes_kw, p_base, zone_limit_kw, feeder_cap_kw,
                       d_max, date_iso, n_hh, fig_dir):
    stress = jes_kw > zone_limit_kw
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    ax1.plot(HOURS, jes_kw / MW_TO_KW, color=INK_2, lw=2,
             label="Jesmond 132/11 kV measured (all feeders)")
    ax1.axhline(zone_limit_kw / MW_TO_KW, color=C_RED, lw=1.2, ls="--")
    ax1.fill_between(HOURS, zone_limit_kw / MW_TO_KW, jes_kw / MW_TO_KW,
                     where=stress, color=C_RED, alpha=0.18, linewidth=0,
                     label="excess the fleet must absorb")
    ax1.annotate(f"zone limit C = {zone_limit_kw / MW_TO_KW:.2f} MW",
                 xy=(0.2, zone_limit_kw / MW_TO_KW), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=INK_2)
    ax1.set_ylabel("zone substation P (MW)")
    ax1.set_title(f"Where the DOE comes from — {date_iso}",
                  loc="left", fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="x", visible=False)

    _shade_stress(ax2, stress)
    ax2.plot(HOURS, p_base, color=MUTED, lw=2,
             label="ensemble no-battery import (baseline)")
    ax2.step(HOURS, d_max, where="post", color=C_BLUE, lw=2,
             label="DOE import limit D_max,k")
    ax2.axhline(feeder_cap_kw, color=MUTED, lw=1, ls=":")
    ax2.annotate(f"feeder cap {feeder_cap_kw:.0f} kW",
                 xy=(0.2, feeder_cap_kw), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=INK_2)
    ax2.set_ylabel(f"ensemble import limit (kW, N={n_hh})")
    ax2.legend(fontsize=9, loc="center left")
    _hour_axis(ax2)
    _save(fig, fig_dir, "doe_derivation")


def fig_feeder_profile(aggs, d_max, stress, metrics, date_iso, n_hh,
                       fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _shade_stress(ax, stress)
    ends = []
    for name in aggs:
        label, color = CASE_STYLE.get(name, (name, INK_2))
        ax.plot(HOURS, aggs[name], color=color, lw=2, label=label)
        ends.append((label, aggs[name][-1]))
    ax.step(HOURS, d_max, where="post", color=INK, lw=1, alpha=0.6,
            label="DOE import limit")
    ax.axhline(0, color=MUTED, lw=0.6)

    m = {row["case"]: row for row in metrics}
    for name, offset in (("static", (-70, 10)), ("doe", (-150, 14))):
        row = m[name]
        ax.annotate(f"{row['label'].split(' (')[0]} peak "
                    f"{row['peak_kw']:.0f} kW at {row['peak_hour']:04.1f} h",
                    xy=(row["peak_hour"], row["peak_kw"]), xytext=offset,
                    textcoords="offset points", fontsize=8, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    span = float(max(a.max() for a in aggs.values())
                 - min(a.min() for a in aggs.values()))
    _label_line_ends(ax, ends, min_gap=0.04 * span)

    ax.set_ylabel("aggregate grid power  Σ p_ik  (kW, +import)")
    ax.set_title(f"Feeder power profile by regime — {date_iso}, "
                 f"N={n_hh} households", loc="left", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    _hour_axis(ax)
    _save(fig, fig_dir, "feeder_profile")


def fig_zone_transformer(results, jes_kw, zone_limit_kw, stress, scale,
                         n_loads, n_hh, date_iso, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _shade_stress(ax, stress)
    ends = []
    for name in results:
        label, color = CASE_STYLE.get(name, (name, INK_2))
        tx = np.asarray(results[name]["tx_p_kw"], dtype=float) / MW_TO_KW
        ax.plot(HOURS, tx, color=color, lw=2, label=f"model: {label}")
        ends.append((label, tx[-1]))
    ax.plot(HOURS, jes_kw / MW_TO_KW, color=INK_2, lw=1.6, ls="--",
            label="measured: Jesmond 132/11 kV (all feeders)")
    ends.append(("Jesmond measured", jes_kw[-1] / MW_TO_KW))
    ax.axhline(zone_limit_kw / MW_TO_KW, color=C_RED, lw=1, ls="--")
    ax.annotate("zone limit C", xy=(0.2, zone_limit_kw / MW_TO_KW),
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                color=INK_2)
    _label_line_ends(ax, ends, min_gap=0.25)

    ax.set_ylabel("zone transformer P (MW, +import)")
    ax.set_title(f"Power through the 132/11 kV zone transformer — "
                 f"{date_iso}", loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.012, f"model = {n_hh} household profiles replicated over "
            f"{n_loads} feeder loads (×{scale:.1f}); the measurement is "
            "the whole substation, so compare shapes, not levels",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
    ax.legend(fontsize=9, loc="upper left")
    _hour_axis(ax)
    _save(fig, fig_dir, "zone_transformer_power")


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Feeder power profile and zone-transformer flow: no "
                    "battery vs static limits vs a DOE derived from the "
                    "Jesmond zone substation's measured headroom",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=str(DATA_CSV),
                   help="Ausgrid customer CSV")
    p.add_argument("--jesmond", default=str(JESMOND_CSV),
                   help="Ausgrid zone-substation CSV (15-min MW)")
    p.add_argument("--date", default=DEFAULT_DATE, help="ISO day")
    p.add_argument("--n-households", type=int, default=152)
    p.add_argument("--mode", choices=["fit", "net"], default="fit")
    p.add_argument("--e-max", type=float, default=vc.E_MAX_DEFAULT,
                   help="Battery capacity per household (kWh)")
    p.add_argument("--export-limit", type=float, default=1.5,
                   help="Flat export cap per household (kW), both regimes")
    p.add_argument("--zone-limit-frac", type=float, default=0.95,
                   help="Zone limit C as a fraction of the day's measured "
                        "substation peak")
    p.add_argument("--zone-limit-mw", type=float, default=None,
                   help="Absolute zone limit C (MW); overrides the fraction")
    p.add_argument("--feeder-cap-frac", type=float, default=1.0,
                   help="Feeder planning cap as a fraction of the "
                        "no-battery peak")
    p.add_argument("--penalty", type=float, default=1e3,
                   help="Soft-envelope slack penalty ($/kW-equivalent)")
    p.add_argument("--glm-dir", default=str(GLM_DIR))
    p.add_argument("--common-dir", default=str(GLM_COMMON))
    p.add_argument("--n-monitors", type=int, default=100)
    p.add_argument("--two-stage-rules", default="",
                   help="Also collect Method B (two-stage DOE allocation: "
                        "the same envelope pre-allocated into soft "
                        "per-household slices) for these comma-separated "
                        "rules, e.g. maxmin,equal; empty = off")
    p.add_argument("--runs-root", default=str(RUNS))
    p.add_argument("--skip-network", action="store_true",
                   help="Stop after the dispatch figures (no OpenDSS)")
    return p.parse_args()


def main():
    args = parse_args()

    jes_kw = load_jesmond_day(args.jesmond, args.date)
    day_arrays = vc.load_day_arrays(args.data)
    households, date_iso, tariff = vc.assemble_ensemble(
        day_arrays, args.n_households, args.date,
        mode=args.mode, e_max=args.e_max)
    n_hh = len(households)
    p_base = np.sum([hh.net for hh in households], axis=0)

    zone_limit_kw = (args.zone_limit_mw * MW_TO_KW if args.zone_limit_mw
                     else args.zone_limit_frac * float(jes_kw.max()))
    feeder_cap_kw = args.feeder_cap_frac * float(p_base.max())
    d_max = zone_headroom_envelope(jes_kw, p_base, zone_limit_kw,
                                   feeder_cap_kw)
    stress = jes_kw > zone_limit_kw
    excess_kwh = float(np.maximum(jes_kw - zone_limit_kw, 0).sum() * DT)
    logger.info("Jesmond %s: peak %.3f MW at %.1f h; zone limit %.3f MW "
                "exceeded for %d intervals (%.0f kWh); fleet holds "
                "%.0f kWh", date_iso, jes_kw.max() / MW_TO_KW,
                HOURS[int(np.argmax(jes_kw))], zone_limit_kw / MW_TO_KW,
                int(stress.sum()), excess_kwh, n_hh * args.e_max)
    logger.info("DOE import limit: %.0f .. %.0f kW (%.2f .. %.2f "
                "kW/household); a static import limit giving the same "
                "substation relief would be the minimum all day, below "
                "the %.0f kW overnight baseline", d_max.min(), d_max.max(),
                d_max.min() / n_hh, d_max.max() / n_hh, p_base.min())

    two_stage_rules = [r.strip() for r in args.two_stage_rules.split(",")
                       if r.strip()]
    cases = solve_cases(households, args.export_limit, d_max, args.penalty,
                        two_stage_rules=two_stage_rules)
    aggs = {name: vc.aggregate_pi(households, cases[name]["B"])
            for name in cases}
    metrics = [case_metrics(name, households, cases[name], tariff,
                            args.mode, jes_kw, p_base, zone_limit_kw)
               for name in cases]
    peak0 = metrics[0]["peak_kw"]
    for row in metrics:
        row["peak_vs_nobatt_pct"] = 100.0 * (row["peak_kw"] - peak0) / peak0

    run_dir = Path(args.runs_root) / (
        f"static_vs_doe_{date_iso}_{datetime.now():%Y%m%d-%H%M%S}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csvs = export_cases(run_dir, households, cases, date_iso, tariff,
                        args.mode)

    fig_doe_derivation(jes_kw, p_base, zone_limit_kw, feeder_cap_kw, d_max,
                       date_iso, n_hh, fig_dir)
    fig_feeder_profile(aggs, d_max, stress, metrics, date_iso, n_hh, fig_dir)

    manifest = {
        "kind": "static_vs_doe_replay",
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "ensemble": {"n_households": n_hh, "date": date_iso,
                     "mode": args.mode, "e_max_kwh": args.e_max,
                     "unique_customers": len({hh.customer
                                              for hh in households})},
        "jesmond": {"peak_kw": float(jes_kw.max()),
                    "peak_hour": float(HOURS[int(np.argmax(jes_kw))]),
                    "zone_limit_kw": zone_limit_kw,
                    "excess_kwh": excess_kwh,
                    "stress_intervals": int(stress.sum()),
                    "kw": jes_kw},
        "envelope": {"feeder_cap_kw": feeder_cap_kw,
                     "d_min_kw": cases["doe"]["d_min"],
                     "d_max_doe_kw": d_max},
        "aggregate_kw": {name: aggs[name] for name in aggs},
        "metrics": metrics,
        "files": {name: Path(p).name for name, p in csvs.items()},
    }

    if not args.skip_network:
        results, scale, n_loads = simulate(run_dir, csvs, args)
        net_rows = network_rows(results)
        fig_zone_transformer(results, jes_kw, zone_limit_kw, stress, scale,
                             n_loads, n_hh, date_iso, fig_dir)
        manifest["network"] = {
            "n_loads": n_loads, "scale": scale,
            "tx_p_kw": {name: results[name]["tx_p_kw"] for name in results},
            "voltages": {name: f"voltages_{name}.npy" for name in results},
            "voltage_monitors": "voltage_monitors.txt",
            "summary": net_rows,
        }
        net_by_case = {net["case"]: net for net in net_rows}
        for row in metrics:
            row.update({k: v for k, v in net_by_case[row["case"]].items()
                        if k != "case"})

    summary = pd.DataFrame(metrics)
    summary.to_csv(run_dir / "summary.csv", index=False)
    # strict JSON (inf -> "inf", NaN -> null), same convention as vpp_export
    (run_dir / "manifest.json").write_text(
        json.dumps(vexport.json_safe(manifest), indent=2, allow_nan=False),
        encoding="utf-8")

    print(f"\n=== Static limits vs DOE on {date_iso}: N={n_hh}, zone "
          f"limit {zone_limit_kw / MW_TO_KW:.2f} MW "
          f"({100 * zone_limit_kw / jes_kw.max():.0f} % of the measured "
          f"peak), feeder cap {feeder_cap_kw:.0f} kW ===")
    cols = ["label", "peak_kw", "peak_hour", "peak_vs_nobatt_pct",
            "max_ramp_kw", "std_kw", "zone_exceed_kwh",
            "import_shortfall_kwh", "export_excess_kwh", "savings_per_day"]
    if "tx_peak_kw" in summary:
        cols += ["tx_peak_kw", "v_min_pu", "n_voltage_violations"]
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.1f}"))
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
