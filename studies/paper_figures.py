"""
paper_figures.py
================

Figures for the final paper that no other script produces, built only
from existing run folders (no solver or power-flow calls):

  savings_cdf              Fig. 6   empirical CDF of per-household daily
                                    savings, Jain's index in the legend
  zone_limit_sensitivity   Fig. 10  feeder peak, fleet savings and
                                    envelope breach against the
                                    zone-limit fraction f
  voltage_envelope         Fig. 8   min / max customer voltage across the
                                    monitored loads, AS IEC 60038 limits
  attribution_2200         (opt.)   under-voltage points by time of day
                                    over the sampled year, 22:00-24:00
                                    highlighted

Every figure carries one series per regime in a fixed colour and line
style -- no battery grey dotted, static limits orange dashed,
centralised VPP blue solid, two-stage DOE purple dash-dot -- and is
written as PNG (300 dpi) plus PDF, sized for one IEEE column. The
numbers the paper quotes are recomputed from the same inputs and listed
in checks.md as expected / computed / match; a CSV twin of each figure's
data sits next to it.

Inputs (all written by studies/static_vs_doe_replay.py unless noted):
  --run              static-vs-centralised run: dispatch_<case>.csv,
                     summary.csv, manifest.json
  --network-run      run holding voltages_<case>.npy (default: --run)
  --two-stage-run    run holding dispatch_<two-stage case>.csv and, for
                     the voltage figure, voltages_<case>.npy (a replay run
                     made with --two-stage-rules); default: --network-run,
                     then --run. Without one the figures carry no
                     two-stage series.
  --sweep-runs       --zone-limit-frac runs, one per f (paths or globs;
                     the latest run per f wins)
  --attribution-csv  network/diagnostics/diag_violation_attribution.py
                     --csv output

Usage:
    python studies/paper_figures.py --run outputs/runs/static_vs_doe_<ts> \\
        --network-run outputs/runs/static_vs_doe_<ts2> \\
        --sweep-runs "outputs/runs/static_vs_doe_2011-02-05_*" \\
        --attribution-csv outputs/figures/paper/attribution_by_hour.csv
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FIGURES, RUNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)   # PDF subsetting
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

T = 48
DT = 0.5
HOURS = np.arange(T) * DT                 # interval start, 0.0 .. 23.5
V_LOWER_PU, V_UPPER_PU = 0.94, 1.10       # AS IEC 60038: -6 % / +10 %
CHARGE_BLOCK = (22.0, 24.0)               # off-peak tariff opens at 22:00
OPERATING_POINT = 0.95                    # zone-limit fraction of the paper
DEFAULT_OUT = FIGURES / "paper"
TWO_STAGE_PREFIX = "two_stage_"

# ---- style -----------------------------------------------------------------
COL_W_IN = 3.5                            # one IEEE column
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
C_LIMIT = "#d03b3b"
# regime -> (label, colour, line style); the two-stage purple was chosen
# with the dataviz palette validator against the orange and the blue
REGIMES = {
    "nobatt": ("no battery", "#898781", ":"),
    "static": ("static limits", "#eb6834", "--"),
    "doe": ("centralised VPP", "#2a78d6", "-"),
    "two_stage": ("two-stage DOE", "#a23fa6", "-."),
}

# ---- what the paper quotes (Tables I, II and Section III-D/E text) --------
PAPER = {
    "fleet_savings": {"static": 260.5, "doe": 205.7},
    "mean_savings": {"static": 1.71, "doe": 1.35},
    "median_savings": {"static": 2.60, "doe": 1.59},
    "n_households": 152,
    "n_worse_off": 136,
    "max_loss": 2.30,
    "jain": {"static": 0.71, "doe": 0.66},
    "op_point": {"peak_kw": 438.0, "savings": 205.7, "shortfall": 0.0},
    "v_min": {"nobatt": 0.749, "static": 0.648, "doe": 0.714},
    "violations": {"nobatt": 340, "static": 297, "doe": 288},
    "total_points": 4800,
    "attribution_added_share": 0.96,     # paper: "added" points
    "attribution_total_share": 0.82,     # WALKTHROUGH: "of the QP's"
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "legend.frameon": False,
        "legend.handlelength": 2.4, "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style(case):
    """(label, colour, line style) for a run-folder case name."""
    if case.startswith(TWO_STAGE_PREFIX):
        return REGIMES["two_stage"]
    return REGIMES[case]


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        logger.info("wrote %s", path)
    plt.close(fig)
    return out_dir / f"{name}.png"


def hour_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")


def shade_charge_block(ax):
    ax.axvspan(*CHARGE_BLOCK, color=INK, alpha=0.06, linewidth=0)


# ---- loaders ---------------------------------------------------------------

def load_manifest(run_dir):
    return json.loads((Path(run_dir) / "manifest.json").read_text("utf-8"))


def load_summary(run_dir):
    return pd.read_csv(Path(run_dir) / "summary.csv").set_index("case")


def load_savings(run_dir, case):
    """Per-household daily savings ($/day) from dispatch_<case>.csv; the
    column repeats one value per customer over its 48 rows."""
    path = Path(run_dir) / f"dispatch_{case}.csv"
    df = pd.read_csv(path, usecols=["customer", "daily_savings"])
    g = df.groupby("customer")["daily_savings"].agg(["first", "nunique"])
    if (g["nunique"] != 1).any():
        raise ValueError(f"{path.name}: daily_savings varies within a "
                         "customer")
    return g["first"].rename(case)


def load_voltages(run_dir, case):
    V = np.load(Path(run_dir) / f"voltages_{case}.npy")
    if V.ndim != 2 or V.shape[1] != T:
        raise ValueError(f"voltages_{case}.npy: expected (n_monitors, {T}), "
                         f"got {V.shape}")
    return V


def has(run_dir, name):
    return run_dir is not None and (Path(run_dir) / name).is_file()


# ---- statistics ------------------------------------------------------------

def jain(values, clip_negative=True):
    """Jain's index (sum s)^2 / (N sum s^2) of the per-household savings.
    By default negatives are clipped to zero first (a household left
    worse off than no battery counts as zero benefit), which is what
    vpp_common.jain_index does and where the paper's 0.71 -> 0.66 come
    from; the index on the raw vector is also reported in checks.md."""
    s = np.asarray(values, dtype=float)
    if clip_negative:
        s = np.maximum(s, 0.0)
    denom = len(s) * float(np.sum(s ** 2))
    return float(s.sum() ** 2 / denom) if denom > 0 else 1.0


def ecdf_xy(values):
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    return np.r_[x[0], x], np.r_[0.0, np.arange(1, n + 1) / n]


def envelope_breach_kwh(agg_kw, d_max_kw):
    """Energy by which an aggregate profile exceeds the feeder import
    envelope: sum_k max(P_k - D_max,k, 0) * DT. For the centralised soft
    solve this equals the reported slack; for a two-stage dispatch it is
    the feeder-level counterpart of its per-household slice breaches."""
    agg = np.asarray(agg_kw, dtype=float)
    d_max = np.asarray(d_max_kw, dtype=float)
    return float(np.maximum(agg - d_max, 0.0).sum() * DT)


class Checks:
    """expected / computed / match rows for checks.md."""

    def __init__(self):
        self.rows = []
        self.sources = []

    def add(self, figure, item, expected, computed, tol=None, unit=""):
        if expected is None:
            match = "info"
        elif isinstance(expected, str):
            match = "yes" if computed == expected else "NO"
        else:
            match = "yes" if abs(float(computed) - expected) <= tol else "NO"
        self.rows.append(dict(figure=figure, item=item, expected=expected,
                              computed=computed, match=match, unit=unit))
        flag = "" if match in ("yes", "info") else "   <-- MISMATCH"
        logger.info("[%s] %-58s expected %-8s computed %-14s %s%s",
                    figure, item, _fmt(expected), _fmt(computed), unit, flag)

    def source(self, figure, what, path):
        self.sources.append(dict(figure=figure, what=what, path=str(path)))

    def write(self, out_dir):
        lines = ["# Paper figure checks", "",
                 "Computed by studies/paper_figures.py from the run folders "
                 "below; expected values are the ones the paper quotes.", "",
                 "## Sources", "", "| figure | data | path |", "|---|---|---|"]
        lines += [f"| {s['figure']} | {s['what']} | `{s['path']}` |"
                  for s in self.sources]
        lines += ["", "## Checks", "",
                  "| figure | item | expected | computed | match |",
                  "|---|---|---|---|---|"]
        for r in self.rows:
            unit = f" {r['unit']}" if r["unit"] else ""
            exp = _fmt(r["expected"]) + (unit if r["expected"] is not None
                                         else "")
            lines.append(f"| {r['figure']} | {r['item']} | {exp} | "
                         f"{_fmt(r['computed'])}{unit} | {r['match']} |")
        path = Path(out_dir) / "checks.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote %s", path)
        return sum(r["match"] == "NO" for r in self.rows)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


# ---- Fig. 6: savings CDF ---------------------------------------------------

def fig_savings_cdf(series, checks, out_dir, clip_negative=True):
    """series: {case: pd.Series of per-household daily savings}."""
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.4))
    for case, s in series.items():
        label, colour, ls = style(case)
        x, y = ecdf_xy(s.to_numpy())
        ax.step(x, y, where="post", color=colour, ls=ls,
                label=f"{label}, J = {jain(s, clip_negative):.2f}")
    ax.axvline(0.0, color=INK_2, lw=0.6)
    ax.set_xlabel("daily savings per household ($/day)")
    ax.set_ylabel("fraction of households")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right")
    path = save(fig, out_dir, "savings_cdf")

    table = pd.DataFrame(series)
    table.index.name = "customer"
    table.to_csv(Path(out_dir) / "savings_per_household.csv")

    n = len(table)
    checks.add("savings_cdf", "households (N)", PAPER["n_households"], n, 0)
    for case in ("static", "doe"):
        if case not in table:
            continue
        s = table[case]
        checks.add("savings_cdf", f"fleet savings, {case}",
                   PAPER["fleet_savings"][case], round(float(s.sum()), 2),
                   0.05, "$/day")
        checks.add("savings_cdf", f"mean per household, {case}",
                   PAPER["mean_savings"][case], round(float(s.mean()), 3),
                   0.005, "$/day")
        checks.add("savings_cdf", f"median per household, {case}",
                   PAPER["median_savings"][case], round(float(s.median()), 3),
                   0.005, "$/day")
        checks.add("savings_cdf", f"Jain (negatives clipped), {case}",
                   PAPER["jain"][case], round(jain(s, clip_negative=True), 3),
                   0.005)
        checks.add("savings_cdf", f"Jain (raw vector), {case}",
                   PAPER["jain"][case], round(jain(s, clip_negative=False), 3),
                   0.005)
    if {"static", "doe"} <= set(table):
        delta = table["static"] - table["doe"]
        checks.add("savings_cdf", "households worse off under centralised",
                   PAPER["n_worse_off"], int((delta > 0).sum()), 0)
        checks.add("savings_cdf", "largest loss vs static",
                   PAPER["max_loss"], round(float(delta.max()), 3), 0.005,
                   "$/day")
        checks.add("savings_cdf", "households better off under centralised",
                   None, int((delta < 0).sum()))
    for case in table:
        if case.startswith(TWO_STAGE_PREFIX):
            s = table[case]
            checks.add("savings_cdf", f"fleet savings, {case}", None,
                       round(float(s.sum()), 2), unit="$/day")
            checks.add("savings_cdf", f"median per household, {case}", None,
                       round(float(s.median()), 3), unit="$/day")
            checks.add("savings_cdf", f"Jain (negatives clipped), {case}",
                       None, round(jain(s, clip_negative=True), 3))
            checks.add("savings_cdf", f"Jain (raw vector), {case}", None,
                       round(jain(s, clip_negative=False), 3))
            if "static" in table:
                d = table["static"] - s
                checks.add("savings_cdf",
                           f"households worse off than static, {case}",
                           None, int((d > 0).sum()))
    return path


# ---- Fig. 10: zone-limit sensitivity ---------------------------------------

def collect_sweep(run_dirs):
    """One row per (f, case) from the sweep runs; the latest run (manifest
    'created') wins when an f was run more than once. Besides the
    summary.csv columns, feeder_breach_kwh recomputes the envelope breach
    from the aggregate profile with one definition for every regime."""
    rows = []
    for d in run_dirs:
        m = load_manifest(d)
        jes = m["jesmond"]
        f = (jes["zone_limit_kw"] / jes["peak_kw"]
             if m["args"].get("zone_limit_mw")
             else float(m["args"]["zone_limit_frac"]))
        d_max = np.asarray(m["envelope"]["d_max_doe_kw"], dtype=float)
        s = load_summary(d)
        for case, r in s.iterrows():
            rows.append(dict(
                f=round(f, 4), excess_kwh=float(jes["excess_kwh"]),
                zone_limit_kw=float(jes["zone_limit_kw"]),
                run=Path(d).name, created=m["created"], case=case,
                peak_kw=float(r["peak_kw"]),
                peak_hour=float(r["peak_hour"]),
                savings_per_day=float(r["savings_per_day"]),
                zone_exceed_kwh=float(r["zone_exceed_kwh"]),
                import_shortfall_kwh=float(r["import_shortfall_kwh"]),
                feeder_breach_kwh=envelope_breach_kwh(
                    m["aggregate_kw"][case], d_max),
                n_failed=int(r["n_failed"]) if "n_failed" in r else 0))
    df = pd.DataFrame(rows)
    return (df.sort_values("created").groupby(["f", "case"]).tail(1)
              .sort_values(["f", "case"]).reset_index(drop=True))


PANELS = (("peak_kw", "(a) feeder peak (kW)"),
          ("savings_per_day", "(b) fleet savings ($/day)"),
          ("feeder_breach_kwh", "(c) feeder-envelope breach (kWh)"))


def fig_zone_limit_sensitivity(sweep, cases, checks, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(COL_W_IN, 4.8), sharex=True,
                             gridspec_kw=dict(hspace=0.24))
    fs = sorted(sweep["f"].unique())
    excess = sweep.groupby("f")["excess_kwh"].first()
    handles = {}
    for ax, (col, title) in zip(axes, PANELS):
        for case in cases:
            if col == "feeder_breach_kwh" and case in ("nobatt", "static"):
                continue                 # not envelope-bound regimes
            sub = sweep[sweep["case"] == case].sort_values("f")
            if sub.empty:
                continue
            label, colour, ls = style(case)
            (h,) = ax.plot(sub["f"], sub[col], color=colour, ls=ls,
                           marker="o", ms=3.2, mec="white", mew=0.6,
                           label=label)
            handles.setdefault(case, h)
        ax.axvline(OPERATING_POINT, color=INK_2, lw=0.6, alpha=0.7)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    axes[0].set_ylim(bottom=400)             # the peaks live in 438..594 kW
    axes[0].annotate("operating point of the paper",
                     xy=(OPERATING_POINT, 0.05),
                     xycoords=("data", "axes fraction"), xytext=(3, 0),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=7, color=INK_2)
    axes[0].legend([handles[c] for c in cases if c in handles],
                   [handles[c].get_label() for c in cases if c in handles],
                   loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=2,
                   columnspacing=1.2, handletextpad=0.6)
    axes[-1].set_xticks(fs)
    axes[-1].set_xticklabels([f"{f:.2f}\n{excess[f] / 1000:.1f}" for f in fs])
    axes[-1].set_xlabel("zone limit $f$ (fraction of measured peak)\n"
                        "substation excess above the limit (MWh)")
    axes[-1].set_xlim(max(fs) + 0.006, min(fs) - 0.006)   # tighter to the right
    path = save(fig, out_dir, "zone_limit_sensitivity")

    sweep.to_csv(Path(out_dir) / "zone_limit_sweep.csv", index=False)
    op = sweep[(np.isclose(sweep["f"], OPERATING_POINT))
               & (sweep["case"] == "doe")]
    if op.empty:
        checks.add("zone_limit_sensitivity", "f = 0.95 run present", "yes",
                   "no")
    else:
        r = op.iloc[0]
        checks.add("zone_limit_sensitivity", "f = 0.95 centralised peak",
                   PAPER["op_point"]["peak_kw"], round(r["peak_kw"], 2), 0.5,
                   "kW")
        checks.add("zone_limit_sensitivity", "f = 0.95 centralised savings",
                   PAPER["op_point"]["savings"], round(r["savings_per_day"], 2),
                   0.05, "$/day")
        checks.add("zone_limit_sensitivity", "f = 0.95 centralised shortfall",
                   PAPER["op_point"]["shortfall"],
                   round(r["import_shortfall_kwh"], 4), 1e-3, "kWh")
    for case in cases:
        sub = sweep[sweep["case"] == case].sort_values("f", ascending=False)
        for _, r in sub.iterrows():
            extra = (f", slice breaches {r['import_shortfall_kwh']:.0f} kWh"
                     if case.startswith(TWO_STAGE_PREFIX) else "")
            extra += f", {r['n_failed']} failed" if r["n_failed"] else ""
            checks.add("zone_limit_sensitivity",
                       f"f = {r['f']:.2f} {case}", None,
                       f"peak {r['peak_kw']:.0f} kW at {r['peak_hour']:04.1f} h,"
                       f" savings ${r['savings_per_day']:.1f}, envelope breach"
                       f" {r['feeder_breach_kwh']:.0f} kWh, substation excess"
                       f" {r['zone_exceed_kwh']:.0f} kWh{extra}")
    return path


# ---- Fig. 8: voltage envelope ----------------------------------------------

def fig_voltage_envelope(volt, checks, out_dir):
    """volt: {case: array (n_monitors, T) in p.u.}. Two panels sharing the
    hour axis: the maximum and the minimum across the monitored loads,
    one line per regime, so the 22:00 dip stays legible next to the
    boost-tap ceiling."""
    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, figsize=(COL_W_IN, 3.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.10))
    for ax in (ax_hi, ax_lo):
        shade_charge_block(ax)
    for case, V in volt.items():
        label, colour, ls = style(case)
        ax_hi.plot(HOURS, V.max(axis=0), color=colour, ls=ls, lw=1.0)
        ax_lo.plot(HOURS, V.min(axis=0), color=colour, ls=ls, label=label)
    ax_hi.axhline(V_UPPER_PU, color=C_LIMIT, lw=0.7, ls=(0, (4, 2)))
    ax_lo.axhline(V_LOWER_PU, color=C_LIMIT, lw=0.7, ls=(0, (4, 2)))
    ax_hi.annotate(f"+10 % ({V_UPPER_PU:.2f} p.u.)", xy=(0.3, V_UPPER_PU),
                   xytext=(0, -2), textcoords="offset points", va="top",
                   fontsize=7, color=C_LIMIT)
    ax_lo.annotate(f"−6 % ({V_LOWER_PU:.2f} p.u.)", xy=(0.3, V_LOWER_PU),
                   xytext=(0, 2), textcoords="offset points", va="bottom",
                   fontsize=7, color=C_LIMIT)
    ax_lo.text(CHARGE_BLOCK[0] - 0.3, 0.995, "22:00–24:00",
               transform=ax_lo.get_xaxis_transform(), ha="right", va="top",
               fontsize=7, color=INK_2)
    if "static" in volt:
        vmin = volt["static"].min(axis=0)
        k = int(np.argmin(vmin))
        ax_lo.annotate(f"{vmin[k]:.3f} p.u. at {HOURS[k]:04.1f} h",
                       xy=(HOURS[k], vmin[k]), xytext=(-10, 9),
                       textcoords="offset points", ha="right", va="bottom",
                       fontsize=7, color=INK_2,
                       arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax_hi.set_ylabel("max. (p.u.)")
    ax_lo.set_ylabel("min. voltage across monitored loads (p.u.)")
    ax_lo.legend(loc="lower left")
    hour_axis(ax_lo)
    path = save(fig, out_dir, "voltage_envelope")

    table = pd.DataFrame({"hour": HOURS})
    for case, V in volt.items():
        table[f"{case}_min_pu"] = V.min(axis=0)
        table[f"{case}_max_pu"] = V.max(axis=0)
    table.to_csv(Path(out_dir) / "voltage_envelope.csv", index=False)

    for case, V in volt.items():
        vmin, vmax = float(V.min()), float(V.max())
        n_under = int((V < V_LOWER_PU).sum())
        n_over = int((V > V_UPPER_PU).sum())
        k = int(np.argmin(V.min(axis=0)))
        checks.add("voltage_envelope", f"monitored points, {case}",
                   PAPER["total_points"], int(V.size), 0)
        checks.add("voltage_envelope", f"minimum voltage, {case}",
                   PAPER["v_min"].get(case), round(vmin, 4), 0.001, "p.u.")
        checks.add("voltage_envelope", f"violation points, {case}",
                   PAPER["violations"].get(case), n_under + n_over, 0)
        checks.add("voltage_envelope", f"under / over split, {case}", None,
                   f"{n_under} / {n_over}")
        checks.add("voltage_envelope", f"hour of minimum, {case}", None,
                   f"{HOURS[k]:04.1f} h")
        checks.add("voltage_envelope", f"maximum voltage, {case}", None,
                   round(vmax, 4), unit="p.u.")
        if case == "static":
            inside = CHARGE_BLOCK[0] <= HOURS[k] < CHARGE_BLOCK[1]
            checks.add("voltage_envelope",
                       "static minimum inside 22:00-24:00", "yes",
                       "yes" if inside else "no")
    return path


# ---- optional: year-sample 22:00 attribution -------------------------------

def read_attribution(csv_path):
    head = Path(csv_path).read_text(encoding="utf-8").splitlines()[0]
    meta = (dict(tok.split("=", 1) for tok in head.lstrip("# ").split()
                 if "=" in tok) if head.startswith("#") else {})
    df = pd.read_csv(csv_path, comment="#")
    if len(df) != T:
        raise ValueError(f"{csv_path}: expected {T} rows, got {len(df)}")
    return df, meta


def fig_attribution_2200(csv_path, checks, out_dir):
    df, meta = read_attribution(csv_path)
    late = (df["hour"] >= CHARGE_BLOCK[0]).to_numpy()
    qp, base = df["qp_under"].to_numpy(), df["base_under"].to_numpy()
    added = qp - base
    total_share = qp[late].sum() / max(1, qp.sum())
    added_share_net = (added[late].sum() / added.sum() if added.sum()
                       else np.nan)
    added_pos = np.maximum(added, 0)
    added_share_pos = added_pos[late].sum() / max(1, added_pos.sum())
    base_share = base[late].sum() / max(1, base.sum())

    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.4))
    shade_charge_block(ax)
    edges = np.r_[HOURS, 24.0]
    for case, y, label in (("nobatt", base, "no battery"),
                           ("static", qp, "tariff-driven scheduler")):
        _label, colour, ls = style(case)
        ax.stairs(y, edges, color=colour, ls=ls, lw=1.2, label=label,
                  baseline=None)
        ax.stairs(y, edges, color=colour, alpha=0.10, fill=True, lw=0)
    days, mons = meta.get("days", "?"), meta.get("monitors", "?")
    every = meta.get("every", "?")
    ax.text(CHARGE_BLOCK[0] - 0.4, 0.58,
            f"22:00–24:00 holds {100 * total_share:.0f} % of the\n"
            f"scheduler's under-voltage points\n"
            f"(no battery: {100 * base_share:.0f} %)",
            transform=ax.get_xaxis_transform(), ha="right", va="center",
            fontsize=7, color=INK_2)
    ax.set_ylabel("under-voltage load-intervals\n(< 0.94 p.u., summed over days)")
    ax.legend(loc="upper left")
    hour_axis(ax)
    ax.set_title(f"{days} sampled days (every {every}th), {mons} monitored "
                 "loads", loc="left", fontsize=7, color=INK_2, pad=3)
    path = save(fig, out_dir, "attribution_2200")

    df.to_csv(Path(out_dir) / "attribution_2200.csv", index=False)
    checks.add("attribution_2200", "sampled days (every Nth) / monitors",
               None, f"{days} (every {every}th) / {mons}")
    checks.add("attribution_2200", "scheduler under-voltage points, total",
               None, int(qp.sum()))
    checks.add("attribution_2200", "no-battery under-voltage points, total",
               None, int(base.sum()))
    checks.add("attribution_2200", "share of scheduler total in 22:00-24:00 "
               "(WALKTHROUGH: 82 %)", PAPER["attribution_total_share"],
               round(total_share, 3), 0.005)
    checks.add("attribution_2200", "share of ADDED (scheduler - baseline, "
               "net) in 22:00-24:00 (paper: 96 %)",
               PAPER["attribution_added_share"], round(added_share_net, 3),
               0.005)
    checks.add("attribution_2200", "share of added (positive part only) in "
               "22:00-24:00", None, round(added_share_pos, 3))
    checks.add("attribution_2200", "share of no-battery total in 22:00-24:00",
               None, round(base_share, 3))
    return path


# ---- main ------------------------------------------------------------------

def latest_run(prefix="static_vs_doe_"):
    runs = sorted(p for p in Path(RUNS).glob(f"{prefix}*") if p.is_dir())
    return runs[-1] if runs else None


def expand(patterns):
    out = []
    for pat in patterns or []:
        hits = [Path(p) for p in glob.glob(str(pat))] or [Path(pat)]
        out += [h for h in hits if h.is_dir()
                and (h / "summary.csv").is_file()]
    return sorted(set(out))


def parse_args():
    p = argparse.ArgumentParser(
        description="Paper figures from existing run folders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", default=None,
                   help="static-vs-centralised run (default: latest "
                        "static_vs_doe_* under outputs/runs)")
    p.add_argument("--network-run", default=None,
                   help="run with voltages_<case>.npy (default: --run)")
    p.add_argument("--two-stage-run", default=None,
                   help="run with dispatch_two_stage_<rule>.csv "
                        "(default: --network-run, then --run)")
    p.add_argument("--two-stage-case", default="two_stage_maxmin",
                   help="which two-stage case to draw")
    p.add_argument("--sweep-runs", nargs="*", default=None,
                   help="zone-limit sweep runs (paths or globs)")
    p.add_argument("--attribution-csv", default=None,
                   help="per-interval CSV from diag_violation_attribution.py")
    p.add_argument("--jain-raw", action="store_true",
                   help="legend Jain index on the raw savings vector "
                        "(default: negatives clipped to zero first, as "
                        "vpp_common.jain_index and the paper's numbers)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks()

    run = Path(args.run) if args.run else latest_run()
    if run is None or not has(run, "summary.csv"):
        raise SystemExit("no static_vs_doe run found; pass --run")
    net_run = Path(args.network_run) if args.network_run else run
    ts_case = args.two_stage_case
    ts_run = None
    for cand in ([Path(args.two_stage_run)] if args.two_stage_run
                 else [net_run, run]):
        if has(cand, f"dispatch_{ts_case}.csv"):
            ts_run = cand
            break
    if args.two_stage_run and ts_run is None:
        raise SystemExit(f"{args.two_stage_run}: no dispatch_{ts_case}.csv")
    logger.info("run: %s", run.name)
    logger.info("network run: %s", net_run.name)
    logger.info("two-stage run: %s (%s)",
                ts_run.name if ts_run else "none", ts_case)

    # --- Fig. 6 ---
    series = {c: load_savings(run, c) for c in ("static", "doe")}
    checks.source("savings_cdf", "static + centralised savings", run)
    if ts_run is not None:
        series[ts_case] = load_savings(ts_run, ts_case)
        checks.source("savings_cdf", f"{ts_case} savings", ts_run)
        if ts_run != run:
            # the two-stage run must carry the same static / centralised
            # dispatch as --run, or the CDFs would mix ensembles
            for c in ("static", "doe"):
                d = float(np.abs(load_savings(ts_run, c).to_numpy()
                                 - series[c].to_numpy()).max())
                checks.add("savings_cdf", f"|{c} savings, two-stage run - "
                           f"--run| max", 0.0, round(d, 6), 1e-4, "$/day")
    fig_savings_cdf(series, checks, out_dir, clip_negative=not args.jain_raw)

    # --- Fig. 10 ---
    sweep_dirs = expand(args.sweep_runs)
    if sweep_dirs:
        sweep = collect_sweep(sweep_dirs)
        cases = ["nobatt", "static", "doe"]
        if ts_case in set(sweep["case"]):
            cases.append(ts_case)
        for d in sorted(set(sweep["run"])):
            checks.source("zone_limit_sensitivity", "sweep run",
                          Path(RUNS) / d)
        fig_zone_limit_sensitivity(sweep, cases, checks, out_dir)
    else:
        logger.warning("no --sweep-runs: zone_limit_sensitivity skipped")

    # --- Fig. 8 ---
    volt = {}
    for case in ("nobatt", "static", "doe"):
        if has(net_run, f"voltages_{case}.npy"):
            volt[case] = load_voltages(net_run, case)
    if volt:
        checks.source("voltage_envelope", "per-monitor voltages", net_run)
        if ts_run is not None and has(ts_run, f"voltages_{ts_case}.npy"):
            volt[ts_case] = load_voltages(ts_run, ts_case)
            checks.source("voltage_envelope", f"{ts_case} voltages", ts_run)
        fig_voltage_envelope(volt, checks, out_dir)
    else:
        logger.warning("no voltages_<case>.npy in %s: voltage_envelope "
                       "skipped (re-run static_vs_doe_replay.py, which now "
                       "saves them)", net_run)

    # --- optional ---
    if args.attribution_csv and Path(args.attribution_csv).is_file():
        checks.source("attribution_2200", "per-interval counts",
                      args.attribution_csv)
        fig_attribution_2200(args.attribution_csv, checks, out_dir)
    elif args.attribution_csv:
        logger.warning("%s missing: attribution_2200 skipped",
                       args.attribution_csv)

    n_bad = checks.write(out_dir)
    print(f"\n{len(checks.rows)} checks, {n_bad} mismatches -> "
          f"{out_dir / 'checks.md'}")


if __name__ == "__main__":
    main()
