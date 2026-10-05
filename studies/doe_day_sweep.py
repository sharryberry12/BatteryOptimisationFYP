"""
doe_day_sweep.py
================

Static limits vs centralised DOE vs two-stage DOE on EVERY day the Jesmond
zone-substation record covers (14 Jan - 30 Apr 2011, 105 fully populated
days), with the full clean ensemble each day.

Per day, the recipe of static_vs_doe_replay.py:
  * ensemble of all clean customers, frozen heuristic weights;
  * DOE import limit from the measured substation headroom (III-2),
    D_max,k = min(C_zone - (L_k - P0_k), C_feeder). By default C_zone is
    95 % of EACH day's measured peak (--zone-limit-of day, the definition
    of the single-day study: the DNSP shaves the top of every day and the
    excess stays within the fleet's energy); --zone-limit-of period sets
    one limit for the whole sweep (a fraction of the period maximum, or
    --zone-limit-mw) so only the stressed days bind. The feeder cap is
    each day's no-battery peak;
  * cases: nobatt | static (Method A soft, import unbounded) |
    doe (Method A soft) | two_stage_<rule> (soft slices, all four rules);
  * optional network stage: nobatt, static, doe and --network-rules
    through the Elermore Vale model with per-interval losses.

One row per (day, case) is appended to sweep_results.csv after every day,
so a killed run loses nothing and --resume continues it. The end of the
run (or --summarise-only on a finished one) writes sweep_summary.csv and
three figures.

Usage:
    python studies/doe_day_sweep.py                          # all 105 days, network on
    python studies/doe_day_sweep.py --skip-network --dates 2011-02-01:2011-02-10
    python studies/doe_day_sweep.py --resume outputs/runs/doe_sweep_<ts>
    python studies/doe_day_sweep.py --summarise-only outputs/runs/doe_sweep_<ts>
"""

import argparse
import json
import logging
import sys
import time
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
from studies import static_vs_doe_replay as sr  # noqa: E402
from studies import battery_location_study as bl  # noqa: E402
from studies.peak_duty_analysis import (  # noqa: E402
    C_AQUA, C_BLUE, C_ORANGE, C_RED, INK_2, MUTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

T = vc.T
DT = vc.DT
MW_TO_KW = sr.MW_TO_KW

BASE_CASES = ("nobatt", "static", "doe")
TWO_STAGE_PREFIX = "two_stage_"
DEFAULT_NETWORK_RULES = "maxmin,prorata_surplus"
PROGRESS_EVERY = 10

CASE_STYLE = {
    "nobatt": ("no battery", MUTED, "-"),
    "static": ("static limits", C_ORANGE, "-"),
    "doe": ("DOE, centralised", C_BLUE, "-"),
    "two_stage_maxmin": ("DOE, two-stage max-min", INK_2, "--"),
    "two_stage_prorata_surplus": ("DOE, two-stage need-proportional", C_AQUA, "--"),
}


# ==========================================================
# INPUTS
# ==========================================================

def jesmond_days(csv_path):
    """{date_iso: kW[T]} for every fully populated day of the substation
    file (same parsing rules as static_vs_doe_replay.load_jesmond_day)."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in sr.QUARTER_HOUR_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(csv_path).name}: missing quarter-hour "
                         f"columns {missing[:4]}")
    dates = pd.to_datetime(df["Date"].astype(str).str.strip().str.upper(),
                           format=sr.JESMOND_DATE_FORMAT)
    values = df[list(sr.QUARTER_HOUR_LABELS)].apply(pd.to_numeric,
                                                    errors="coerce")
    full = values.notna().all(axis=1).to_numpy()
    out = {}
    for row in np.flatnonzero(full):
        q = values.iloc[row].to_numpy(dtype=float)
        out[dates.iloc[row].strftime("%Y-%m-%d")] = \
            q.reshape(T, 2).mean(axis=1) * MW_TO_KW
    return out


def select_dates(available, spec):
    """Filter sorted ISO dates by a 'YYYY-MM-DD:YYYY-MM-DD' range or a
    comma-separated list; None keeps everything."""
    dates = sorted(available)
    if not spec:
        return dates
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        return [d for d in dates if lo <= d <= hi]
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = wanted - set(dates)
    if unknown:
        raise ValueError(f"dates not in the substation file: {sorted(unknown)}")
    return [d for d in dates if d in wanted]


def profiles_from_dispatch(households, B, date_iso):
    """
    The nested dict elermorevale_openDSS.load_profiles_from_csv() would
    return for this dispatch, built in memory (pseudo customer ids 1..N,
    one day each) so the network stage needs no CSV round trip.
    """
    B = np.asarray(B, dtype=float)
    out = {}
    for i, (hh, b) in enumerate(zip(households, B)):
        out[i + 1] = [{
            "date": date_iso,
            "load": np.asarray(hh.load, dtype=float),
            "pv": np.asarray(hh.pv, dtype=float),
            "battery": b.copy(),
            "grid": hh.load - hh.pv - b,
            "soc": vc.SOC_INIT_FRAC * hh.e_max - np.cumsum(b) * DT,
            "savings": 0.0,
        }]
    return out


# ==========================================================
# ONE DAY
# ==========================================================

def solve_day(households, tariff, args, jes_kw, zone_limit_kw, rules):
    """All cases for one day -> (cases dict, metrics rows, envelope info)."""
    n = len(households)
    p_base = np.sum([hh.net for hh in households], axis=0)
    feeder_cap_kw = args.feeder_cap_frac * float(p_base.max())
    d_max = sr.zone_headroom_envelope(jes_kw, p_base, zone_limit_kw,
                                      feeder_cap_kw)
    d_min = -args.export_limit * n * np.ones(T)
    stress = jes_kw > zone_limit_kw
    envelope = {
        "zone_limit_kw": zone_limit_kw,
        "feeder_cap_kw": feeder_cap_kw,
        "measured_peak_kw": float(jes_kw.max()),
        "measured_excess_kwh": float(np.maximum(jes_kw - zone_limit_kw, 0)
                                     .sum() * DT),
        "stress_intervals": int(stress.sum()),
        "d_max_min_kw": float(d_max.min()),
        "nobatt_peak_kw": float(p_base.max()),
    }

    cases = sr.solve_cases(households, args.export_limit, d_max, args.penalty)
    for rule in rules:
        B, curtail_kw, shortfall_kw, n_failed = ts.run_rule(
            rule, households, d_min, d_max, soft=True)
        cases[TWO_STAGE_PREFIX + rule] = dict(
            B=B, d_min=d_min, d_max=d_max, n_failed=n_failed,
            result=SimpleNamespace(slack_up=shortfall_kw, slack_lo=curtail_kw))

    rows = []
    for name, case in cases.items():
        row = sr.case_metrics(name, households, case, tariff, args.mode,
                              jes_kw, p_base, zone_limit_kw)
        row["rule"] = name[len(TWO_STAGE_PREFIX):] \
            if name.startswith(TWO_STAGE_PREFIX) else ""
        row["n_failed"] = int(case.get("n_failed", 0))
        row.update(envelope)
        rows.append(row)
    peak0 = rows[0]["peak_kw"]
    for row in rows:
        row["peak_vs_nobatt_pct"] = 100.0 * (row["peak_kw"] - peak0) / peak0
    return cases, rows, envelope


def network_day(ev, args, lc_map, monitored, households, cases, date_iso,
                network_cases):
    """Zone-transformer flow, losses and voltages per case -> {case: dict}."""
    out = {}
    for name in network_cases:
        profiles = profiles_from_dispatch(households, cases[name]["B"],
                                          date_iso)
        r = bl.simulate_case(ev, args, lc_map, monitored, profiles, 0)
        tx = np.asarray(r["tx_p_kw"], dtype=float)
        out[name] = {
            "tx_peak_kw": float(tx.max()),
            "tx_import_kwh": float(np.maximum(tx, 0).sum() * DT),
            "loss_kwh": float(np.sum(r["loss_kw"]) * DT),
            "v_min_pu": r["v_min_pu"],
            "v_max_pu": r["v_max_pu"],
            "n_under": r["n_under"],
            "n_over": r["n_over"],
            "total_points": r["total_points"],
        }
    return out


# ==========================================================
# SUMMARY + FIGURES
# ==========================================================

def summarise(df):
    """Per-case totals over all days and over the binding days."""
    df = df.copy()
    df["binding"] = df["measured_excess_kwh"] > 0
    agg = {
        "days": ("date", "nunique"),
        "peak_kw_mean": ("peak_kw", "mean"),
        "peak_kw_max": ("peak_kw", "max"),
        "peak_vs_nobatt_pct_mean": ("peak_vs_nobatt_pct", "mean"),
        "std_kw_mean": ("std_kw", "mean"),
        "max_ramp_kw_mean": ("max_ramp_kw", "mean"),
        "zone_exceed_kwh_total": ("zone_exceed_kwh", "sum"),
        "import_shortfall_kwh_total": ("import_shortfall_kwh", "sum"),
        "savings_total": ("savings_per_day", "sum"),
        "n_failed_total": ("n_failed", "sum"),
    }
    for col, src in (("tx_peak_kw_mean", "tx_peak_kw"),
                     ("loss_kwh_total", "loss_kwh"),
                     ("v_min_pu_min", "v_min_pu"),
                     ("n_under_total", "n_under"),
                     ("n_over_total", "n_over")):
        if src in df.columns:
            fn = "min" if col.endswith("_min") else \
                ("mean" if col.endswith("_mean") else (lambda s: s.sum(min_count=1)))
            agg[col] = (src, fn)
    rows = []
    for scope, sub in (("all", df), ("binding", df[df["binding"]])):
        if sub.empty:
            continue
        g = sub.groupby("case").agg(**agg).reset_index()
        g.insert(0, "scope", scope)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    order = list(CASE_STYLE) + sorted(set(out["case"]) - set(CASE_STYLE))
    out["case"] = pd.Categorical(out["case"], order)
    return out.sort_values(["scope", "case"]).reset_index(drop=True)


def _style(name):
    return CASE_STYLE.get(name, (name, MUTED, ":"))


def _date_axis(ax, dates):
    ax.set_xlim(dates.min(), dates.max())
    ax.grid(axis="x", visible=False)
    ax.figure.autofmt_xdate()


def fig_daily(df, fig_dir, column, ylabel, title, name, cases):
    pivot = df.pivot(index="date", columns="case", values=column)
    dates = pd.to_datetime(pivot.index)
    binding = df.groupby("date")["measured_excess_kwh"].first() > 0
    fig, ax = plt.subplots(figsize=(11, 5))
    for d, b in binding.items():
        if b:
            ax.axvspan(pd.Timestamp(d) - pd.Timedelta(hours=12),
                       pd.Timestamp(d) + pd.Timedelta(hours=12),
                       color=C_RED, alpha=0.06, linewidth=0)
    for c in cases:
        if c in pivot:
            label, color, ls = _style(c)
            ax.plot(dates, pivot[c], color=color, ls=ls, lw=1.6, label=label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _date_axis(ax, dates)
    fig.tight_layout()
    path = Path(fig_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def fig_cumulative_savings(df, fig_dir, cases):
    pivot = df.pivot(index="date", columns="case", values="savings_per_day")
    dates = pd.to_datetime(pivot.index)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in cases:
        if c in pivot and c != "nobatt":
            label, color, ls = _style(c)
            ax.plot(dates, pivot[c].cumsum(), color=color, ls=ls, lw=1.8,
                    label=label)
    ax.set_ylabel("cumulative fleet savings ($)")
    ax.set_title("What the envelope costs the households, day by day",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    _date_axis(ax, dates)
    fig.tight_layout()
    path = Path(fig_dir) / "sweep_cumulative_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def make_figures(df, run_dir):
    fig_dir = Path(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASE_STYLE if c in set(df["case"])]
    fig_daily(df, fig_dir, "peak_kw", "daily feeder-profile peak (kW)",
              "Daily peak of the feeder power profile by regime "
              "(shaded: substation above its limit)",
              "sweep_daily_peak", cases)
    fig_daily(df, fig_dir, "zone_exceed_kwh",
              "substation excess above the zone limit (kWh)",
              "What the substation still sees above its limit",
              "sweep_zone_exceedance", cases)
    fig_cumulative_savings(df, fig_dir, cases)


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Static limits vs centralised DOE vs two-stage DOE on "
                    "every day the Jesmond substation record covers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=str(DATA_CSV))
    p.add_argument("--jesmond", default=str(JESMOND_CSV))
    p.add_argument("--dates", default=None,
                   help="'YYYY-MM-DD:YYYY-MM-DD' range or comma list "
                        "(default: every populated day)")
    p.add_argument("--n-households", type=int, default=152)
    p.add_argument("--mode", choices=["fit", "net"], default="fit")
    p.add_argument("--e-max", type=float, default=vc.E_MAX_DEFAULT)
    p.add_argument("--export-limit", type=float, default=1.5,
                   help="Flat export cap per household (kW), all regimes")
    p.add_argument("--zone-limit-frac", type=float, default=0.95,
                   help="Zone limit as a fraction of the reference peak")
    p.add_argument("--zone-limit-of", choices=["day", "period"],
                   default="day",
                   help="Reference peak for the fraction: each day's own "
                        "measured peak, or the sweep period's maximum")
    p.add_argument("--zone-limit-mw", type=float, default=None,
                   help="Absolute zone limit (MW) for every day; overrides "
                        "the fraction")
    p.add_argument("--feeder-cap-frac", type=float, default=1.0,
                   help="Feeder cap as a fraction of each day's no-battery peak")
    p.add_argument("--penalty", type=float, default=vc.SOFT_PENALTY_DEFAULT)
    p.add_argument("--rules", default=",".join(ts.RULES),
                   help="Two-stage allocation rules to solve (soft slices)")
    p.add_argument("--network-rules", default=DEFAULT_NETWORK_RULES,
                   help="Two-stage rules also pushed through the network")
    p.add_argument("--skip-network", action="store_true")
    p.add_argument("--glm-dir", default=str(GLM_DIR))
    p.add_argument("--common-dir", default=str(GLM_COMMON))
    p.add_argument("--n-monitors", type=int, default=100)
    p.add_argument("--runs-root", default=str(RUNS))
    p.add_argument("--resume", default=None,
                   help="Existing sweep run directory to continue")
    p.add_argument("--summarise-only", default=None,
                   help="Existing sweep run directory: rebuild summary and "
                        "figures, solve nothing")
    return p.parse_args()


def finish(run_dir, manifest):
    df = pd.read_csv(Path(run_dir) / "sweep_results.csv")
    summary = summarise(df)
    summary.to_csv(Path(run_dir) / "sweep_summary.csv", index=False)
    make_figures(df, run_dir)
    manifest["days_done"] = int(df["date"].nunique())
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(vexport.json_safe(manifest), indent=2, allow_nan=False),
        encoding="utf-8")
    cols = ["scope", "case", "days", "peak_kw_mean", "peak_vs_nobatt_pct_mean",
            "std_kw_mean", "zone_exceed_kwh_total",
            "import_shortfall_kwh_total", "savings_total"]
    cols += [c for c in ("n_under_total", "v_min_pu_min", "loss_kwh_total")
             if c in summary.columns]
    print(f"\n=== DOE day sweep: {manifest['days_done']} days, zone limit "
          f"{manifest['zone_limit_desc']} ===")
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.1f}"))
    print(f"\nArtifacts: {run_dir}")


def main():
    args = parse_args()
    if args.summarise_only:
        run_dir = Path(args.summarise_only)
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        finish(run_dir, manifest)
        return

    jes = jesmond_days(args.jesmond)
    dates = select_dates(jes, args.dates)
    if not dates:
        raise SystemExit("no dates selected")
    period_peak = max(float(jes[d].max()) for d in dates)
    if args.zone_limit_mw:
        zone_limit_of = lambda d: args.zone_limit_mw * MW_TO_KW  # noqa: E731
        zone_desc = f"{args.zone_limit_mw:.2f} MW fixed"
    elif args.zone_limit_of == "period":
        zone_limit_of = lambda d: args.zone_limit_frac * period_peak  # noqa: E731
        zone_desc = (f"{args.zone_limit_frac * period_peak / MW_TO_KW:.2f} MW "
                     f"({100 * args.zone_limit_frac:.0f} % of period peak "
                     f"{period_peak / MW_TO_KW:.2f} MW)")
    else:
        zone_limit_of = lambda d: args.zone_limit_frac * float(jes[d].max())  # noqa: E731
        zone_desc = f"{100 * args.zone_limit_frac:.0f} % of each day's peak"
    zone_key = [args.zone_limit_of, args.zone_limit_frac, args.zone_limit_mw]
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    net_rules = [r.strip() for r in args.network_rules.split(",")
                 if r.strip()] if not args.skip_network else []
    unknown = set(net_rules) - set(rules)
    if unknown:
        raise SystemExit(f"--network-rules {sorted(unknown)} not in --rules")
    network_cases = list(BASE_CASES) + [TWO_STAGE_PREFIX + r for r in net_rules]

    if args.resume:
        run_dir = Path(args.resume)
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        done = set(pd.read_csv(run_dir / "sweep_results.csv")["date"]) \
            if (run_dir / "sweep_results.csv").is_file() else set()
        if list(manifest["zone_limit"]) != zone_key:
            raise SystemExit("resume: zone-limit settings differ from the "
                             f"manifest {manifest['zone_limit']}")
        logger.info("resuming %s: %d days already done", run_dir.name, len(done))
    else:
        run_dir = Path(args.runs_root) / (
            f"doe_sweep_{dates[0]}_{dates[-1]}_{datetime.now():%Y%m%d-%H%M%S}")
        run_dir.mkdir(parents=True, exist_ok=True)
        done = set()
        manifest = {
            "kind": "doe_day_sweep",
            "created": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "dates": dates,
            "period_peak_kw": period_peak,
            "zone_limit": zone_key,
            "zone_limit_desc": zone_desc,
            "rules": rules,
            "network_cases": network_cases,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(vexport.json_safe(manifest), indent=2, allow_nan=False),
            encoding="utf-8")
    csv_path = run_dir / "sweep_results.csv"
    todo = [d for d in dates if d not in done]
    logger.info("SWEEP start: %d days to solve (%d total), zone limit %s, "
                "network %s", len(todo), len(dates), zone_desc,
                "off" if args.skip_network else ", ".join(network_cases))

    day_arrays = vc.load_day_arrays(args.data)
    ev = lc_map = monitored = None
    if not args.skip_network:
        from network import elermorevale_openDSS as ev
        ev.build_elermorevale(args.glm_dir, args.common_dir,
                              skip_generators=True)
        ids = list(range(1, args.n_households + 1))
        lc_map = ev.map_customers_to_network_loads(
            ids, ev.get_network_load_names())
        monitored = ev.select_monitored_loads(lc_map, n_monitors=args.n_monitors)

    t_start = time.perf_counter()
    for k, date_iso in enumerate(todo, 1):
        t0 = time.perf_counter()
        households, _date, tariff = vc.assemble_ensemble(
            day_arrays, args.n_households, date_iso,
            mode=args.mode, e_max=args.e_max)
        cases, rows, envelope = solve_day(households, tariff, args,
                                          jes[date_iso],
                                          zone_limit_of(date_iso), rules)
        if not args.skip_network:
            net = network_day(ev, args, lc_map, monitored, households, cases,
                              date_iso, network_cases)
            for row in rows:
                row.update(net.get(row["case"], {}))
        for row in rows:
            row["date"] = date_iso
        frame = pd.DataFrame(rows)
        frame.to_csv(csv_path, mode="a", index=False,
                     header=not csv_path.is_file())
        by = {r["case"]: r for r in rows}
        logger.info("day %s (%d/%d, %.0f s): measured %.2f MW, excess %.0f kWh"
                    " | peak nobatt %.0f static %.0f doe %.0f kW | shortfall"
                    " doe %.0f, two-stage %s kWh",
                    date_iso, k, len(todo), time.perf_counter() - t0,
                    envelope["measured_peak_kw"] / MW_TO_KW,
                    envelope["measured_excess_kwh"],
                    by["nobatt"]["peak_kw"], by["static"]["peak_kw"],
                    by["doe"]["peak_kw"], by["doe"]["import_shortfall_kwh"],
                    {r: round(by[TWO_STAGE_PREFIX + r]["import_shortfall_kwh"])
                     for r in rules})
        if k % PROGRESS_EVERY == 0 or k == len(todo):
            elapsed = time.perf_counter() - t_start
            logger.info("SWEEP progress %d/%d days, %.1f min elapsed, "
                        "~%.1f min left", k, len(todo), elapsed / 60,
                        elapsed / k * (len(todo) - k) / 60)

    finish(run_dir, manifest)
    logger.info("SWEEP complete: %s", run_dir)


if __name__ == "__main__":
    main()
