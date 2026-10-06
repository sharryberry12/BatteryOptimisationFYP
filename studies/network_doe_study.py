"""
network_doe_study.py
====================

Network-aware operating envelopes on the Elermore Vale model for one day,
next to the regimes of static_vs_doe_replay.py:

  nobatt / static / doe   as in static_vs_doe_replay: 152 households, each
                          dispatch replicated over the network loads
  two_stage_<rule>        Method B rule slices of the same zone envelope
  network_doe_import      every network load is its own customer with its
                          own import cap per half-hour
                          (vpp/network_doe.py). The caps come from the
                          under-voltage limit at the monitored loads,
                          linearised at each half-hour's no-battery point
                          (network/voltage_sensitivity.py), plus the
                          substation-headroom row. Each load then solves
                          alone (soft HouseholdSolver); import above its
                          cap is reported as shortfall. Exports are not
                          capped.
  network_doe             the same import caps plus per-load export caps
                          from the over-voltage limit; PV above an export
                          cap is curtailed (and earns no feed-in credit).

The 152 households are dealt round-robin over the ~1,785 loads, so a
per-household cap would average a dozen locations; per-load customers
keep the location. Per-load results are reported at the scale of the
152-household ensemble (every household counts once, as the mean of its
copies) so the rows of summary.csv are comparable.

Only network_doe curtails PV: it is the only regime with per-customer
export caps. The 152-household regimes share one aggregate export cap
that the fleet rarely reaches; whatever export excess they report stays
in the flow, as in static_vs_doe_replay. Compare over-voltage counts and
savings of network_doe with that in mind; network_doe_import has no
export caps and is like-for-like with the other regimes.

The two network cases are separate because the model's no-load voltage
sits at 1.06-1.13 p.u.: the upper limit leaves little export headroom
anywhere, so the export caps are tight for reasons no dispatch can fix,
and would otherwise hide what the import caps do.

Artifacts land in outputs/runs/network_doe_<date>_<timestamp>/:
summary.csv, profiles.csv (every regime's feeder power across the day,
the two envelopes, the zone-transformer flow), manifest.json,
envelope.npz (per-load caps and, per case, dispatch, curtailment,
shortfall), voltages_<case>.npy, voltage_monitors.txt and
figures/{feeder_profile,envelope_profile,zone_transformer_power}.png.

Usage:
    python studies/network_doe_study.py                    # 2011-02-05
    python studies/network_doe_study.py --date 2011-04-08
    python studies/network_doe_study.py --no-zone-row --v-margin 0.005
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_CSV, GLM_COMMON, GLM_DIR, JESMOND_CSV, RUNS  # noqa: E402
from network import voltage_sensitivity as vs  # noqa: E402
from vpp import network_doe as nd  # noqa: E402
from vpp import vpp_common as vc  # noqa: E402
from vpp import vpp_export as vexport  # noqa: E402
from studies import battery_location_study as bl  # noqa: E402
from studies import doe_day_sweep as sweep  # noqa: E402
from studies import static_vs_doe_replay as sr  # noqa: E402

logger = logging.getLogger(__name__)

T = vc.T
DT = vc.DT
HOURS = sr.HOURS
NETWORK_DOE_IMPORT = "network_doe_import"
NETWORK_DOE = "network_doe"
C_TEAL, C_TEAL_DARK = "#1f9e89", "#0b5d52"
STYLE = {**sr.CASE_STYLE,
         NETWORK_DOE_IMPORT: ("network-aware DOE (import caps)", C_TEAL),
         NETWORK_DOE: ("network-aware DOE (import + export caps)",
                       C_TEAL_DARK)}
LABELS = {name: style[0] for name, style in STYLE.items()}
SLACK_TOL_KW = 1e-6          # solver noise below this is not a breach
CAP_BAND = (10, 90)          # percentiles of the per-load caps drawn


@dataclass(frozen=True)
class DayInputs:
    households: list
    date_iso: str
    tariff: np.ndarray
    jes_kw: np.ndarray        # (T,) measured zone-substation load
    p_base: np.ndarray        # (T,) no-battery ensemble load
    zone_limit_kw: float
    d_max: np.ndarray         # (T,) zone-headroom import limit, ensemble


@dataclass(frozen=True)
class LoadPoints:
    loads: tuple              # network load names, row order of (J, T) arrays
    owner: np.ndarray         # (J,) index of the household behind each load
    lc_map: dict              # load -> pseudo customer id 1..N
    monitored: tuple
    scale: float              # loads per household, on average
    weight: np.ndarray        # (J,) 1 / copies of the load's household


# ==========================================================
# INPUTS
# ==========================================================

def load_day(args) -> DayInputs:
    jes_kw = sr.load_jesmond_day(args.jesmond, args.date)
    households, date_iso, tariff = vc.assemble_ensemble(
        vc.load_day_arrays(args.data), args.n_households, args.date,
        mode=args.mode, e_max=args.e_max)
    p_base = np.sum([hh.net for hh in households], axis=0)
    zone_limit_kw = (args.zone_limit_mw * sr.MW_TO_KW if args.zone_limit_mw
                     else args.zone_limit_frac * float(jes_kw.max()))
    d_max = sr.zone_headroom_envelope(
        jes_kw, p_base, zone_limit_kw,
        args.feeder_cap_frac * float(p_base.max()))
    return DayInputs(households, date_iso, tariff, jes_kw, p_base,
                     zone_limit_kw, d_max)


def map_load_points(ev, n_households: int, args) -> LoadPoints:
    """Build the circuit and deal the households over its loads."""
    ev.build_elermorevale(args.glm_dir, args.common_dir,
                          skip_generators=True, oltc=ev.OLTC_ACTIVE)
    loads = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads(
        list(range(1, n_households + 1)), loads)
    monitored = ev.select_monitored_loads(lc_map, n_monitors=args.n_monitors)
    owner = np.array([lc_map[name] - 1 for name in loads], dtype=int)
    return LoadPoints(tuple(loads), owner, lc_map, tuple(monitored),
                      len(loads) / n_households, household_weights(owner))


def household_weights(owner: np.ndarray) -> np.ndarray:
    """
    Weight of each load when reporting at ensemble scale: 1 / the number
    of loads its household is dealt to (11 or 12 on Elermore Vale), so
    every household counts once.
    """
    return 1.0 / np.bincount(owner)[owner]


# ==========================================================
# ENVELOPE AND PER-LOAD DISPATCH
# ==========================================================

def day_envelope(ev, points: LoadPoints, net: np.ndarray,
                 zone_cap_kw: Optional[np.ndarray], args
                 ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Per-load caps (hi, lo), each (J, T), and one stats dict per interval.
    The circuit of map_load_points() must still be loaded in the engine.
    """
    hi, lo, stats = np.empty_like(net), np.empty_like(net), []
    v_lo = ev.V_LOWER_PU + args.v_margin
    v_hi = ev.V_UPPER_PU - args.v_margin
    for k in range(T):
        sens = vs.voltage_sensitivity(ev, dict(zip(points.loads, net[:, k])),
                                      points.monitored,
                                      delta_kw=args.delta_kw)
        hi_t, lo_t = nd.envelope_targets(net[:, k], vc.P_MAX)
        env = nd.interval_envelope(
            sens.dv_dp, sens.v0_pu, net[:, k], hi_t, lo_t, v_lo, v_hi,
            zone_cap_kw=None if zone_cap_kw is None else float(zone_cap_kw[k]),
            cross_phase=args.cross_phase)
        hi[:, k], lo[:, k] = env.hi, env.lo
        stats.append({
            "hour": float(HOURS[k]), "status": env.status,
            "n_rows": env.n_rows, "n_binding": env.n_binding,
            "n_unfixable": env.n_unfixable, "shrink": env.shrink,
            "v0_min_pu": float(sens.v0_pu.min()),
            "v0_max_pu": float(sens.v0_pu.max()),
            "import_cap_mean_kw": float(env.hi.mean()),
            "import_cap_cut_kw": float((hi_t - env.hi).mean()),
            "export_cap_cut_kw": float((env.lo - lo_t).mean()),
        })
        logger.info("%04.1f h: %d rows, %d binding, %d unfixable; mean "
                    "import cap %.2f kW (cut %.2f), export cut %.2f",
                    HOURS[k], env.n_rows, env.n_binding, env.n_unfixable,
                    env.hi.mean(), stats[-1]["import_cap_cut_kw"],
                    stats[-1]["export_cap_cut_kw"])
    return hi, lo, stats


def dispatch_loads(households: Sequence, owner: np.ndarray, hi: np.ndarray,
                   lo: np.ndarray) -> Tuple[np.ndarray, int]:
    """Every load solves alone under its own caps -> (B (J, T), n_failed)."""
    B, n_failed = np.zeros(hi.shape), 0
    for j, i in enumerate(owner):
        solver = vc.HouseholdSolver(households[i], d_min=lo[j], d_max=hi[j],
                                    soft=True)
        b, status = solver.solve()
        if "solved" not in status:
            n_failed += 1
        B[j] = b
    return B, n_failed


def realised_flows(net: np.ndarray, pv: np.ndarray, B: np.ndarray,
                   hi: np.ndarray, lo: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (grid, curtail, shortfall) of a dispatch against its caps. Export
    beyond the cap is PV the inverter curtails (at most the PV there is);
    import beyond the cap cannot be shed and stays in the flow.
    """
    raw = net - B
    curtail = np.clip(lo - raw, 0.0, np.maximum(pv, 0.0))
    shortfall = np.maximum(raw - hi, 0.0)
    return raw + curtail, curtail, shortfall


def load_savings(households: Sequence, owner: np.ndarray, B: np.ndarray,
                 curtail: np.ndarray, tariff: np.ndarray,
                 mode: str) -> np.ndarray:
    """$ per day vs no battery for each load; curtailed PV earns nothing."""
    out = np.empty(len(owner))
    for j, i in enumerate(owner):
        hh = households[i]
        no_batt = vc.base.bill(hh.load, hh.pv, np.zeros(T), tariff, mode)
        out[j] = no_batt - vc.base.bill(hh.load, hh.pv - curtail[j], B[j],
                                        tariff, mode)
    return out


def load_profiles(households: Sequence, owner: np.ndarray,
                  loads: Sequence[str], B: np.ndarray, curtail: np.ndarray,
                  date_iso: str) -> Tuple[dict, dict]:
    """
    (load map, profiles) for the network stage with one profile per load:
    the map sends each load to itself, so attach_shapes() needs no change.
    """
    profiles = {}
    for j, name in enumerate(loads):
        hh = households[owner[j]]
        pv = hh.pv - curtail[j]
        profiles[name] = [{
            "date": date_iso,
            "load": np.asarray(hh.load, dtype=float),
            "pv": pv,
            "battery": B[j].copy(),
            "grid": hh.load - pv - B[j],
            "soc": vc.SOC_INIT_FRAC * hh.e_max - np.cumsum(B[j]) * DT,
            "savings": 0.0,
        }]
    return {name: name for name in loads}, profiles


def dispatch_case(day: DayInputs, owner: np.ndarray, net: np.ndarray,
                  pv: np.ndarray, hi: np.ndarray, lo: np.ndarray,
                  mode: str) -> dict:
    """Per-load dispatch under the caps (hi, lo) and what it realises."""
    B, n_failed = dispatch_loads(day.households, owner, hi, lo)
    grid, curtail, shortfall = realised_flows(net, pv, B, hi, lo)
    return dict(B=B, net=net, grid=grid, curtail=curtail,
                shortfall=shortfall, n_failed=n_failed,
                # export still beyond the cap after curtailing all the PV
                breach=np.maximum(lo - grid, 0.0),
                savings=load_savings(day.households, owner, B, curtail,
                                     day.tariff, mode))


def network_doe_cases(ev, args, day: DayInputs,
                      points: LoadPoints) -> Tuple[dict, dict]:
    """
    The day's envelope and the two per-load regimes built on it.
    Returns ({case: dispatch_case dict}, envelope dict).
    """
    net = np.vstack([day.households[i].net for i in points.owner])
    pv = np.vstack([day.households[i].pv for i in points.owner])
    # the row caps the plain sum of the import caps, the report weights
    # households equally: the two differ by the 11-vs-12 copies, a few %
    zone_cap = (None if args.no_zone_row
                else np.maximum(points.scale * day.d_max, 0.0))
    t0 = time.time()
    hi, lo, stats = day_envelope(ev, points, net, zone_cap, args)
    t1 = time.time()
    _, lo_free = nd.envelope_targets(net, vc.P_MAX)
    cases = {
        NETWORK_DOE_IMPORT: dispatch_case(day, points.owner, net, pv, hi,
                                          lo_free, args.mode),
        NETWORK_DOE: dispatch_case(day, points.owner, net, pv, hi, lo,
                                   args.mode),
    }
    t2 = time.time()
    logger.info("network DOE: envelope %.0f s, 2 x %d load solves %.0f s, "
                "%d failed", t1 - t0, len(points.owner), t2 - t1,
                sum(c["n_failed"] for c in cases.values()))
    envelope = dict(hi=hi, lo=lo, stats=stats,
                    timing_s={"envelope": t1 - t0, "dispatch": t2 - t1})
    return cases, envelope


# ==========================================================
# METRICS
# ==========================================================

def summary_row(name: str, label: str, n_customers: int, agg_kw: np.ndarray,
                base_kw: np.ndarray, jes_kw: np.ndarray, zone_limit_kw: float,
                savings: np.ndarray, shortfall_kwh: float, excess_kwh: float,
                curtailed_kwh: float, n_failed: int) -> dict:
    """
    One regime at ensemble scale: agg_kw / base_kw are the regime's and
    the no-battery feeder profiles of the households, savings is $ per
    day per household, the energies are fleet totals.
    """
    zone = jes_kw - base_kw + agg_kw
    k_peak = int(np.argmax(agg_kw))
    return {
        "case": name,
        "label": label,
        "n_customers": int(n_customers),
        "savings_per_day": float(np.sum(savings)),
        "jain_savings": vc.jain_index(savings),
        "peak_kw": float(agg_kw[k_peak]),
        "peak_hour": float(HOURS[k_peak]),
        "zone_peak_kw": float(zone.max()),
        "zone_exceed_kwh": float(np.maximum(zone - zone_limit_kw, 0.0).sum()
                                 * DT),
        "import_shortfall_kwh": float(shortfall_kwh),
        "export_excess_kwh": float(excess_kwh),
        "curtailed_kwh": float(curtailed_kwh),
        "n_failed": int(n_failed),
    }


def ensemble_rows(day: DayInputs, cases: dict, mode: str) -> list:
    rows = []
    for name, case in cases.items():
        m = sr.case_metrics(name, day.households, case, day.tariff, mode,
                            day.jes_kw, day.p_base, day.zone_limit_kw)
        rows.append(summary_row(
            name, LABELS.get(name, name), len(day.households),
            vc.aggregate_pi(day.households, case["B"]), day.p_base,
            day.jes_kw, day.zone_limit_kw,
            vc.savings_vector(day.households, case["B"], day.tariff, mode),
            m["import_shortfall_kwh"], m["export_excess_kwh"], 0.0,
            m["n_failed"]))
    return rows


def load_case_row(name: str, n_households: int, jes_kw: np.ndarray,
                  zone_limit_kw: float, points: LoadPoints,
                  case: dict) -> dict:
    """
    A per-load regime at ensemble scale: every household counts once, as
    the mean of its copies, so the row is comparable with ensemble_rows.
    """
    w = points.weight

    def fleet_kwh(kw: np.ndarray) -> float:
        return float(w @ np.where(kw > SLACK_TOL_KW, kw, 0.0).sum(axis=1)
                     * DT)

    savings = np.bincount(points.owner, weights=w * case["savings"],
                          minlength=n_households)
    return summary_row(
        name, LABELS[name], len(points.owner), w @ case["grid"],
        w @ case["net"], jes_kw, zone_limit_kw, savings,
        fleet_kwh(case["shortfall"]),
        fleet_kwh(case["curtail"] + case["breach"]),
        fleet_kwh(case["curtail"]), case["n_failed"])


def network_columns(result: dict) -> dict:
    tx = np.asarray(result["tx_p_kw"], dtype=float)
    return {
        "tx_peak_kw": float(tx.max()),
        "tx_peak_hour": float(HOURS[int(np.argmax(tx))]),
        "loss_kwh": float(np.sum(result["loss_kw"]) * DT),
        "v_min_pu": float(result["v_min_pu"]),
        "v_max_pu": float(result["v_max_pu"]),
        "n_under": int(result["n_under"]),
        "n_over": int(result["n_over"]),
        "total_points": int(result["total_points"]),
    }


# ==========================================================
# NETWORK STAGE AND ARTIFACTS
# ==========================================================

def replay_all(ev, args, day: DayInputs, points: LoadPoints, cases: dict,
               load_cases: dict, run_dir: Path) -> dict:
    """Every regime through the model -> {case: simulate_case result}."""
    monitored = list(points.monitored)
    jobs = {name: (points.lc_map, sweep.profiles_from_dispatch(
                day.households, case["B"], day.date_iso))
            for name, case in cases.items()}
    jobs.update({name: load_profiles(
                     day.households, points.owner, points.loads, case["B"],
                     case["curtail"], day.date_iso)
                 for name, case in load_cases.items()})
    out = {}
    for name, (lc_map, profiles) in jobs.items():
        logger.info("simulating %r ...", name)
        result = bl.simulate_case(ev, args, lc_map, monitored, profiles, 0)
        sr.save_voltages(run_dir, name, result, monitored)
        out[name] = result
    (run_dir / "voltage_monitors.txt").write_text(
        "\n".join(monitored) + "\n", encoding="utf-8")
    return out


def profiles_frame(day: DayInputs, points: LoadPoints, aggs: dict,
                   envelope: dict, results: dict) -> pd.DataFrame:
    """
    The day at ensemble scale, one row per half-hour: the measured
    substation load, the zone-headroom import limit, the network-aware
    caps summed over the households, every regime's feeder power
    (p_<case>_kw) and zone-transformer flow (tx_<case>_kw).
    """
    w = points.weight
    frame = {"hour": HOURS, "jes_kw": day.jes_kw, "d_max_zone_kw": day.d_max,
             "import_cap_sum_kw": w @ envelope["hi"],
             "export_cap_sum_kw": w @ envelope["lo"]}
    frame.update({f"p_{name}_kw": np.asarray(agg, dtype=float)
                  for name, agg in aggs.items()})
    frame.update({f"tx_{name}_kw": np.asarray(r["tx_p_kw"], dtype=float)
                  for name, r in results.items()})
    return pd.DataFrame(frame)


# ==========================================================
# FIGURES
# ==========================================================

REFERENCES = ("nobatt", "static")      # drawn faint in every panel


def _methods_in(frame: pd.DataFrame, prefix: str) -> list:
    """Regimes of the frame that get a panel of their own, in STYLE order."""
    return [name for name in STYLE
            if name not in REFERENCES and f"{prefix}{name}_kw" in frame]


def _panels(n: int, title: str):
    """A grid of n shared-axis panels, two per row; (fig, axes)."""
    ncols = 2 if n > 1 else 1
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.1 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig, axes[:n]


def _compare_panel(ax, frame: pd.DataFrame, prefix: str, name: str,
                   scale: float, stress: np.ndarray) -> np.ndarray:
    """One regime bold against the faint reference regimes; returns its y."""
    sr._shade_stress(ax, stress)
    for ref in REFERENCES:
        col = f"{prefix}{ref}_kw"
        if col in frame:
            label, color = STYLE[ref]
            ax.plot(HOURS, frame[col].to_numpy() / scale, color=color,
                    lw=1.4, alpha=0.85, label=label)
    label, color = STYLE[name]
    y = frame[f"{prefix}{name}_kw"].to_numpy() / scale
    ax.plot(HOURS, y, color=color, lw=2.4, label=label)
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
    ax.axhline(0, color=sr.MUTED, lw=0.6)
    ax.grid(axis="x", visible=False)
    return y


def _finish_panels(fig, axes, ylabel: str, fig_dir: Path, name: str) -> None:
    ncols = 2 if len(axes) > 1 else 1
    for i, ax in enumerate(axes):
        ax.legend(fontsize=7.5, loc="upper left")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i >= len(axes) - ncols:
            sr._hour_axis(ax)
        else:
            ax.set_xlim(0, 24)
    sr._save(fig, fig_dir, name)


def fig_feeder_profile(frame: pd.DataFrame, date_iso: str,
                       zone_limit_kw: float, n_hh: int, fig_dir: Path) -> None:
    """Each regime's feeder power against no battery and static limits."""
    methods = _methods_in(frame, "p_")
    stress = frame["jes_kw"].to_numpy() > zone_limit_kw
    fig, axes = _panels(len(methods),
                        f"Feeder power by regime, each against no battery "
                        f"and static limits — {date_iso}, N={n_hh} households")
    for ax, name in zip(axes, methods):
        y = _compare_panel(ax, frame, "p_", name, 1.0, stress)
        ax.step(HOURS, frame["d_max_zone_kw"], where="post", color=sr.INK,
                lw=1, alpha=0.6, label="zone-headroom import limit")
        if name in (NETWORK_DOE_IMPORT, NETWORK_DOE):
            ax.step(HOURS, frame["import_cap_sum_kw"], where="post",
                    color=C_TEAL, lw=1.2, ls="--",
                    label="network-aware import caps, summed")
        k = int(np.argmax(y))
        ax.annotate(f"peak {y[k]:.0f} kW at {HOURS[k]:04.1f} h",
                    xy=(HOURS[k], y[k]), xytext=(-110, -18),
                    textcoords="offset points", fontsize=8, color=sr.INK_2,
                    arrowprops=dict(arrowstyle="-", color=sr.MUTED, lw=0.8))
    _finish_panels(fig, axes, "aggregate grid power  Σ p_ik  (kW, +import)",
                   fig_dir, "feeder_profile")


def fig_envelope_profile(hi: np.ndarray, lo: np.ndarray, net: np.ndarray,
                         day: DayInputs, fig_dir: Path) -> None:
    """Spread of the per-load caps over the day against the loads' own flows."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    stress = day.jes_kw > day.zone_limit_kw
    panels = ((ax1, hi, np.maximum(net, 0.0), "import cap (kW)",
               "own import with no battery"),
              (ax2, lo, np.minimum(net, 0.0), "export cap (kW)",
               "own export with no battery"))
    for ax, caps, own, ylabel, own_label in panels:
        sr._shade_stress(ax, stress)
        p_lo, p_hi = np.percentile(caps, CAP_BAND, axis=0)
        ax.fill_between(HOURS, p_lo, p_hi, step="post", color=C_TEAL,
                        alpha=0.18, linewidth=0,
                        label=f"{CAP_BAND[0]}–{CAP_BAND[1]} % of loads")
        ax.step(HOURS, np.median(caps, axis=0), where="post", color=C_TEAL,
                lw=2, label="median load")
        ax.plot(HOURS, own.mean(axis=0), color=sr.MUTED, lw=1.5, ls=":",
                label=f"{own_label} (mean)")
        ax.axhline(0, color=sr.MUTED, lw=0.6)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="x", visible=False)
    ax1.set_title(f"Per-load caps across the day — {day.date_iso}, "
                  f"{hi.shape[0]} loads", loc="left", fontweight="bold")
    sr._hour_axis(ax2)
    sr._save(fig, fig_dir, "envelope_profile")


def fig_zone_transformer(frame: pd.DataFrame, date_iso: str,
                         zone_limit_kw: float, n_hh: int, n_loads: int,
                         scale: float, fig_dir: Path) -> None:
    """Each regime's zone-transformer flow against the references."""
    methods = _methods_in(frame, "tx_")
    stress = frame["jes_kw"].to_numpy() > zone_limit_kw
    fig, axes = _panels(
        len(methods),
        f"Power through the 132/11 kV zone transformer, each regime against "
        f"no battery and static limits — {date_iso}\nmodel = {n_hh} household "
        f"profiles over {n_loads} feeder loads (×{scale:.1f}); the measurement "
        "is the whole substation: compare shapes, not levels")
    jes_mw = frame["jes_kw"].to_numpy() / sr.MW_TO_KW
    for ax, name in zip(axes, methods):
        _compare_panel(ax, frame, "tx_", name, sr.MW_TO_KW, stress)
        ax.plot(HOURS, jes_mw, color=sr.INK_2, lw=1.4, ls="--",
                label="measured: Jesmond 132/11 kV (all feeders)")
        ax.axhline(zone_limit_kw / sr.MW_TO_KW, color=sr.C_RED, lw=1, ls="--",
                   label="zone limit C")
    _finish_panels(fig, axes, "zone transformer P (MW, +import)", fig_dir,
                   "zone_transformer_power")


def make_figures(run_dir: Path, frame: pd.DataFrame, day: DayInputs,
                 points: LoadPoints, envelope: dict, net: np.ndarray) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_hh = len(day.households)
    fig_feeder_profile(frame, day.date_iso, day.zone_limit_kw, n_hh, fig_dir)
    fig_envelope_profile(envelope["hi"], envelope["lo"], net, day, fig_dir)
    fig_zone_transformer(frame, day.date_iso, day.zone_limit_kw, n_hh,
                         len(points.loads), points.scale, fig_dir)


def refigure(run_dir: Path) -> None:
    """
    Redraw the two profile figures of an existing run from its
    profiles.csv and manifest.json: no data, solver or power flow needed.
    (envelope_profile.png needs the per-load net flows and is not redrawn.)
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    frame = pd.read_csv(run_dir / "profiles.csv")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    date_iso = manifest["ensemble"]["date"]
    n_hh = int(manifest["ensemble"]["n_households"])
    zone_limit_kw = float(manifest["jesmond"]["zone_limit_kw"])
    network = manifest["network"]
    fig_feeder_profile(frame, date_iso, zone_limit_kw, n_hh, fig_dir)
    fig_zone_transformer(frame, date_iso, zone_limit_kw, n_hh,
                         int(network["n_loads"]), float(network["scale"]),
                         fig_dir)


# ==========================================================
# ARTIFACTS
# ==========================================================

def write_run(run_dir: Path, args, day: DayInputs, points: LoadPoints,
              load_cases: dict, envelope: dict, rows: list,
              profiles: pd.DataFrame) -> None:
    per_case = {f"{key}_{name}": case[field]
                for name, case in load_cases.items()
                for key, field in (("battery", "B"), ("curtail", "curtail"),
                                   ("shortfall", "shortfall"))}
    np.savez_compressed(
        run_dir / "envelope.npz", loads=np.array(points.loads),
        owner=points.owner, hi=envelope["hi"], lo=envelope["lo"], **per_case)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    profiles.to_csv(run_dir / "profiles.csv", index=False)
    manifest = {
        "kind": "network_doe_study",
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "ensemble": {"n_households": len(day.households),
                     "date": day.date_iso, "mode": args.mode,
                     "e_max_kwh": args.e_max},
        "network": {"n_loads": len(points.loads), "scale": points.scale,
                    "n_monitors": len(points.monitored)},
        "jesmond": {"peak_kw": float(day.jes_kw.max()),
                    "zone_limit_kw": day.zone_limit_kw},
        "envelope": {"zone_row": not args.no_zone_row,
                     "cross_phase": args.cross_phase,
                     "v_margin_pu": args.v_margin,
                     "d_max_zone_kw": day.d_max,
                     "intervals": envelope["stats"]},
        "timing_s": envelope["timing_s"],
        "summary": rows,
        "files": {"summary": "summary.csv", "profiles": "profiles.csv",
                  "envelope": "envelope.npz",
                  "voltages": {name: f"voltages_{name}.npy"
                               for name in (row["case"] for row in rows)},
                  "voltage_monitors": "voltage_monitors.txt"},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(vexport.json_safe(manifest), indent=2, allow_nan=False),
        encoding="utf-8")


# ==========================================================
# MAIN
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Network-aware per-load operating envelopes vs static "
                    "limits and the zone-headroom DOE on one day",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=str(DATA_CSV), help="Ausgrid customer CSV")
    p.add_argument("--jesmond", default=str(JESMOND_CSV),
                   help="Ausgrid zone-substation CSV (15-min MW)")
    p.add_argument("--date", default=sr.DEFAULT_DATE, help="ISO day")
    p.add_argument("--n-households", type=int, default=152)
    p.add_argument("--mode", choices=["fit", "net"], default="fit")
    p.add_argument("--e-max", type=float, default=vc.E_MAX_DEFAULT,
                   help="Battery capacity per household (kWh)")
    p.add_argument("--export-limit", type=float, default=1.5,
                   help="Flat export cap per household (kW) of the static "
                        "and zone-DOE regimes")
    p.add_argument("--zone-limit-frac", type=float, default=0.95,
                   help="Zone limit as a fraction of the day's measured peak")
    p.add_argument("--zone-limit-mw", type=float, default=None,
                   help="Absolute zone limit (MW); overrides the fraction")
    p.add_argument("--feeder-cap-frac", type=float, default=1.0,
                   help="Feeder cap as a fraction of the no-battery peak")
    p.add_argument("--penalty", type=float, default=1e3,
                   help="Soft-envelope slack penalty of the ensemble solves")
    p.add_argument("--two-stage-rules", default="maxmin",
                   help="Method B rules to include, comma-separated; "
                        "empty = none")
    p.add_argument("--no-zone-row", action="store_true",
                   help="Network DOE from the voltage rows only")
    p.add_argument("--cross-phase", action="store_true",
                   help="Keep the cross-phase terms: caps safe for every "
                        "combination of customers, far tighter")
    p.add_argument("--v-margin", type=float, default=0.0,
                   help="Tighten both voltage limits by this much (p.u.) "
                        "to cover linearisation error")
    p.add_argument("--delta-kw", type=float, default=vs.DELTA_KW_DEFAULT,
                   help="Perturbation step of the voltage sensitivities")
    p.add_argument("--n-monitors", type=int, default=100)
    p.add_argument("--glm-dir", default=str(GLM_DIR))
    p.add_argument("--common-dir", default=str(GLM_COMMON))
    p.add_argument("--runs-root", default=str(RUNS))
    p.add_argument("--refigure", metavar="RUN_DIR", default=None,
                   help="Only redraw the feeder-profile and zone-transformer "
                        "figures of this existing run folder, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.refigure:
        refigure(Path(args.refigure))
        return
    day = load_day(args)
    rules = [r.strip() for r in args.two_stage_rules.split(",") if r.strip()]
    cases = sr.solve_cases(day.households, args.export_limit, day.d_max,
                           args.penalty, two_stage_rules=rules)

    from network import elermorevale_openDSS as ev
    points = map_load_points(ev, len(day.households), args)
    logger.info("%d loads / %d households -> x%.2f; %d monitors",
                len(points.loads), len(day.households), points.scale,
                len(points.monitored))
    load_cases, envelope = network_doe_cases(ev, args, day, points)

    run_dir = Path(args.runs_root) / (
        f"network_doe_{day.date_iso}_{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = ensemble_rows(day, cases, args.mode) + [
        load_case_row(name, len(day.households), day.jes_kw,
                      day.zone_limit_kw, points, case)
        for name, case in load_cases.items()]
    results = replay_all(ev, args, day, points, cases, load_cases, run_dir)
    rows = [{**row, **network_columns(results[row["case"]])} for row in rows]
    aggs = {**{name: vc.aggregate_pi(day.households, case["B"])
               for name, case in cases.items()},
            **{name: points.weight @ case["grid"]
               for name, case in load_cases.items()}}
    profiles = profiles_frame(day, points, aggs, envelope, results)
    write_run(run_dir, args, day, points, load_cases, envelope, rows, profiles)
    make_figures(run_dir, profiles, day, points, envelope,
                 load_cases[NETWORK_DOE]["net"])

    cols = ["label", "savings_per_day", "jain_savings", "peak_kw",
            "zone_exceed_kwh", "import_shortfall_kwh", "curtailed_kwh",
            "tx_peak_kw", "v_min_pu", "v_max_pu", "n_under", "n_over"]
    table = pd.DataFrame(rows)[cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}")
    logger.info("Network-aware DOE on %s: %d loads, %d monitors\n%s\n"
                "Artifacts: %s", day.date_iso, len(points.loads),
                len(points.monitored), table, run_dir)


if __name__ == "__main__":
    main()
