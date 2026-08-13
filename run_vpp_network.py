"""
run_vpp_network.py
==================

End-to-end VPP -> OpenDSS pipeline (PIPELINE_DESIGN.md): one command
builds the household ensemble, solves the coupled dispatch with the
chosen vpp/ method, exports the three dispatch scenarios (no-battery /
uncoupled / VPP-coupled) as profiles CSVs, injects them into the
Elermore Vale OpenDSS model, and reports the physical network impact.

Stages:
    1. ensemble setup      (vpp_common.setup_ensemble)
    2. method dispatch     (vpp/vpp_registry.py)
    3. artifact export     (vpp/vpp_export.py -> runs/<run_id>/)
    4. network simulation  (elermorevale_openDSS, one run per scenario)
    5. reporting           (3-way figures + network_summary.csv)

Figures and CSVs land in runs/<method>_<scenario>_<date>_<timestamp>/.
Every run is reproducible from its manifest.json.

Examples (repo root):
    python run_vpp_network.py sharing_admm --data data1.csv
    python run_vpp_network.py admm --rho 5 --n-households 40 --data data1.csv
    python run_vpp_network.py centralised_qp --soft --scenario tight_tou \
        --data data1.csv
    python run_vpp_network.py fcas --skip-network --data data1.csv
    python run_vpp_network.py resume --run-dir runs/sharing_admm_static_...

The method subcommands accept short aliases: centralised, two_stage,
dual, admm, price, fcas.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # the pipeline always saves figures, never shows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
VPP_DIR = REPO_ROOT / "vpp"
for _p in (str(REPO_ROOT), str(VPP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vpp_common as vc  # noqa: E402
import vpp_export as vexport  # noqa: E402
import vpp_registry as vreg  # noqa: E402

logger = logging.getLogger("pipeline")

SCEN_STYLE = {
    "nobatt": ("no battery", "tab:gray"),
    "uncoupled": ("uncoupled QP", "tab:red"),
    "coupled": ("VPP coupled", "tab:blue"),
}


# ==========================================================
# CLI
# ==========================================================

def build_parser():
    shared = argparse.ArgumentParser(add_help=False)
    g = shared.add_argument_group("ensemble (stage 1)")
    g.add_argument("--n-households", type=int, default=20,
                   help="Ensemble size N (profiles replicate beyond the "
                        "clean customer pool)")
    g.add_argument("--date", default=None,
                   help="ISO day to simulate (default: max-PV common day)")
    g.add_argument("--mode", choices=["fit", "net"], default="fit",
                   help="Billing topology for savings reporting")
    g.add_argument("--e-max", type=float, default=vc.E_MAX_DEFAULT,
                   help="Battery capacity per household (kWh)")
    g.add_argument("--data", default=str(vc.DATA_CSV),
                   help="Path to the Ausgrid CSV")
    g.add_argument("--scenario", default="static",
                   choices=["none", "static", "tight_tou", "dynamic_solar"],
                   help="Feeder envelope scenario")
    g.add_argument("--export-limit", type=float, default=1.5,
                   help="Per-household export base for the envelope (kW)")
    g.add_argument("--import-limit", type=float, default=np.inf,
                   help="Per-household import base for the envelope (kW)")

    network = argparse.ArgumentParser(add_help=False)
    n = network.add_argument_group("network (stages 4-5)")
    n.add_argument("--network", choices=["elermorevale"],
                   default="elermorevale",
                   help="Network backend (Elermore Vale only for now)")
    n.add_argument("--glm-dir", default=str(REPO_ROOT / "Elermorevale"),
                   help="GridLAB-D source directory")
    n.add_argument("--common-dir", default=str(REPO_ROOT / "common"),
                   help="Shared GLM includes directory")
    n.add_argument("--n-monitors", type=int, default=100,
                   help="Number of voltage-monitored loads")
    n.add_argument("--skip-network", action="store_true",
                   help="Stop after stage 3 (export only); resume later "
                        "with the 'resume' subcommand")
    n.add_argument("--runs-root", default=str(REPO_ROOT / "runs"),
                   help="Parent directory for run artifacts")

    parser = argparse.ArgumentParser(
        prog="run_vpp_network.py",
        description="End-to-end VPP -> OpenDSS pipeline "
                    "(see PIPELINE_DESIGN.md)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'run_vpp_network.py <method> --help' for "
               "method-specific flags.")
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="method")
    for name, spec in vreg.REGISTRY.items():
        sp = sub.add_parser(name, parents=[shared, network],
                            aliases=list(spec.aliases),
                            help=spec.description,
                            description=spec.description)
        spec.add_args(sp)
        sp.set_defaults(method=name)
    rp = sub.add_parser("resume", parents=[network],
                        help="Re-run stages 4-5 from an existing run "
                             "directory (skips the VPP solve)",
                        description="Re-run the network simulation and "
                                    "reporting from previously exported "
                                    "dispatch artifacts.")
    rp.add_argument("--run-dir", required=True,
                    help="Run directory containing manifest.json")
    rp.set_defaults(method="resume")
    return parser


# ==========================================================
# STAGES 1-3
# ==========================================================

def solve_and_export(args):
    spec = vreg.REGISTRY[args.method]

    logger.info("=== Stage 1: ensemble setup ===")
    households, date_iso, tariff, d_min, d_max = vc.setup_ensemble(args)
    ctx = vreg.EnsembleContext(
        households=households, d_min=d_min, d_max=d_max,
        tariff=tariff, date_iso=date_iso, mode=args.mode)
    logger.info("day %s, N=%d, envelope scenario %r",
                date_iso, len(households), args.scenario)

    logger.info("=== Stage 2: method dispatch (%s) ===", spec.name)
    dispatch = spec.run(ctx, args)
    logger.info("solved in %.2f s, converged=%s, iterations=%s",
                dispatch.solve_time, dispatch.converged,
                dispatch.iterations)
    if not dispatch.converged:
        logger.warning(
            "method reported a NON-converged dispatch -- it is still "
            "exported (a result, not an error); details in the manifest")
    n_bad = vc.validate_ensemble(households, dispatch.B)
    if n_bad:
        logger.warning("%d household dispatches violate the Section 10 "
                       "invariants; check the method parameters", n_bad)

    logger.info("=== Stage 3: artifact export ===")
    run_id = (f"{spec.name}_{args.scenario}_{date_iso}_"
              f"{datetime.now():%Y%m%d-%H%M%S}")
    run_dir = Path(args.runs_root) / run_id
    manifest = vexport.export_run(
        run_dir, households, tariff, args.mode, date_iso,
        d_min, d_max, dispatch, cli_args=vars(args))
    logger.info("run directory: %s", run_dir)
    return run_dir, manifest


# ==========================================================
# STAGE 4 -- NETWORK SIMULATION
# ==========================================================

def injected_aggregate(lc_map, profiles, day_idx):
    """Sum of the grid loadshapes actually attached to the network (kW)."""
    agg = np.zeros(vc.T)
    for _lname, cid in lc_map.items():
        days = profiles.get(cid, [])
        if day_idx < len(days):
            agg += days[day_idx]["grid"]
    return agg


def check_consistency(scen, manifest, result, injected, scale,
                      n_loads, n_customers):
    """
    Two checks per scenario (PIPELINE_DESIGN.md Section 3.4):
      1. bookkeeping -- the injected aggregate must equal
         scale x the VPP-level agg_pi (exact under uniform replication);
      2. physical -- the measured substation power should track the
         injected aggregate to within losses/voltage dependence.

    n_loads / n_customers come from the actual load map and CSV, not the
    manifest, so the check stays valid even if they disagree with the
    manifest (which is separately warned about).
    """
    agg_vpp = np.asarray(
        manifest["scenario_metrics"][scen]["agg_pi_kw"], dtype=float)
    uniform = n_loads % n_customers == 0

    map_err = float(np.abs(injected - scale * agg_vpp).max())
    if uniform and map_err > 1e-3:
        logger.warning("  %s: mapping consistency FAILED -- "
                       "|injected - scale*agg_pi| = %.4f kW", scen, map_err)
    tx = np.asarray(result["tx_p_kw"], dtype=float)
    phys_err = float(np.abs(tx - injected).max())
    denom = max(float(np.abs(injected).max()), 1.0)
    logger.info("  %s: injected peak %.0f kW | mapping err %.3f kW%s | "
                "max |tx - injected| %.0f kW (%.1f%%)",
                scen, np.abs(injected).max(), map_err,
                "" if uniform else " (non-uniform replication)",
                phys_err, 100.0 * phys_err / denom)
    # Losses are quadratic in flow, so peak intervals genuinely deviate
    # by tens of percent on a heavily loaded feeder; only a mismatch far
    # beyond plausible losses (or a sign flip, ~200%) is flagged.
    if phys_err / denom > 0.5:
        logger.warning("  %s: substation power deviates >50%% from the "
                       "injected aggregate -- likely a sign-convention "
                       "or convergence problem", scen)


def network_stage(run_dir, manifest, args):
    if args.network != "elermorevale":
        raise vreg.PipelineError(
            f"network backend {args.network!r} is not wired into the "
            "pipeline yet (elermorevale only)")
    import elermorevale_openDSS as ev

    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev.OUTPUT_DIR = str(fig_dir)   # ev plot helpers save instead of show

    date_iso = manifest["ensemble"]["date"]
    n_households = manifest["ensemble"]["n_households"]

    logger.info("=== Stage 4: OpenDSS simulation (%s) ===", args.network)
    profiles, day_idx = {}, {}
    for scen in vexport.SCENARIOS:
        csv_path = run_dir / manifest["files"][scen]
        profiles[scen] = ev.load_profiles_from_csv(str(csv_path))
        idx = ev.day_index_for_date(profiles[scen], date_iso)
        if idx is None:
            logger.warning("date %s not found in %s; falling back to "
                           "day 0", date_iso, csv_path.name)
            idx = 0
        day_idx[scen] = idx

    ids = sorted(profiles["coupled"].keys())
    if len(ids) != n_households:
        logger.warning("CSV has %d customers but manifest says N=%d",
                       len(ids), n_households)

    logger.info("building network to enumerate loads ...")
    ev.build_elermorevale(args.glm_dir, args.common_dir,
                          skip_generators=True)
    load_names = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads(ids, load_names)
    monitored = ev.select_monitored_loads(lc_map,
                                          n_monitors=args.n_monitors)
    scale = len(lc_map) / len(ids)
    logger.info("replication: %d loads / %d households -> x%.1f feeder "
                "scale", len(lc_map), len(ids), scale)

    results, injected = {}, {}
    for scen in vexport.SCENARIOS:
        logger.info("simulating scenario %r ...", scen)
        results[scen] = ev.simulate_scenario(
            args.glm_dir, args.common_dir, lc_map, monitored,
            profiles[scen], day_idx[scen])
        injected[scen] = injected_aggregate(lc_map, profiles[scen],
                                            day_idx[scen])
        check_consistency(scen, manifest, results[scen], injected[scen],
                          scale, len(lc_map), len(ids))

    report_stage(run_dir, fig_dir, manifest, results, injected, scale,
                 len(lc_map), len(ids), args, ev)


# ==========================================================
# STAGE 5 -- REPORTING
# ==========================================================

def _save_fig(fig, fig_dir, name):
    path = Path(fig_dir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure saved: %s", path)


def plot_voltage_envelope_3way(results, date_iso, fig_dir, ev):
    hours = np.arange(ev.T) * ev.DT
    fig, ax = plt.subplots(figsize=(10, 5))
    for scen, (label, color) in SCEN_STYLE.items():
        V = np.array(list(results[scen]["voltages"].values()))
        ax.fill_between(hours, V.min(axis=0), V.max(axis=0),
                        alpha=0.15, color=color)
        ax.plot(hours, V.min(axis=0), color=color, lw=0.9)
        ax.plot(hours, V.max(axis=0), color=color, lw=0.9, label=label)
    ax.axhline(ev.V_UPPER_PU, color="red", ls="--", lw=1,
               label="statutory limits")
    ax.axhline(ev.V_LOWER_PU, color="red", ls="--", lw=1)
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage envelope across monitored loads ({date_iso})")
    ax.set_xlim(0, 24)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    _save_fig(fig, fig_dir, "voltage_envelope_3way")


def plot_substation_power_3way(results, d_min_s, d_max_s, date_iso,
                               fig_dir, ev):
    hours = np.arange(ev.T) * ev.DT
    fig, ax = plt.subplots(figsize=(10, 5))
    for scen, (label, color) in SCEN_STYLE.items():
        ax.plot(hours, results[scen]["tx_p_kw"], color=color, lw=1.5,
                label=label)
    if np.isfinite(d_min_s).any():
        ax.plot(hours, d_min_s, "k--", lw=1.2,
                label="DOE envelope (feeder scale)")
    if np.isfinite(d_max_s).any():
        ax.plot(hours, d_max_s, "k--", lw=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Zone substation P (kW, +import)")
    ax.set_title(f"Measured feeder-head power vs DOE envelope "
                 f"({date_iso})")
    ax.set_xlim(0, 24)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save_fig(fig, fig_dir, "substation_power_3way")


def plot_delta_heatmap(results, date_iso, fig_dir, ev):
    from matplotlib.colors import TwoSlopeNorm
    names = sorted(results["uncoupled"]["voltages"])
    U = np.array([results["uncoupled"]["voltages"][n] for n in names])
    C = np.array([results["coupled"]["voltages"][n] for n in names])
    delta = U - C
    abs_max = max(abs(delta.min()), abs(delta.max()), 0.001)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(delta, aspect="auto", cmap="coolwarm", norm=norm,
                   extent=[0, 24, len(names), 0], interpolation="nearest")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Monitored load index")
    ax.set_title(f"Voltage delta: uncoupled − coupled ({date_iso})  "
                 f"[max |Δ| = {abs_max * 1000:.1f} mV/V]")
    plt.colorbar(im, ax=ax, label="Δ voltage (p.u.)", extend="both")
    _save_fig(fig, fig_dir, "voltage_delta_uncoupled_vs_coupled")


def plot_injected_vs_measured(results, injected, date_iso, fig_dir, ev):
    hours = np.arange(ev.T) * ev.DT
    fig, ax = plt.subplots(figsize=(10, 5))
    for scen, (label, color) in SCEN_STYLE.items():
        ax.plot(hours, injected[scen], ls="--", color=color, lw=1.0,
                label=f"{label}: injected")
        ax.plot(hours, results[scen]["tx_p_kw"], color=color, lw=1.5,
                label=f"{label}: measured")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Feeder-head P (kW)")
    ax.set_title(f"Injected loadshape aggregate vs measured substation "
                 f"power ({date_iso})")
    ax.set_xlim(0, 24)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    _save_fig(fig, fig_dir, "injected_vs_measured")


def build_summary(manifest, results, injected, d_min_s, d_max_s):
    rows = []
    for scen in vexport.SCENARIOS:
        r = results[scen]
        m = manifest["scenario_metrics"][scen]
        tx = np.asarray(r["tx_p_kw"], dtype=float)
        tx_viol = vc.envelope_violation(tx, d_min_s, d_max_s)
        rows.append({
            "scenario": scen,
            "label": SCEN_STYLE[scen][0],
            "v_min_pu": r["v_min_pu"],
            "v_max_pu": r["v_max_pu"],
            "n_violations": r["n_violations"],
            "total_points": r["total_points"],
            "peak_tx_kw": float(np.max(np.abs(tx))),
            "loss_kw": r["loss_kw"],
            "tx_envelope_exceed_kw": tx_viol["max_kw"],
            "tx_envelope_exceed_intervals": tx_viol["n_intervals"],
            "injected_peak_kw": float(np.max(np.abs(injected[scen]))),
            "vpp_objective": m["objective"],
            "vpp_envelope_exceed_kw": m["envelope_violation"]["max_kw"],
            "savings_total_per_day": m["savings_total"],
        })
    return rows


def report_stage(run_dir, fig_dir, manifest, results, injected, scale,
                 n_loads, n_customers, args, ev):
    logger.info("=== Stage 5: reporting ===")
    date_iso = manifest["ensemble"]["date"]
    d_min_s = vexport.envelope_array(
        manifest["envelope"]["d_min_kw"]) * scale
    d_max_s = vexport.envelope_array(
        manifest["envelope"]["d_max_kw"]) * scale
    scale_exact = n_loads % n_customers == 0

    summary = build_summary(manifest, results, injected, d_min_s, d_max_s)
    df = pd.DataFrame(summary)
    csv_path = run_dir / "network_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("summary written: %s", csv_path)

    plot_voltage_envelope_3way(results, date_iso, fig_dir, ev)
    plot_substation_power_3way(results, d_min_s, d_max_s, date_iso,
                               fig_dir, ev)
    for scen in vexport.SCENARIOS:
        ev.plot_voltage_heatmap(results[scen], date_iso,
                                title_prefix=f"{SCEN_STYLE[scen][0]}: ")
    plot_delta_heatmap(results, date_iso, fig_dir, ev)
    plot_injected_vs_measured(results, injected, date_iso, fig_dir, ev)

    manifest["network"] = {
        "backend": args.network,
        "n_loads": n_loads,
        "n_customers_in_csv": n_customers,
        "n_monitored": len(next(iter(results.values()))["voltages"]),
        "scale_factor": scale,
        "scale_exact": scale_exact,
        "scale_note": None if scale_exact else (
            "non-uniform replication (n_loads % n_customers != 0): the "
            "first n_loads mod n_customers households get one extra "
            "copy, so tx_envelope_exceed_* values carry a bias of up to "
            "one household profile near the envelope boundary"),
        "summary": summary,
        "completed": datetime.now().isoformat(timespec="seconds"),
    }
    vexport.save_manifest(run_dir, manifest)

    cols = ["scenario", "v_min_pu", "v_max_pu", "n_violations",
            "peak_tx_kw", "loss_kw", "tx_envelope_exceed_kw",
            "savings_total_per_day"]
    print("\n" + df[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:.3f}"))
    if not scale_exact:
        print(f"\nNote: replication is non-uniform ({n_loads} loads / "
              f"{n_customers} households); envelope-exceedance figures "
              "are approximate near the boundary (see manifest "
              "network.scale_note).")
    print(f"\nArtifacts: {run_dir}")


# ==========================================================
# MAIN
# ==========================================================

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.method == "resume":
        run_dir = Path(args.run_dir)
        manifest = vexport.load_manifest(run_dir)
        logger.info("resuming run %s (method %s, day %s)",
                    manifest["run_id"], manifest["method"],
                    manifest["ensemble"]["date"])
        network_stage(run_dir, manifest, args)
        return

    run_dir, manifest = solve_and_export(args)
    if args.skip_network:
        logger.info("--skip-network set; stopping after export. Resume "
                    "later with: python run_vpp_network.py resume "
                    "--run-dir \"%s\"", run_dir)
        return
    network_stage(run_dir, manifest, args)


if __name__ == "__main__":
    try:
        main()
    except vreg.PipelineError as exc:
        logger.error("pipeline aborted: %s", exc)
        sys.exit(1)
