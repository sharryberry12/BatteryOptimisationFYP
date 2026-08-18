"""
vpp_export.py
=============

Stage-3 artifact export for run_vpp_network.py (PIPELINE_DESIGN.md
Section 3.2).

Writes the three dispatch scenarios (no-battery / uncoupled / coupled)
in the exact long-format schema that
elermorevale_openDSS.load_profiles_from_csv() reads, so the network
stage consumes them unchanged, plus a manifest.json recording how the
run was produced.

Sign conventions (shared by both halves of the pipeline):
    battery_kw  b  : +ve discharge, -ve charge
    grid_kw     p  : load - pv - battery, +ve import, -ve export

Household names like "350#2" (replicated profiles) are not integers, so
each household gets a sequential pseudo customer id 1..N in the CSVs;
the pseudo-id -> household mapping is recorded in the manifest.

manifest.json is strict JSON: non-finite floats are encoded as the
strings "inf"/"-inf" (unbounded envelope entries) or null (NaN, e.g.
the Gini coefficient of the all-zero nobatt savings). Decode envelope
lists back to floats with envelope_array().
"""

import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = VPP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vpp import vpp_common as vc  # noqa: E402

logger = logging.getLogger("vpp.export")

SCENARIOS = ("nobatt", "uncoupled", "coupled")
MANIFEST_NAME = "manifest.json"
EXTRAS_NPZ_NAME = "extras.npz"


def scenario_csv_name(scenario):
    return f"dispatch_{scenario}.csv"


def _json_safe(obj):
    """
    Recursively convert numpy/Path values into strict-JSON-dumpable
    ones. Non-finite floats become "inf"/"-inf" strings (NaN -> null)
    so the manifest parses under jq/JSON.parse, not just Python.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return None
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, Path):
        return str(obj)
    return obj


def envelope_array(values):
    """Decode a manifest envelope list ("inf"/"-inf" strings allowed)."""
    return np.array([float(v) for v in values], dtype=float)


def _git_sha(repo_root):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


def profile_frame(households, B, date_iso, tariff, mode):
    """
    Long-format dataframe for one dispatch matrix B, column-compatible
    with osqp_daily.save_profiles() (the schema
    elermorevale_openDSS.load_profiles_from_csv() reads).
    """
    N, T = len(households), vc.T
    B = np.asarray(B, dtype=float)
    if B.shape != (N, T):
        raise ValueError(f"dispatch shape {B.shape} != ({N}, {T})")
    if not np.all(np.isfinite(B)):
        bad = sorted({households[i].name
                      for i in np.nonzero(~np.isfinite(B))[0]})
        raise ValueError(f"non-finite dispatch for households: {bad}")

    load = np.vstack([hh.load for hh in households])
    pv = np.vstack([hh.pv for hh in households])
    grid = load - pv - B                       # p = l - g - b
    soc = np.vstack([vc.SOC_INIT_FRAC * hh.e_max - np.cumsum(b) * vc.DT
                     for hh, b in zip(households, B)])
    savings = np.array([vc.day_savings(hh, b, tariff, mode)
                        for hh, b in zip(households, B)])

    return pd.DataFrame({
        "customer": np.repeat(np.arange(1, N + 1), T),
        "date": date_iso,
        "interval": np.tile(np.arange(1, T + 1), N),
        "hour": np.tile(np.arange(T) * vc.DT, N),
        "load_kw": load.ravel(),
        "pv_kw": pv.ravel(),
        "battery_kw": B.ravel(),
        "grid_kw": grid.ravel(),
        "soc_kwh": soc.ravel(),
        "daily_savings": np.repeat(savings, T),
    })


def _scenario_metrics(households, B, tariff, mode, d_min, d_max):
    """VPP-level (pre-power-flow) metrics for one dispatch."""
    agg_pi = vc.aggregate_pi(households, B)
    sav = vc.savings_vector(households, B, tariff, mode)
    return {
        "objective": vc.objective_surrogate(households, B),
        "agg_pi_kw": agg_pi,
        "envelope_violation": vc.envelope_violation(agg_pi, d_min, d_max),
        "savings_total": float(sav.sum()),
        "savings_mean": float(sav.mean()),
        "jain": vc.jain_index(sav),
        "gini": vc.gini(sav),
    }


def export_run(run_dir, households, tariff, mode, date_iso, d_min, d_max,
               dispatch, cli_args):
    """
    Write the three scenario CSVs, extras.npz and manifest.json into
    run_dir. Returns the manifest dict.

    dispatch : vpp_registry.VPPDispatch
    cli_args : dict of the fully-resolved CLI namespace (vars(args))
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    B_by_scenario = {
        "nobatt": np.zeros((len(households), vc.T)),
        "uncoupled": np.vstack([hh.b_uncoupled for hh in households]),
        "coupled": np.asarray(dispatch.B, dtype=float),
    }

    files, metrics = {}, {}
    for scen in SCENARIOS:
        df = profile_frame(households, B_by_scenario[scen],
                           date_iso, tariff, mode)
        path = run_dir / scenario_csv_name(scen)
        df.to_csv(path, index=False)
        files[scen] = path.name
        metrics[scen] = _scenario_metrics(
            households, B_by_scenario[scen], tariff, mode, d_min, d_max)
        logger.info("exported %s (%d rows)", path.name, len(df))

    extras_scalar, extras_arrays = {}, {}
    for key, val in dispatch.extras.items():
        if isinstance(val, np.ndarray):
            extras_arrays[key] = val
        else:
            extras_scalar[key] = _json_safe(val)
    if extras_arrays:
        np.savez_compressed(run_dir / EXTRAS_NPZ_NAME, **extras_arrays)

    manifest = {
        "run_id": run_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(vc.REPO_ROOT),
        "pipeline_version": 1,
        "method": dispatch.method,
        "cli_args": _json_safe(cli_args),
        "ensemble": {
            "n_households": len(households),
            "date": date_iso,
            "mode": mode,
            "e_max_kwh": float(households[0].e_max),
        },
        "envelope": {
            "d_min_kw": _json_safe(np.asarray(d_min, dtype=float)),
            "d_max_kw": _json_safe(np.asarray(d_max, dtype=float)),
        },
        "households": [
            {"pseudo_id": i + 1, "name": hh.name, "customer": hh.customer,
             "e_max_kwh": hh.e_max,
             "savings_uncoupled": hh.savings_uncoupled}
            for i, hh in enumerate(households)],
        "dispatch": {
            "converged": bool(dispatch.converged),
            "iterations": dispatch.iterations,
            "solve_time_s": float(dispatch.solve_time),
            "extras": extras_scalar,
            "extras_npz": sorted(extras_arrays) if extras_arrays else None,
        },
        "scenario_metrics": _json_safe(metrics),
        "files": files,
    }
    manifest = _json_safe(manifest)
    manifest_path = run_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("manifest written: %s", manifest_path)
    return manifest


def load_manifest(run_dir):
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {run_dir} -- is this a pipeline run "
            "directory produced by run_vpp_network.py?")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(run_dir, manifest):
    (Path(run_dir) / MANIFEST_NAME).write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False),
        encoding="utf-8")
