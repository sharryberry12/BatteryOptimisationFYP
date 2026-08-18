"""
paths.py -- the single source of truth for where things live in this repo.

Every script and test derives its default input/output locations from here,
so nothing depends on the current working directory:

    ROOT/
      data/                 inputs (gitignored): data.csv, data_3_years.csv
      dispatch/             Part A -- QP battery scheduling (osqp_daily*.py)
      network/              Elermore Vale OpenDSS model, GLM sources, validation
      vpp/                  Part B -- multi-household coupling methods + pipeline
      studies/              peak-duty study, event replay, write-ups
      docs/                 cross-cutting docs (WALKTHROUGH.md)
      tests/                pytest suite
      outputs/              everything generated (gitignored):
        profiles/           long-format dispatch CSVs from dispatch/
        figures/            plots from every script (subfolders per run)
        runs/               vpp/run_vpp_network.py artefacts (manifests tracked)
        cache/              cleaned-day-array pickles (vpp_common, peak_duty)

Usage from a script anywhere in the tree (repo root must be on sys.path;
scripts add it from their own location):

    from paths import PROFILES, FIGURES, GLM_DIR, GLM_COMMON
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- inputs --------------------------------------------------------------
DATA_DIR = ROOT / "data"
DATA_CSV = DATA_DIR / "data.csv"                 # Ausgrid one-year window
DATA_3Y_CSV = DATA_DIR / "data_3_years.csv"      # optional 3-year file (peak-duty study)

# ---- code-side inputs (tracked) -----------------------------------------
NETWORK_DIR = ROOT / "network"
GLM_DIR = NETWORK_DIR / "glm" / "Elermorevale"   # GridLAB-D feeder sources
GLM_COMMON = NETWORK_DIR / "glm" / "common"      # shared GLM includes (Line Configs.glm ...)
VALIDATION_DIR = NETWORK_DIR / "validation"      # GridLAB-D cross-validation harness

# ---- outputs (generated, gitignored) ------------------------------------
OUTPUTS = ROOT / "outputs"
PROFILES = OUTPUTS / "profiles"
FIGURES = OUTPUTS / "figures"
RUNS = OUTPUTS / "runs"
CACHE = OUTPUTS / "cache"

DASHBOARD_HTML = OUTPUTS / "elermorevale_dashboard_v2.html"


def ensure_output_dirs():
    """Create the outputs/ tree (idempotent)."""
    for d in (OUTPUTS, PROFILES, FIGURES, RUNS, CACHE):
        d.mkdir(parents=True, exist_ok=True)


def rel(path):
    """Repo-relative string for logging (falls back to the absolute path)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
