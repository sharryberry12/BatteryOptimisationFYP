"""
Shared fixtures for the Elermore Vale translation test suite.

The suite verifies the GLM -> OpenDSS translation in elermorevale_openDSS.py
at two levels (see MODEL_VERIFICATION.md):

  Level 1 (test_glm_translation.py)        -- pure translation functions,
                                              synthetic inputs, no repo data.
  Level 2 (test_translation_invariants.py) -- invariants of the real GLM
                                              source vs the built DSS circuit.

Session-scoped fixtures parse the GLM tree and build the DSS circuit once;
the build takes well under a second so the whole suite stays fast.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GLM_DIR = REPO_ROOT / "Elermorevale"
COMMON_DIR = REPO_ROOT / "common"

requires_glm_sources = pytest.mark.skipif(
    not GLM_DIR.is_dir() or not (COMMON_DIR / "Line Configs.glm").is_file(),
    reason="Elermorevale/ or common/Line Configs.glm not present",
)


@pytest.fixture(scope="session")
def ev():
    """The module under test (import deferred so a missing dss-python
    dependency skips the suite instead of erroring at collection)."""
    return pytest.importorskip("elermorevale_openDSS")


@pytest.fixture(scope="session")
def glm_objects(ev):
    """All parsed (source_file, object_type, props) tuples from Elermorevale/."""
    return ev.parse_all_glm(str(GLM_DIR))


@pytest.fixture(scope="session")
def line_tables(ev):
    """(conductors, configs, linecodes) from common/Line Configs.glm."""
    conductors, configs = ev.parse_line_configs(str(COMMON_DIR))
    linecodes = ev.extract_impedances(conductors, configs)
    return conductors, configs, linecodes


@pytest.fixture(scope="session")
def built_circuit(ev):
    """Build the full OpenDSS circuit once (PV + batteries included) and
    return (stats_dict, dss_engine). No power-flow solve is performed --
    Level 2 only checks that the built circuit matches the GLM source."""
    stats = ev.build_elermorevale(
        str(GLM_DIR), str(COMMON_DIR), skip_generators=False)
    return stats, ev.dss
