"""
Level 1.5 — line_impedance vs GridLAB-D itself (frozen mini-model).

The reference values below are a frozen voltdump from GridLAB-D 5.3
solving a four-feeder mini-model (2026-08-24, Windows build, NR solver,
``nominal_frequency 50``): one 433 V SWING source feeding

  n_oh3 : 1000 ft overhead ABCN  (OH_BARE_SYSTEM_506, line_Spacing_OH)
  n_oh1 :  500 ft overhead AN    (same config)
  n_ug3 :  800 ft underground ABCN (UG_415V_Cable_185, line_Spacing_UG)
  n_ug1 :  300 ft underground BN   (same cable)

with constant-power loads. Conductor, cable and spacing values are copied
verbatim from common/Line Configs.glm (the UG cable as the harness strips
it: shield_* dropped, concentric-neutral kept). Reproducing those solved
voltages through this module's matrices pins every modelling choice at
once: the modified-Carson coefficients at 50 Hz / 100 Ohm-m, GMR-ft and
Ohm/mile units, the k=1 concentric-neutral equivalent, the cable-used-as-
neutral-conductor handling, and the Kron reduction over exactly the
declared phases. GridLAB-D matched to < 1e-6 pu when frozen.

Regenerate the reference (scratchpad mini.glm pattern) whenever the
module's physics change; do NOT tune the module to the test.
"""

import cmath
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import line_impedance as li  # noqa: E402

V_NOM = 250.0
_A = cmath.exp(-1j * 2 * cmath.pi / 3)
V3 = np.array([V_NOM, V_NOM * _A, V_NOM * _A.conjugate()])
FT_PER_MILE = 5280.0

OH_506 = {"geometric_mean_radius": "0.0736", "resistance": "0.499"}
UG_185 = {"outer_diameter": "0.665", "conductor_gmr": "0.02",
          "conductor_diameter": "0.604", "conductor_resistance": "0.274",
          "neutral_gmr": "0.02", "neutral_diameter": "0.604",
          "neutral_resistance": "0.274", "neutral_strands": "1"}
SP_OH = {"name": "line_Spacing_OH", "distance_AB": "2", "distance_BC": "1",
         "distance_AC": "3", "distance_AN": "1", "distance_BN": "3",
         "distance_CN": "4"}
SP_UG = {"name": "line_Spacing_UG", "distance_AB": "0.1", "distance_BC": "0.1",
         "distance_AC": "0.1", "distance_AN": "0.1", "distance_BN": "0.1",
         "distance_CN": "0.1"}

# GridLAB-D 5.3 voltdump (RECT), frozen 2026-08-24.
GLD = {
    "n_oh3": [247.990791 - 0.964199j, -124.017267 - 214.597731j,
              -122.463380 + 215.022997j],
    "n_oh1": [249.354415 - 0.308484j],
    "n_ug3": [248.369309 + 0.534987j, -124.210153 - 215.631590j,
              -124.643049 + 214.569067j],
    "n_ug1": [-124.798477 - 216.396131j],
}
LOADS = {
    "n_oh3": np.array([5000 + 1000j, 4000 + 800j, 6000 + 1200j]),
    "n_oh1": np.array([2000 + 400j]),
    "n_ug3": np.array([4000 + 900j, 3000 + 600j, 5000 + 1100j]),
    "n_ug1": np.array([1500 + 300j]),
}


def _solve(v_src, z_ohm, s_va, iters=60):
    """Fixed-point constant-power solve, the same phase-frame equations
    both engines reduce to for a single radial line."""
    v = v_src.copy()
    for _ in range(iters):
        v = v_src - z_ohm @ np.conj(s_va / v)
    return v


def _case(node):
    oh = {"A": OH_506, "B": OH_506, "C": OH_506, "N": OH_506}
    ug = {"A": UG_185, "B": UG_185, "C": UG_185, "N": UG_185}
    return {
        "n_oh3": ("ABCN", oh, SP_OH, li.overhead_z_matrix, 1000, V3),
        "n_oh1": ("AN", oh, SP_OH, li.overhead_z_matrix, 500, V3[:1]),
        "n_ug3": ("ABCN", ug, SP_UG, li.underground_z_matrix, 800, V3),
        "n_ug1": ("BN", ug, SP_UG, li.underground_z_matrix, 300,
                  np.array([V3[1]])),
    }[node]


@pytest.mark.parametrize("node", sorted(GLD))
def test_matches_gridlabd_voltdump(node):
    phases, conds, sp, fn, length_ft, v_src = _case(node)
    z_mile, order = fn(phases, conds, sp, freq=50.0)
    v = _solve(v_src, z_mile * (length_ft / FT_PER_MILE), LOADS[node])
    err_pu = np.abs(v - np.array(GLD[node])) / V_NOM
    assert err_pu.max() < 1e-5, (
        f"{node}: {err_pu.max():.2e} pu from the GridLAB-D reference")


def test_frequency_matters():
    """The pin is meaningful: at 60 Hz the same model misses GridLAB-D by
    >1e-3 pu, so the 50 Hz agreement is not coincidence."""
    phases, conds, sp, fn, length_ft, v_src = _case("n_oh3")
    z_mile, _ = fn(phases, conds, sp, freq=60.0)
    v = _solve(v_src, z_mile * (length_ft / FT_PER_MILE), LOADS["n_oh3"])
    err_pu = np.abs(v - np.array(GLD["n_oh3"])) / V_NOM
    assert err_pu.max() > 1e-3


def test_carson_coefficients_at_50hz():
    fcr, fci, fat = li.carson_coeffs(50.0, 100.0)
    assert fcr == pytest.approx(0.079418, abs=1e-6)
    assert fci == pytest.approx(0.1011185, abs=1e-6)
    assert fat == pytest.approx(7.6786 + 0.5 * np.log(2.0), abs=1e-6)


def test_config_matrix_km_units_and_order():
    cfg = {"name": "c", "conductor_A": "oh", "conductor_B": "oh",
           "conductor_C": "oh", "conductor_N": "oh", "spacing": "sp"}
    full = {"oh": dict(OH_506, __type="overhead_line_conductor")}
    z_km, order = li.config_matrix_km(cfg, full, {"sp": SP_OH}, "ABCN",
                                      is_underground=False, freq=50.0)
    assert order == "ABC" and z_km.shape == (3, 3)
    z_mile, _ = li.overhead_z_matrix("ABCN",
                                     {p: OH_506 for p in "ABCN"},
                                     SP_OH, 50.0)
    assert np.allclose(z_km, z_mile / li.MI_TO_KM)


def test_zmatrix_sequence_balanced_and_unbalanced():
    balanced = {"z11": "0.2+0.3j", "z12": "0.1+0.15j", "z13": "0.1+0.15j",
                "z21": "0.1+0.15j", "z22": "0.2+0.3j", "z23": "0.1+0.15j",
                "z31": "0.1+0.15j", "z32": "0.1+0.15j", "z33": "0.2+0.3j"}
    z1, z0 = li.zmatrix_sequence_km(balanced)
    assert z1 == pytest.approx((0.1 + 0.15j) / li.MI_TO_KM)
    assert z0 == pytest.approx((0.4 + 0.6j) / li.MI_TO_KM)
    with pytest.raises(ValueError):
        li.zmatrix_sequence_km(dict(balanced, z22="0.9+0.3j"))
