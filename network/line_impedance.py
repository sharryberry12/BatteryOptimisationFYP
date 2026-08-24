"""
GridLAB-D-faithful line impedance computation for the Elermore Vale
translation.

Mirrors powerflow/line.cpp, overhead_line.cpp and underground_line.cpp
(GridLAB-D 5.x) so the OpenDSS translation solves the same per-length
phase impedance matrices as the GridLAB-D reference model:

- modified Carson's equations at the model's nominal frequency (the GLM
  sets ``nominal_frequency 50`` in common/ModulePowerflow.glm) and the
  GridLAB-D default earth resistivity of 100 Ohm-m;
- overhead lines: self/mutual terms from conductor GMR (ft), resistance
  (Ohm/mile) and spacing distances (ft), with the neutral conductor
  Kron-reduced when the line's phases include N;
- underground lines: Kersting's concentric-neutral cable model
  (equivalent-neutral GMR/resistance from neutral_gmr, neutral_strands k
  and the radial distance (outer_diameter - neutral_diameter)/24 ft),
  phase conductors + concentric neutrals + the separate neutral cable
  all Kron-reduced to the phase frame. The shield_* parameters are
  ignored, matching the stripped Line Configs the GridLAB-D 5.x harness
  actually solves (validation/gen_harness.py drops them).
- z-matrix line configurations (the 11 kV backbone): every matrix in the
  source is balanced (equal diagonals, equal off-diagonals — asserted at
  extraction), so the exact sequence impedances are z1 = z11 - z12 and
  z0 = z11 + 2 z12. OpenDSS rebuilds the correct phase submatrix from
  these for any declared phase subset (1-ph: (z0+2z1)/3 = z11 exactly).

Empirically pinned: a four-feeder mini-model (OH/UG x 3-ph/1-ph) solved
by GridLAB-D 5.3 matches this module's matrices to <1e-6 pu at 50 Hz
(tests/test_line_impedance.py re-derives the frozen voltdump values).

Units in = GridLAB-D native (Ohm/mile, ft, inches); units out = Ohm/km.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

MI_TO_KM = 1.60934
FT_PER_MILE = 5280.0
EARTH_RESISTIVITY = 100.0   # Ohm-m, GridLAB-D powerflow default (not set in GLM)
DEFAULT_FREQ = 50.0         # common/ModulePowerflow.glm `nominal_frequency 50`

_PHASES = "ABC"


def carson_coeffs(freq=DEFAULT_FREQ, rho=EARTH_RESISTIVITY):
    """Modified-Carson coefficients exactly as powerflow/line.cpp:
    freq_coeff_real, freq_coeff_imag (Ohm/mile), freq_additive_term."""
    fcr = 0.00158836 * freq
    fci = 0.00202237 * freq
    fat = math.log(rho / freq) / 2.0 + 7.6786
    return fcr, fci, fat


def z_self(r_mile, gmr_ft, freq=DEFAULT_FREQ):
    """Self impedance of a conductor, Ohm/mile. GridLAB-D zeroes the term
    when GMR or resistance is not positive (line.cpp guard)."""
    if not (gmr_ft > 0.0 and r_mile > 0.0):
        return 0j
    fcr, fci, fat = carson_coeffs(freq)
    return complex(r_mile + fcr, fci * (math.log(1.0 / gmr_ft) + fat))


def z_mutual(d_ft, freq=DEFAULT_FREQ):
    """Mutual impedance between two conductors d_ft apart, Ohm/mile."""
    if not d_ft > 0.0:
        return 0j
    fcr, fci, fat = carson_coeffs(freq)
    return complex(fcr, fci * (math.log(1.0 / d_ft) + fat))


def _kron(z, n_keep):
    """Eliminate rows/cols >= n_keep (neutrals) from the primitive matrix."""
    if z.shape[0] == n_keep:
        return z
    zii = z[:n_keep, :n_keep]
    zin = z[:n_keep, n_keep:]
    znn = z[n_keep:, n_keep:]
    znj = z[n_keep:, :n_keep]
    return zii - zin @ np.linalg.inv(znn) @ znj


def _spacing_distance(spacing, a, b):
    """Distance (ft) between positions a and b from a line_spacing object."""
    for key in (f"distance_{a}{b}", f"distance_{b}{a}"):
        if key in spacing:
            try:
                return float(str(spacing[key]).split()[0])
            except ValueError:
                pass
    raise KeyError(
        f"line_spacing {spacing.get('name', '?')} has no distance for "
        f"{a}-{b}")


def _cable_params(props):
    """Concentric-neutral cable parameters from an underground_line_conductor
    (shield_* ignored — the GridLAB-D harness strips them)."""
    g = lambda k: float(str(props.get(k, "0")).split()[0])  # noqa: E731
    return {
        "cgmr": g("conductor_gmr"),
        "cr": g("conductor_resistance"),
        "od": g("outer_diameter"),
        "nd": g("neutral_diameter"),
        "ngmr": g("neutral_gmr"),
        "nr": g("neutral_resistance"),
        "k": int(g("neutral_strands") or 0),
    }


def overhead_z_matrix(phase_str, cond_props, spacing, freq=DEFAULT_FREQ):
    """Phase-frame impedance matrix (Ohm/mile) of an overhead line.

    phase_str  : the LINE's phases (e.g. 'ABCN', 'AN') — GridLAB-D builds
                 the matrix only over the phases the line declares.
    cond_props : {position: conductor props dict} for positions in 'ABCN'.
    spacing    : line_spacing props dict (distances in ft).
    Returns (matrix over the present ABC phases, phase order string).
    """
    ph = [c for c in phase_str if c in _PHASES]
    positions = list(ph)
    has_n = "N" in phase_str and "N" in cond_props
    if has_n:
        p = cond_props["N"]
        gmr = float(str(p.get("geometric_mean_radius", "0")).split()[0])
        r = float(str(p.get("resistance", "0")).split()[0])
        if gmr > 0.0 and r > 0.0:
            positions.append("N")
    n = len(positions)
    z = np.zeros((n, n), dtype=complex)
    for i, a in enumerate(positions):
        pa = cond_props[a]
        gmr_a = float(str(pa.get("geometric_mean_radius", "0")).split()[0])
        r_a = float(str(pa.get("resistance", "0")).split()[0])
        z[i, i] = z_self(r_a, gmr_a, freq)
        for j in range(i + 1, n):
            zm = z_mutual(_spacing_distance(spacing, a, positions[j]), freq)
            z[i, j] = z[j, i] = zm
    return _kron(z, len(ph)), "".join(ph)


def underground_z_matrix(phase_str, cond_props, spacing, freq=DEFAULT_FREQ):
    """Phase-frame impedance matrix (Ohm/mile) of an underground line with
    concentric-neutral cables (Kersting; underground_line.cpp).

    Row order in the primitive matrix: phase conductors, their concentric
    neutrals, then the separate neutral cable (its phase conductor only —
    GridLAB-D warns 'phase N conductor should just be a normal conductor'
    and uses it as one). Everything after the phase rows is Kron-reduced.
    """
    ph = [c for c in phase_str if c in _PHASES]
    cables = {p: _cable_params(cond_props[p]) for p in ph if p in cond_props}

    rows = []                      # (label, self_z, phase_position)
    for p in ph:
        c = cables[p]
        rows.append((p, z_self(c["cr"], c["cgmr"], freq), p))
    cn_rad = {}
    for p in ph:
        c = cables[p]
        rad = (c["od"] - c["nd"]) / 24.0          # inches -> ft radius
        if c["ngmr"] > 0.0 and c["k"] > 0 and rad > 0.0:
            gmr_cn = (c["ngmr"] * c["k"] * rad ** (c["k"] - 1)) ** (1.0 / c["k"])
            r_cn = c["nr"] / c["k"]
            cn_rad[p] = rad
            rows.append(("cn" + p, z_self(r_cn, gmr_cn, freq), p))
    if "N" in phase_str and "N" in cond_props:
        cn = _cable_params(cond_props["N"])
        if cn["cgmr"] > 0.0 and cn["cr"] > 0.0:
            rows.append(("N", z_self(cn["cr"], cn["cgmr"], freq), "N"))

    n = len(rows)
    z = np.zeros((n, n), dtype=complex)
    for i, (la, za, pa) in enumerate(rows):
        z[i, i] = za
        for j in range(i + 1, n):
            lb, _, pb = rows[j]
            if pa == pb:                       # phase and its own CN
                d = cn_rad[pa]
            else:
                d = _spacing_distance(spacing, pa, pb)
            zm = z_mutual(d, freq)
            z[i, j] = z[j, i] = zm
    return _kron(z, len(ph)), "".join(ph)


def config_matrix_km(config, conductors_full, spacings, phase_str,
                     is_underground, freq=DEFAULT_FREQ):
    """Ohm/km phase matrix for one (line_configuration, phase set).

    conductors_full : {name: props} for every *_line_conductor object.
    spacings        : {name: props} for every line_spacing object.
    Raises KeyError/ValueError when the config lacks the data — callers
    decide the fallback; the Level-2 invariants assert no used config does.
    """
    cond_props = {}
    for pos in "ABCN":
        ref = config.get(f"conductor_{pos}")
        if ref:
            if ref not in conductors_full:
                raise KeyError(f"conductor {ref} not defined")
            cond_props[pos] = conductors_full[ref]
    for p in phase_str:
        if p in _PHASES and p not in cond_props:
            raise ValueError(
                f"config {config.get('name', '?')} has no conductor for "
                f"declared phase {p}")
    sp_ref = config.get("spacing", "")
    if sp_ref not in spacings:
        raise KeyError(f"line_spacing {sp_ref!r} not defined")
    spacing = spacings[sp_ref]

    fn = underground_z_matrix if is_underground else overhead_z_matrix
    z_mile, order = fn(phase_str, cond_props, spacing, freq)
    return z_mile / MI_TO_KM, order


def zmatrix_sequence_km(props):
    """(z1, z0) in Ohm/km from a z-matrix line_configuration.

    The source matrices are balanced (asserted): z1 = z11 - z12,
    z0 = z11 + 2 z12 — exact, including OpenDSS's reconstruction of any
    phase subset. Malformed entries degrade to 0 (legacy behaviour).
    """
    def c(key):
        try:
            return complex(str(props.get(key, "0")).replace(" ", ""))
        except ValueError:
            return 0j
    z11 = c("z11")
    z12 = c("z12")
    for k in ("z22", "z33"):
        if abs(c(k) - z11) > 1e-9 * max(1.0, abs(z11)):
            raise ValueError(
                f"z-matrix config {props.get('name', '?')} is not balanced "
                f"({k} != z11); the sequence reduction does not apply")
    for k in ("z13", "z21", "z23", "z31", "z32"):
        if abs(c(k) - z12) > 1e-9 * max(1.0, abs(z12)):
            raise ValueError(
                f"z-matrix config {props.get('name', '?')} is not balanced "
                f"({k} != z12); the sequence reduction does not apply")
    z1 = (z11 - z12) / MI_TO_KM
    z0 = (z11 + 2.0 * z12) / MI_TO_KM
    return z1, z0
