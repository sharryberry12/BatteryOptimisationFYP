# Elermore Vale Model Verification

How we establish that the GLM → OpenDSS translation in
[elermorevale_openDSS.py](elermorevale_openDSS.py) is correct, and what
"correct" can mean for it. The feeder model is translated **at runtime** from
the GridLAB-D sources ([Elermorevale/](Elermorevale/) +
[common/Line Configs.glm](common/Line%20Configs.glm)) — there are no static
`.dss` files to inspect, so the translation itself must be verified.

Correctness splits into two independent claims:

1. **Translation fidelity** — the OpenDSS circuit faithfully represents what
   the GLM files say (parser, unit conversions, references, topology).
2. **Physical plausibility** — the resulting power flow behaves like a real
   11 kV feeder.

The test suite currently proves (1) up to the documented approximations below.
(2) is partially covered by the existing snapshot checks and is future work
(Levels 3–4).

## Running the tests

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest                  # 51 tests, ~1.5 s
```

`pytest.ini` disables pytest's faulthandler: dss-python's FreePascal engine
raises a *handled* SEH exception (`0xe0465043`) while its DLL loads, which
faulthandler would otherwise print as a scary-but-benign
"Windows fatal exception". Exit code and results are unaffected.

## The verification pyramid

| Level | What | Status |
|-------|------|--------|
| 1 | Unit tests on the pure translation functions | ✅ [tests/test_glm_translation.py](tests/test_glm_translation.py) |
| 2 | Invariants: GLM source vs built DSS circuit | ✅ [tests/test_translation_invariants.py](tests/test_translation_invariants.py) |
| 3 | Physics sanity tests (known-answer power flows) | ⬜ future work (recipes below) |
| 4 | Cross-validation against GridLAB-D | ⬜ future work (recipes below) |

## Level 1 — unit tests (synthetic inputs, no repo data)

Each pure function is tested against hand-constructed inputs. The
highest-value assertions and the silent failure they guard against:

| Test | Guards against |
|------|----------------|
| `test_zmatrix_ohm_per_mile_to_km` | wrong Ω/mile → Ω/km conversion — a **61 % error in every impedance** that still produces plausible-looking voltages |
| `test_zmatrix_rating_tiers` | mis-binned heuristic ampacity ratings |
| `test_zmatrix_malformed_z11_degrades_to_zero` | pins the documented degrade-to-jumper behaviour for unparseable z-matrices |
| `test_conductor_reference_*` | LV conductor lookup, estimated reactances (0.25 OH / 0.08 UG Ω/km), 1-vs-3-phase detection |
| `test_gfloat_*` | unit-suffix stripping (`"11.59 m^2"`), fallback defaults |
| `test_phase_mapping*` | GLM `AN/BN/CN/ABCN` → DSS `.1/.2/.3` bus suffixes; delta/neutral markers |
| `test_safe_name*` | name sanitisation and idempotence |
| `test_parse_glm_*` | object/property extraction, comment stripping, dotted keys, empty objects |

## Level 2 — translation invariants (real GLM source + built circuit)

**Group 1 — source invariants** (parse-only):

| Test | Guards against |
|------|----------------|
| `test_flat_brace_assumption_holds` | `parse_glm`'s regex silently **drops objects with nested braces**; raw `object X {` count must equal parsed count in every file |
| `test_glm_object_census` | changed/corrupted GLM sources; counts pinned to measured ground truth (below) |
| `test_every_line_configuration_resolves` | lines silently falling back to guessed fallback impedance (currently **0** do) |
| `test_every_conductor_reference_resolves` | configs referencing unknown conductors (would become r=0 superconductors) |
| `test_every_transformer_configuration_resolves` | the builder's silent 300 kVA default when a transformer config is missing |
| `test_safe_name_injective_per_namespace` | two GLM names sanitising to the same DSS name → silently merged elements |
| `test_referenced_impedances_are_physical` | every linecode actually used has 0 < r < 10, 0 ≤ x < 1 Ω/km, positive rating, 1 or 3 phases |
| `test_all_loads_connect_to_the_network` | loads whose parent chain (load → meter → node) never reaches a branch endpoint |

**Group 2 — engine reconciliation** (builds the DSS circuit, ~0.5 s, no solve):

| Test | Guards against |
|------|----------------|
| `test_builder_stats_match_glm_census` | builder counters diverging from the GLM census |
| `test_engine_element_counts_match_source` | `New` commands rejected or collided inside OpenDSS (engine count ≠ builder count) |
| `test_engine_total_load_power` | dropped/merged loads: total connected load must be exactly 3 kW × 1,810 |
| `test_engine_no_isolated_elements` | islanded subtrees — OpenDSS's topology processor must report **0** isolated branches and loads |

### Measured ground truth (checked-in GLM sources)

| Quantity | Value |
|----------|-------|
| Total GLM objects | 7,958 |
| Loads | **1,810** (the module docstring originally said 1,785; corrected after measurement) |
| PV systems (`solar`) | 155 |
| Batteries | 40 |
| Transformers | 24 (23 distribution + 1 zone sub) |
| Overhead / underground lines | 1,743 / 422 |
| Switches / fuses | 246 / 40 |
| Line configurations / conductors | 3,834 / 395 (1,136 z-matrix) |
| Configs referenced by lines | 82 — all resolve, 0 fallbacks |
| DSS engine after build | 2,451 Lines, 1,810 Loads, 25 Transformers (+OLTC +zone sub), 155 Generators, 40 Storage |

If the GLM sources are ever regenerated, re-derive these numbers (parse with
`parse_all_glm` and count by type) instead of deleting the assertions.

## Known approximations (what "correct" does NOT include)

The translation is an approximation **by design**; the tests pin fidelity *to
this approximation*, not to a perfect electromagnetic model:

- **Balanced-line reduction**: `extract_impedances` keeps only `z11` from the
  GLM's 3×3 impedance matrix — off-diagonal mutual coupling and inter-phase
  asymmetry are discarded.
- **Estimated LV reactances**: x = 0.25 (overhead) / 0.08 (underground) Ω/km
  are engineering guesses, not data.
- **Heuristic ampacity ratings**: binned by resistance tier, not from
  conductor datasheets.
- **Default load power**: every load is created at 3 kW / 0.95 pf; real
  time-series enter via LoadShapes during profile-driven simulation.
- **OLTC simplification**: the GridLAB-D LDC lookup-table regulator is
  modelled as a unity-ratio autotransformer with a RegControl.

These bound the achievable agreement in Level 4: expect *close*, not exact.

## Level 3 — physics sanity tests (future work)

Known-answer snapshot solves, in rough order of value:

1. **Zero-load test**: set every load to 0 kW → all buses ≈ 1.0 pu, losses
   ≈ 0. The single best detector of impedance-unit bugs.
2. **Energy conservation**: substation input P = Σ load P + total losses.
3. **Scaling monotonicity**: loads × 1.1 → min voltage falls, losses rise
   ≈ quadratically (×1.21).
4. **One hand-calculated voltage drop** along a single radial path
   (zone sub → 11 kV section → distribution transformer → load), compared
   within a few percent.
5. **Golden-file regression**: freeze snapshot results (losses, min/mean/max
   pu voltage) once validated; assert against them on every change.

## Level 4 — cross-validation against GridLAB-D (future work)

Run the original `.glm` model in GridLAB-D on the same snapshot and compare
substation P/Q, feeder-head current, and voltages at matched nodes. Because
of the approximations above, report the deviation ("voltages agree within
X %") rather than expecting identity — *verified translation, measured
approximation* is the defensible claim for the thesis.
