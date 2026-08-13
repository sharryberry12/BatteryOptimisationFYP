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
python -m pytest                  # 57 tests, all pass (~4 s)
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
| 3 | Physics sanity tests (known-answer power flows) | ✅ [tests/test_physics_sanity.py](tests/test_physics_sanity.py) |
| 4 | Cross-validation against GridLAB-D | ⬜ planned (see plan below) |

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
| DSS engine after build | 2,451 Lines, 1,810 Loads, 25 Transformers (+OLTC +zone sub), 195 Generators (155 PV + 40 batteries — see Known defects #1), 0 Storage |

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

## Level 3 — physics sanity tests (known-answer power flows)

These solve the circuit and assert physical identities. They run the
**load-only model at 1 kW/household** — the builder's 3 kW default is a
placeholder that overloads the LV network (mean voltage ~0.74 pu, 19 %
losses), pushing constant-P loads below their 0.85 pu model floor.

| Test | Physical identity checked |
|------|---------------------------|
| `test_zero_load_gives_flat_profile_and_no_losses` | no load → all buses 0.97–1.03 pu, losses ≈ 0. Best single detector of impedance-unit bugs |
| `test_energy_conservation` | source P = Σ actual load P + losses (0.1 %) |
| `test_transformer_voltage_drop_matches_hand_calc` | solved 11 kV bus dip matches dV ≈ P·R + Q·X from the GLM's zone-transformer impedance (measured agreement ~3×10⁻⁴ pu) |
| `test_losses_scale_superlinearly_with_load` | loads ×1.2 → min V falls, loss ratio in (1.2, 1.2³); measured 1.51 ≈ quadratic |
| `test_golden_snapshot_regression` | frozen reference solve: source 1,945 kW, losses 133.7 kW, V min/mean/max 0.790/0.906/1.000 pu (dss-python 0.15.7 / DSS C-API 0.14.5) |
| `test_full_model_snapshot_energises_network` | full model (PV + generator-modelled batteries) solves and energises all nodes — guards the Storage-defect workaround below |

## Known defects (found by this suite)

1. **[WORKED AROUND] Storage elements destabilise the snapshot solve**
   (dss-python 0.15.7 / DSS C-API 0.14.5). With 2+ active `Storage`
   elements on this network the solve either fails to converge or — worse —
   collapses in 2 iterations to a **dead circuit (all voltages 0, zero
   power) that still reports `Converged=True`**. Any *single* storage is
   fine. Ruled out during diagnosis: connectivity (all 40 battery buses are
   real network buses), idle draw (0 doesn't help), explicit `kva`,
   `dispmode=EXTERNAL`, `controlmode=static` with a high cap, and
   `algorithm=Newton` (partial energisation only). A minimal 3-bus circuit
   does **not** reproduce it, and no newer dss-python is available for this
   Python version, so the root cause inside the engine remains open.
   **Workaround (implemented)**: `build_elermorevale` models the 40 Redflow
   batteries as dispatchable `Generator` elements — `kw=-0.18` idle
   (parasitic draw), `kva=5` rating; dispatch via the Generators API
   (kw > 0 discharge, kw < 0 charge). The full model now solves; guarded by
   `test_full_model_snapshot_energises_network`.
   **Fidelity caveat**: generators carry no state of charge, capacity, or
   round-trip efficiency, and the engine does not clamp `|kw|` to the 5 kW
   rating — any future dispatch built on these elements must enforce
   `|kw| ≤ P_Max` and do SOC/efficiency bookkeeping externally (the
   profile-driven pipeline already does: batteries enter via net-load
   LoadShapes with `skip_generators=True`). If a future dss-python fixes
   Storage, revert by swapping the generator block back and re-adding the
   engine-count expectations for Storage in the Level 2 tests.

2. **`solve_snapshot` trusts `Converged` alone.** In the dead-circuit state
   above it logs "converged: True", losses 0.0, and skips the voltage
   summary (the >0.01 pu filter removes every bus), so the failure is
   silent. A robust check should also require a non-empty energised-bus
   set (see `live_voltages_pu` in the physics tests).

## Level 4 — cross-validation against GridLAB-D (plan)

Goal: quantify the translation's approximation error by solving the *same*
operating point in both engines. Because of the documented approximations
(z11-only lines, estimated LV reactances), expect *close*, not identical —
the deliverable is a measured deviation, e.g. "voltages agree within X %".

1. **Install GridLAB-D** (v4.x binary release) and check the original model
   still runs: `gridlabd Elermorevale/main.glm` from the repo root. Expect
   to stub out `climate`/`player`/`recorder` includes that reference
   missing input files.
2. **Freeze one comparable operating point.** In GridLAB-D, override every
   load to constant 1 kW at 0.95 pf (matching `REALISTIC_KW` in the Level 3
   tests); disable PV, batteries, and the OLTC lookup-table control so both
   engines solve the same passive network snapshot.
3. **Record from GridLAB-D**: zone-substation P/Q, and voltage magnitudes
   at matched buses — the 11 kV bus, each of the 23 distribution-transformer
   LV buses, and ~20 spot loads spread across feeders (add `recorder`
   objects; single timestep).
4. **Record from OpenDSS**: the same quantities via the Level 3 helpers
   (`node_mean_pu`, `source_pq_kw`) on `build_elermorevale(...,
   skip_generators=True)` + `BatchEdit Load..* kW=1.0`.
5. **Compare in a script** (`tests/` or a notebook): join on bus name
   (GridLAB-D names map to OpenDSS via `safe_name`), report per-bus ΔV (pu),
   substation ΔP/ΔQ (%), and plot ΔV vs electrical distance. Acceptance
   guide: ≤1 % ΔV at 11 kV buses (z-matrix diagonal is faithful), a few %
   at LV extremities (reactances there are estimates).
6. **Write the result into the thesis** as: translation verified (Levels
   1–3) + approximation measured (Level 4). If LV deviations exceed a few
   percent, revisit the estimated x values (0.25/0.08 Ω/km) first — they
   are the loosest part of the translation.

Prerequisite/caveat: GridLAB-D's solver (FBS/NR on the full 3×3 z-matrices)
uses the mutual-coupling terms the translation drops, so some deviation is
expected *by construction*; that measured gap **is** the result, not a bug.
