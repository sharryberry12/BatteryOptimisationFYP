# Elermore Vale Model Verification

How we establish that the GLM → OpenDSS translation in
[elermorevale_openDSS.py](elermorevale_openDSS.py) is correct, and what
"correct" can mean for it. The feeder model is translated **at runtime** from
the GridLAB-D sources ([glm/Elermorevale/](glm/Elermorevale/) +
[glm/common/Line Configs.glm](glm/common/Line%20Configs.glm)) — there are no static
`.dss` files to inspect, so the translation itself must be verified.

Correctness splits into two independent claims:

1. **Translation fidelity** — the OpenDSS circuit faithfully represents what
   the GLM files say (parser, unit conversions, references, topology).
2. **Physical plausibility** — the resulting power flow behaves like a real
   11 kV feeder.

The test suite proves (1) up to the documented approximations below (Levels
1–2), and (2) through known-answer power flows (Level 3) and a measured
cross-validation against GridLAB-D (Level 4). Two runtime guards back the
tests: `solve_snapshot` rejects a converged-but-dead (all 0 V) solution, and
every profile-driven day asserts all voltage monitors stayed energised
(`assert_monitors_energised` → `DeadMonitorError`), because the engine's
`Converged=True` has twice been shown to be necessary but not sufficient
(defects #1 and #6).

## Running the tests

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest                  # 101 tests, all pass (~7 s)
```

`pytest.ini` disables pytest's faulthandler: dss-python's FreePascal engine
raises a *handled* SEH exception (`0xe0465043`) while its DLL loads, which
faulthandler would otherwise print as a scary-but-benign
"Windows fatal exception". Exit code and results are unaffected.

## The verification pyramid

| Level | What | Status |
|-------|------|--------|
| 1 | Unit tests on the pure translation functions | ✅ [../tests/test_glm_translation.py](../tests/test_glm_translation.py) |
| 2 | Invariants: GLM source vs built DSS circuit | ✅ [../tests/test_translation_invariants.py](../tests/test_translation_invariants.py) |
| 3 | Physics sanity tests (known-answer power flows) | ✅ [../tests/test_physics_sanity.py](../tests/test_physics_sanity.py) |
| 4 | Cross-validation against GridLAB-D | ✅ [validation/](validation/) + [../tests/test_validation_harness.py](../tests/test_validation_harness.py) — measured agreement ~1.0 % mean at 11 kV, ~1.1 % at LV, 3.9 % max at one feeder tail (results below) |

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
| `test_residential_load_phases_are_subset_of_service_point` | a load declaring a phase its parent meter/node does not carry (GridLAB-D's child-phase rule) — it would float at 0 V; the 25 BlueGen CHP `load`s are the pinned exception (14 mismatches) and are why the builder skips them (defect #6) |

**Group 2 — engine reconciliation** (builds the DSS circuit, ~0.5 s, no solve):

| Test | Guards against |
|------|----------------|
| `test_builder_stats_match_glm_census` | builder counters diverging from the GLM census (built loads + skipped CHP units = 1,810) |
| `test_engine_element_counts_match_source` | `New` commands rejected or collided inside OpenDSS (engine count ≠ builder count) |
| `test_engine_total_load_power` | dropped/merged loads: total connected load must be exactly 3 kW × 1,785 |
| `test_zone_regcontrol_built_only_when_requested` / `test_zone_regcontrol_settings_target_one_pu` | the zone RegControl appearing in profile mode without `--oltc`, or regulating to the wrong target (pins vreg 120 / ptratio 52.92 / band 2.4 / ±16/10 taps at 1.25 %) |
| `test_engine_no_isolated_elements` | islanded subtrees — OpenDSS's topology processor must report **0** isolated branches and loads (bus-level; it cannot see a load floating on a missing phase — that is Level 3's `test_every_load_sits_on_an_energised_node_phase`) |

### Measured ground truth (checked-in GLM sources)

| Quantity | Value |
|----------|-------|
| Total GLM objects | 7,958 |
| GLM `load` objects | **1,810** = 1,785 residential loads (subs/*.glm) + 25 BlueGen CHP units (generators/Generators2.glm, excluded from the build — defect #6). The module docstring's original "~1,785" was right for households. |
| PV systems (`solar`) | 155 |
| Batteries | 40 |
| Transformers | 24 (23 distribution + 1 zone sub) |
| Overhead / underground lines | 1,743 / 422 |
| Switches / fuses | 246 / 40 |
| Line configurations / conductors | 3,834 / 395 (1,136 z-matrix) |
| Configs referenced by lines | 82 — all resolve, 0 fallbacks |
| DSS engine after build | 2,451 Lines, 1,785 Loads, 25 Transformers (+OLTC +zone sub), 195 Generators (155 PV + 40 batteries — see Known defects #1), 0 Storage, 1 RegControl (0 in profile mode unless `--oltc`); 4,597 network node-phases, all energised in the load-only model |

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
  modelled as a unity-ratio autotransformer with a RegControl (vreg 120 V
  on a 52.92 PT ratio = 1.0 pu at the 11 kV winding, ±1 % band, 16 raise /
  10 lower taps of 1.25 %). It is built in the full model and, with
  `--oltc`, in profile mode; every solve runs `controlmode=off` otherwise.
  Measured 2026-08-16: with controls active on the representative days the
  11 kV bus stays at 0.9955–1.0004 pu and the tap never leaves neutral, so
  OLTC-on results equal OLTC-off (see [../studies/NETWORK_AWARE_DISPATCH.md](../studies/NETWORK_AWARE_DISPATCH.md)).
- **BlueGen CHP units not modelled**: the 25 `object load`s in
  `generators/Generators2.glm` are fuel-cell CHP units driven by a schedule
  (`schedule_BlueGen0`) that no checked-in file defines; `main.glm` does not
  include that file at all (only the 40 Redflow batteries from it are
  translated). See defect #6.

These bound the achievable agreement in Level 4: expect *close*, not exact.

## Level 3 — physics sanity tests (known-answer power flows)

These solve the circuit and assert physical identities. They run the
**load-only model at 1 kW/household** — the builder's 3 kW default is a
placeholder that overloads the LV network (mean voltage 0.84 pu, 12.7 %
losses, measured post-fix 2026-08), pushing constant-P loads below their
0.85 pu model floor.

| Test | Physical identity checked |
|------|---------------------------|
| `test_zero_load_gives_flat_profile_and_no_losses` | no load → all buses 0.97–1.03 pu, losses ≈ 0. Best single detector of impedance-unit bugs |
| `test_energy_conservation` | source P = Σ actual load P + losses (0.1 %) |
| `test_transformer_voltage_drop_matches_hand_calc` | solved 11 kV bus dip matches dV ≈ P·R + Q·X from the GLM's zone-transformer impedance (measured agreement ~2×10⁻⁴ pu) |
| `test_losses_scale_superlinearly_with_load` | loads ×1.2 → min V falls, loss ratio in (1.2, 1.2³); measured 1.49 ≈ quadratic |
| `test_golden_snapshot_regression` | frozen reference solve: source 1,859.5 kW, losses 74.5 kW, V min/mean/max 0.830/0.949/1.005 pu (dss-python 0.15.7 / DSS C-API 0.14.5, post fixes #3–#6 below; was 1,871.1 kW / 75.1 kW with the 11 energised BlueGen loads) |
| `test_full_model_snapshot_energises_network` | full model (PV + generator-modelled batteries) solves and energises all 4,597 network node-phases — guards the Storage-defect workaround below |
| `test_every_load_sits_on_an_energised_node_phase` | (×3: profile mode, profile mode + OLTC, full model) no Load connects to a node-phase that solves to < 0.5 pu — the per-node-phase check that `Topology.NumIsolatedLoads` cannot do; guards defect #6 |

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

2. **[FIXED 2026-08-16] `Converged` was trusted alone.** In the dead-circuit
   state above `solve_snapshot` logged "converged: True", losses 0.0, and
   skipped the voltage summary (the >0.01 pu filter removed every bus), so
   the failure was silent; the daily path had no check at all, which is how
   defect #6 reached the sweep CSVs. Now `solve_snapshot` returns False on
   an empty energised-bus set, and `simulate_scenario` calls
   `assert_monitors_energised`, which raises `DeadMonitorError` if any
   monitor reads ≤ 0.01 pu at any interval (a floating load *or* a daily
   run that aborted part-way and left zero-padded buffers).
   `run_full_sweep` re-raises that error instead of logging-and-skipping the
   day, and reports any other skipped days explicitly.

3. **[FIXED] Bare GLM lengths misread as metres** — caught by the Level 4
   cross-validation. GridLAB-D's default length unit is FEET; all 62 of
   the 11 kV backbone lengths are bare numbers while every LV length is
   metre-suffixed. The translation treated bare lengths as metres,
   inflating every 11 kV section impedance by 3.28× — a systematic ~4–5 %
   voltage-drop overstatement across the whole feeder (and inflated
   losses) in ALL results generated before 2026-08-13. Fixed by
   `glm_length_m()` in `elermorevale_openDSS.py` (explicit suffix wins,
   bare = feet); guarded by `test_glm_length_m` and the re-pinned golden
   regression. **Network results produced before this fix should be
   regenerated.**

4. **[FIXED] 3-phase loads built with the line-to-neutral `kv`** — caught
   by review of the first Level 4 harness. OpenDSS reads Load `kv` as L-N
   for 1-phase loads but **L-L for 2/3-phase loads**; the builder passed
   the GLM `nominal_voltage` (240 V L-N) for all of them. The 7 `ABCN`
   loads saw a per-phase base of 138.6 V (~1.8 pu at a ~250 V bus), above
   `vmaxpu=1.15`, so the constant-P model silently clamped them to
   constant-Z overdraw (~2.4× the commanded kW). Fixed with
   `kv = nominal_voltage·√3` for multi-phase loads; guarded by
   `test_multiphase_load_kv_uses_line_to_line_base`.

5. **[FIXED] Phantom phases from linecode-driven Line phase counts** —
   caught by the DSS-coverage guard added to `compare_voltages.py`. The
   builder never set `phases=` on `New Line`, so OpenDSS took the phase
   count from the **linecode**; a 1-phase GLM line using a shared 3-phase
   configuration padded `bus.1` to `bus.1.2.3`, energising ~1,900 phantom
   node-phases (6,485 vs the true 4,610) that polluted every voltage
   statistic. Fixed by an explicit `phases={nph}` after `linecode=`
   (OpenDSS rebuilds Z for the declared count from the code's
   symmetrical components); guarded by the 100 % join-coverage bound and
   the re-pinned node census.

6. **[FIXED 2026-08-16] BlueGen CHP "loads" floating at 0 V, zeroing every
   sweep's V min** — surfaced when the OLTC investigation reported
   `V min = 0.0000`, then traced to the plain model. `generators/Generators2.glm`
   declares 25 `object load`s named `BLUEGEN_*` (fuel-cell CHP units driven
   by `schedule_BlueGen0`, which no checked-in file defines); 14 of them
   declare a phase (e.g. `CN`) that their parent service-point meter and
   feeding line do not carry (`BN`). `parse_all_glm` globs every `.glm`
   under `Elermorevale/` even though `main.glm` never includes this file,
   so the builder created them as residential loads on their declared
   phase — a node-phase with no conductor to it. Until fix #5, phantom
   phases energised them anyway; after it they solved to 0 V while
   `Converged=True` and `Topology.NumIsolatedLoads == 0` (bus-level) both
   passed. Two of the fourteen (`bluegen_12926098`, `bluegen_5335434`) fell in
   the every-18th monitored set, so **every post-2026-08-13 sweep row had
   `v_min = 0.0` and +96 fake under-voltage violations per day**; the
   representative-day summaries from 2026-08-13 17:22/17:39 (pre-#5) were
   unaffected. Fixed by excluding CHP `load`s from the residential build in
   every mode (`is_chp_load`, `stats["chp_units_skipped"] = 25`; the 40
   Redflow batteries in the same file are still built), which also removes
   11 spurious 1 kW loads from the Level 4 comparison (the harness only
   ever processed `subs/*.glm`). Guarded by
   `test_residential_load_phases_are_subset_of_service_point`,
   `test_every_load_sits_on_an_energised_node_phase`, the dead-monitor
   guard (#2) and the re-pinned goldens. **Network results generated
   between 2026-08-13 and 2026-08-16 should be regenerated** (done for
   `outputs/figures/` and `outputs/figures/net/`; the pre-fix files were kept as
   `summaries.stale-20260813.txt`).

## Level 4 — cross-validation against GridLAB-D (results)

The *same* frozen operating point (load-only network, 1 kW @ 0.95 pf per
household, OLTC at neutral, source at 1.0 pu) solved independently by
GridLAB-D 5.3.0 (full 3×3 z-matrices, NR) and by the OpenDSS translation.
Reproduce from the repo root:

```bash
python network/validation/gen_harness.py       # strip GLM sources -> validation/stripped/ + harness.glm
gridlabd network/validation/harness.glm        # -> validation/voltages_gld.csv (4,252 nodes)
python network/validation/compare_voltages.py  # -> validation/voltage_comparison.csv + report
```

The harness strips what the comparison must not contain: solar/inverter
blocks, the LDC runtime-class regulator control (taps pinned to neutral),
the temperature-dependent load transforms (loads pinned to constant_power
totalling 1 kW @ 0.95 pf **per load, split across declared phases** — the
same total as OpenDSS's `kW=1.0`), and the tape-shield parameters
GridLAB-D 5.x rejects on concentric-neutral cables. The 25 loads outside
`subs/` are zeroed on the OpenDSS side for parity.

The comparison **fails loudly** (non-zero exit) instead of printing a
counter when its own inputs are broken: join coverage of live OpenDSS
node-phases must be ≥99 % (measured: **100.00 %**), no node may be dead
in one engine but live in the other (catches the all-zero
Converged=True defect, Known defects #1/#2), and no GridLAB-D node may
be collapsed rather than absent. `tests/test_validation_harness.py`
unit-tests the harness text surgery and pins the raw-name join contract
and load-kv semantics the comparison depends on.

**Result (4,597 matched node-phases — 100 % of live OpenDSS nodes):**

| Level | n | mean \|ΔV\| | p95 | max |
|-------|---|---------|-----|-----|
| 132 kV | 3 | 0.03 % | 0.03 % | 0.03 % |
| 11 kV | 306 | 1.01 % | 1.33 % | 1.35 % |
| LV | 4,288 | 1.10 % | 1.82 % | 3.90 % |

Stated as measured: **~1.0 % mean at 11 kV (p95 1.33 %), ~1.1 % mean at
LV, 3.9 % max** at the tail of one long feeder (FDR_61210L). This is
consistent with the pre-registered expectation of roughly 1 % at 11 kV
and a few % at LV extremities; the 11 kV mean sits marginally above the
1.0 % figure, and the residual gap is the documented approximation by
construction: the translation drops the z-matrix mutual-coupling terms
and uses estimated LV reactances, and OpenDSS sits consistently slightly
lower — worst at feeder tails.
**The thesis claim this supports: translation verified (Levels 1–3),
approximation measured at ~1 % mean / ≤3.9 % max (Level 4).**

History: the first comparison run showed 4.8–5.3 % systematic deviation
and caught the bare-length=feet bug (Known defects #3). The first
*harness* revision then reported 1.02 %/1.09 % over only 4,366
node-phases — a join-key bug (safe_name applied to one side only)
silently excluded 77 of the 101 11 kV backbone buses, and 3-phase loads
sat at 3× the OpenDSS operating point; those figures are superseded by
the table above, computed over the complete join after Known defects
#4/#5 were fixed. The cross-validation caught every one of these.
