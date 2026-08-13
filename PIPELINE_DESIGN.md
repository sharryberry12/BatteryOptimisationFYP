# End-to-End VPP → OpenDSS Pipeline — Design

Design for a single-command pipeline that takes a VPP coupling method, solves the
coupled dispatch, injects it into an OpenDSS network simulation, and reports the
physical network impact. Design only — no implementation is prescribed beyond the
interfaces below.

**Status:** implemented — `run_vpp_network.py` (orchestrator),
`vpp/vpp_registry.py` and `vpp/vpp_export.py` (named with the `vpp_` prefix to
match `vpp_common.py`, instead of the `registry.py` / `export.py` names used
below), plus the two refactors inside `elermorevale_openDSS.py`. Stage 4/5
re-runs use the `resume` subcommand (`run_vpp_network.py resume --run-dir
runs/...`) rather than a `--skip-solve` flag; `--skip-network` stops after
stage 3. Implementation note: the zone-substation monitor sign in
`collect_tx_power()` was found inverted relative to its documented
"+import" convention and fixed, since the envelope overlay depends on it.
**Scope:** one new orchestrator script + two small `vpp/` modules + two minimal
refactors inside `elermorevale_openDSS.py`. Everything else stays as-is and every
existing script remains independently runnable.

---

## 1. Problem statement

Today the two halves of the project don't talk to each other:

- **VPP side** (`vpp/`): six coupling methods, each ultimately producing an
  `(N, T)` battery dispatch matrix `B` for one day over a `households` ensemble
  built by `vpp_common.setup_ensemble()`. But each method exposes `B` differently
  (`CentralisedResult.B`, `run_admm → (B, hist, t)`, `run_dual → {"B_last",
  "B_avg"}`, `run_rule → (B, curtail_kwh, n_failed)`, `solve_fcas → (B, R, dt,
  status)`, price-based `→ B`), each inside its own `main()` with method-specific
  CLI flags. Results live in memory and figures; nothing is exported for the
  network stage.

- **OpenDSS side** (`elermorevale_openDSS.py`): knows nothing about
  optimisation. It consumes a long-format profiles CSV
  (`customer, date, interval, load_kw, pv_kw, battery_kw, grid_kw, soc_kwh,
  daily_savings`), round-robins customers over the ~1,785 network loads, attaches
  `grid = load − pv − battery` as loadshapes, runs a daily power flow, and
  collects voltages / substation power / losses.

The natural seam already exists: **the long-format profiles CSV is the
contract**. The pipeline's job is (a) a uniform way to invoke any method, (b) an
exporter from `(households, B)` to the CSV schema the network script already
reads, and (c) an orchestrator that chains the stages and produces a three-way
comparison (no-battery / uncoupled QP / VPP-coupled).

---

## 2. Architecture overview

```
run_vpp_network.py  (orchestrator, repo root)
│
├─ Stage 1  Ensemble setup        vpp_common.setup_ensemble(args)
│           → households, date_iso, tariff, (d_min, d_max)
│
├─ Stage 2  Method dispatch       vpp/registry.py
│           registry["sharing_admm"].run(households, d_min, d_max, cfg)
│           → VPPDispatch(B, converged, iters, solve_time, extras)
│
├─ Stage 3  Artifact export       vpp/export.py
│           writes runs/<run_id>/dispatch_{nobatt,uncoupled,coupled}.csv
│           (exact schema load_profiles_from_csv() already reads)
│           + manifest.json (all args, date, method params, git SHA)
│
├─ Stage 4  Network simulation    thin adapter over elermorevale_openDSS
│           for each scenario: build → map → attach shapes → solve → collect
│
└─ Stage 5  Reporting
            3-way voltage envelope / heatmaps / substation P–Q,
            measured feeder-head power vs the DOE envelope,
            summary CSV per run
```

New code is only three pieces — a **registry**, an **exporter**, and the
**orchestrator**. The six method scripts and the OpenDSS scripts stay standalone.

---

## 3. Components

### 3.1 Stage 2 — method registry (`vpp/registry.py`)

The key abstraction: one entry per method, normalising the heterogeneous solve
functions behind a uniform interface.

```python
@dataclass
class VPPDispatch:
    B: np.ndarray          # (N, T) battery dispatch, +ve = discharge
    method: str
    converged: bool        # False when e.g. ADMM hits its iteration cap
    iterations: int | None
    solve_time: float
    extras: dict           # method-specific: residual history, curtail_kwh,
                           # FCAS reserve schedule R, duals, slack, ...

@dataclass
class MethodSpec:
    name: str
    add_args: Callable[[argparse.ArgumentParser], None]   # rho, iters, rule, tau, ...
    run: Callable[[households, d_min, d_max, args], VPPDispatch]

REGISTRY: dict[str, MethodSpec]   # centralised_qp, dual_decomposition,
                                  # sharing_admm, price_based_control,
                                  # two_stage_doe_allocation, fcas_cooptimisation
```

- The `run` adapters are thin wrappers around functions that **already exist**
  (`solve_centralised`, `run_admm`, `run_dual`, `run_rule`, `solve_fcas`, …).
  Zero changes to the method files.
- `VPPDispatch` carries convergence metadata so the manifest records *how* the
  dispatch was obtained, not just the numbers — this matters when ADMM exhausts
  iterations or the hard centralised QP is infeasible.
- For dual decomposition, the adapter must pick `B_last` vs `B_avg` — make it a
  method flag with `B_avg` as the default (the feasibility-averaged iterate).
- FCAS is a slight outlier: its reserve schedule `R` has no meaning in an
  energy-only power flow. The adapter injects only `B` and stashes `R` in
  `extras`.

**CLI shape:** argparse **subcommands**
(`run_vpp_network.py sharing_admm --rho 0.5 --n-households 145 …`). Each
method's flags stay namespaced via its `add_args`; shared flags (ensemble,
envelope scenario, network, output) are registered on the parent parser,
reusing the option set of `vpp_common.standard_argparser`.

### 3.2 Stage 3 — exporter (`vpp/export.py`)

The CSV file is deliberately the boundary rather than passing arrays in memory:

- `load_profiles_from_csv()` in the network script works **unchanged**.
- Every run leaves a durable artifact (`runs/<run_id>/`) — every thesis figure
  becomes reproducible from the manifest.
- The (slow) network stage can be re-run against an old dispatch without
  re-solving the VPP stage.

Per run, write **three CSVs** in the existing schema, differing only in
`battery_kw`:

| File | `battery_kw` | Meaning |
|---|---|---|
| `dispatch_nobatt.csv` | `0` | no-battery baseline |
| `dispatch_uncoupled.csv` | `hh.b_uncoupled` | today's selfish per-household QP (already stored on `HouseholdDay`) |
| `dispatch_coupled.csv` | `VPPDispatch.B[i]` | the chosen coupling method |

Details:

- **Pseudo-IDs**: household names like `"350#2"` (replicated profiles) aren't
  ints, but the CSV schema expects integer customer IDs. Assign sequential
  pseudo-IDs at export; record the pseudo-ID → true household mapping in the
  manifest.
- **Sign convention assertion**: both sides agree already (`b` positive on
  discharge; VPP `pi = net − b`, network `grid = load − pv − battery`), but the
  exporter should assert `grid_kw == load_kw − pv_kw − battery_kw` per row so a
  future convention change fails loudly.
- **Manifest** (`manifest.json`): full resolved args, method params, ensemble
  date, N, envelope scenario and `(d_min, d_max)`, convergence metadata,
  pseudo-ID map, timings, git SHA.

Run directory layout:

```
runs/<method>_<scenario>_<date>_<timestamp>/
├── manifest.json
├── dispatch_nobatt.csv
├── dispatch_uncoupled.csv
├── dispatch_coupled.csv
├── network_summary.csv        # stage 5
└── figures/
```

### 3.3 Stage 4 — network adapter

Minimal-touch: the orchestrator imports `elermorevale_openDSS` and calls its
existing functions (`build_elermorevale`, `get_network_load_names`,
`map_customers_to_network_loads`, `select_monitored_loads`,
`simulate_scenario`, collectors). Two small refactors inside the network script
are worth doing:

1. **Generalise loadshape attachment.** `attach_loadshapes` and
   `attach_baseline_shapes` are near-duplicates differing only in which series
   they inject. Replace with one function taking a scenario descriptor (a
   per-day `grid_kw` extractor). The three-scenario comparison then falls out
   naturally — and with three exported CSVs it may reduce to "attach the `grid`
   column", making the baseline variant redundant.
2. **Address days by date, not index.** The VPP stage optimises one specific ISO
   date; the network stage currently selects `day_idx`. Pass the date through
   end-to-end so there is no off-by-one between what was optimised and what was
   simulated.

**Network backend selection:** `--network elermorevale|ieee13|lv_feeder` as a
flag, but target **Elermore Vale first**. The three network scripts share a CLI
surface but not code; unifying them behind a real `NetworkModel` protocol
(`build / attach / run_day / collect`) is a separate refactor that should not be
coupled to this pipeline. The synthetic LV feeder serves as the fast smoke-test
backend while iterating.

### 3.4 Stage 5 — reporting

Extend the existing plot set from 2-way (baseline vs QP) to 3-way
(no-battery / uncoupled / coupled):

- voltage envelope (min/max p.u. vs hour, three curves)
- voltage heatmaps + delta heatmaps (uncoupled − coupled is the new one)
- substation transformer P/Q
- per-day summary table → `network_summary.csv`

The one genuinely new figure — the payoff of the whole pipeline —
**measured substation power from the OpenDSS monitor overlaid with the DOE
envelope** (scaled to feeder level per Section 4). It answers the physical
question the VPP stage only answers in kW-bookkeeping: does the coupled method
actually keep the real feeder head inside the envelope, does the uncoupled
dispatch actually violate it, and do voltage violations move accordingly.

**Consistency assertion:** the sum of injected loadshapes must equal
`replication_factor × agg_pi` from the VPP result within tolerance, so a mapping
bug cannot silently produce plausible-looking plots.

---

## 4. Key design decision: N households vs ~1,785 network loads

The VPP ensemble has `N` households; the Elermore Vale network has ~1,785 load
elements. Three reconciliation semantics:

| Option | Semantics | Cost / notes |
|---|---|---|
| **Replicate** (round-robin, like today) | Feeder = k copies of the VPP ensemble; envelope arithmetic stays honest because the envelope is `export_limit × N` and scales proportionally with replication | Cheap. **Recommended default.** |
| **Exact** (`N = n_loads`) | Every network load individually coupled — the "true" whole-feeder VPP | Centralised QP ≈ 86k variables (fine for sparse OSQP); iterative methods solve 1,785 sub-QPs per iteration — slow but tractable. Offer as a flag for a headline result. |
| **Subset** | N VPP households mapped to specific loads (e.g. one distribution transformer's customers); rest of feeder runs baseline profiles | Models partial VPP penetration — the most realistic scenario, but needs topology-aware mapping. **Future work.** |

Default **replicate**: consistent with the repo's existing practice
(`map_customers_to_network_loads` already round-robins), and the feeder-head
envelope overlay in Stage 5 is scaled by the same replication factor, keeping
the comparison self-consistent.

---

## 5. Failure modes and gotchas

- **Non-convergence is a result, not a crash.** ADMM/dual hitting the iteration
  cap, or a hard-infeasible envelope, should still export a dispatch with
  `converged=False` recorded in the manifest and a prominent warning. The
  centralised method's `--soft` penalty mode is the existing fallback for
  infeasible envelopes.
- **Runtime budget.** Each network scenario rebuilds the full GLM → DSS model.
  Three scenarios × one day is fine; date sweeps multiply fast. The orchestrator
  must support consuming previously exported artifacts
  (`--skip-solve --run-dir runs/...`) so Stage 4/5 can re-run alone.
- **Validation before injection.** Run `vpp_common.validate_ensemble` on `B`
  before export (Section 10 invariants: SOC neutrality, SOC bounds, rate limit).
- **NaN hygiene.** Recent history (pro-rata surplus NaN bug) says: exporter
  should reject NaN/inf in `B` outright.
- **Monitored subset.** Voltage stats come from ~100 monitored loads, not all
  1,785 — keep the monitor selection identical across the three scenarios (same
  `select_monitored_loads` call, computed once) so deltas are apples-to-apples.

---

## 6. Build order

Each step leaves the repo in a working state:

1. **`vpp/registry.py`** — `VPPDispatch`, `MethodSpec`, adapters for all six
   methods (pure wrapping, no method-file changes).
2. **`vpp/export.py`** — three-CSV export + `runs/` layout + manifest.
3. **Orchestrator Stages 1–3** — verify end-to-end by feeding the exported CSV
   to the *unmodified* `elermorevale_openDSS.py --profiles ...` by hand.
4. **Network-side refactors + Stages 4–5** — generic shape attachment,
   date-addressed day selection, three-way plots, envelope-vs-measured overlay,
   summary CSV.

Later extensions (explicitly out of scope for v1): multi-method comparison in
one run reusing the same ensemble (`--methods a,b,c`), date/N/scenario sweeps
with aggregated reporting, the subset-mapping penetration study, and a unified
`NetworkModel` protocol across the three network scripts.
