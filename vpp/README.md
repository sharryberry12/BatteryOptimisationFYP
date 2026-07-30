# VPP Extension — Coupling Method Implementations

Implementations of the multi-household VPP coupling methodologies designed in
[../VPP_EXTENSION.md](../VPP_EXTENSION.md), built on the Part A single-household
QP scheduler ([../osqp_daily.py](../osqp_daily.py)). Read
[../paper_context.md](../paper_context.md) for notation and the base formulation.

## Layout

| Folder | Method (VPP_EXTENSION.md §) | One-liner |
|---|---|---|
| [centralised_qp/](centralised_qp/) | A (§3) | One stacked OSQP problem — the exact ground truth every other method benchmarks against |
| [two_stage_doe_allocation/](two_stage_doe_allocation/) | B (§4) | DNSP splits the feeder envelope per household, households solve independently — deployed Australian practice |
| [dual_decomposition/](dual_decomposition/) | C (§5) | Shadow-price coordination via projected subgradient — slow but the prices *are* the product |
| [sharing_admm/](sharing_admm/) | D (§6) | Boyd sharing ADMM — the recommended decomposition; same solver, same sparsity, same warm start |
| [price_based_control/](price_based_control/) | E (§7) | One-shot broadcast price, selfish response — the counterexample showing why envelopes are needed |
| [fcas_cooptimisation/](fcas_cooptimisation/) | §9 | Contingency-raise FCAS co-optimisation; quantifies static vs dynamic envelope FCAS capacity |

Method F (receding-horizon MPC, §8) is deliberately **not** a folder here: it is a
wrapper around any of the above, not an alternative coupling method. Layer it on
once a coupling method is chosen.

`vpp_common.py` holds everything shared: ensemble assembly, feeder envelope
scenarios, the centralised benchmark solve, persistent per-household OSQP
workspaces, validation invariants (paper_context.md §10) and savings/fairness
metrics. It **imports** `osqp_daily.py` rather than copying it — the data
pipeline, tariff, billing and heuristic live in exactly one place.

## Key formulation fact

The existing code solves in reduced b-space (battery power only, positive on
discharge), with grid power `pi = net - b` substituted out. The feeder coupling

```
D_min <= sum_i pi_i <= D_max      becomes      agg_net - D_max <= sum_i b_i <= agg_net - D_min
```

so coupling rows are `[I I ... I]` and `P` stays **diagonal** in every method,
including the ADMM proximal shift. All methods share the identical strictly
convex objective (weights `h_i` frozen from the uncoupled per-household
heuristic), so optimality gaps between methods are well defined.

## Running

All scripts run from the **repo root** and share a common CLI
(`--n-households`, `--date`, `--scenario {none,static,tight_tou,dynamic_solar}`,
`--export-limit`, `--mode {fit,net}`, `--save`):

```bash
python vpp/centralised_qp/centralised_qp.py --n-households 20 --save
python vpp/two_stage_doe_allocation/two_stage_doe_allocation.py --save
python vpp/dual_decomposition/dual_decomposition.py --save
python vpp/sharing_admm/sharing_admm.py --save
python vpp/price_based_control/price_based_control.py --save
python vpp/fcas_cooptimisation/fcas_cooptimisation.py --save
```

The first run cleans the full Ausgrid CSV (~1 min) and caches the day arrays in
`vpp/cache/` (delete to force a rebuild). Subsequent runs start in seconds. The
default day is the highest-PV day covered by every clean customer — the regime
where export envelopes actually bind.

Recommended experiment order (VPP_EXTENSION.md §11): centralised first (ground
truth + scaling curve), two-stage second (the policy-relevant efficiency/fairness
comparison), ADMM third (only if centralised hits a wall), price-based as the
cautionary baseline, FCAS as the headline static-vs-dynamic result.

## Shared caveats

- **Feeder infeasibility is the default, not an error.** Tight envelopes make
  the coupled problem infeasible; `centralised_qp.py --soft` adds penalised
  slack whose values identify *who/when*. Iterative methods cannot converge on
  an infeasible envelope — check the centralised solve first.
- **Weights are frozen** from the uncoupled heuristic. Re-running the greedy
  heuristic inside a coupled loop would make the objective method-dependent and
  the gap numbers meaningless.
- Modelling gaps from paper_context.md §9 (no round-trip efficiency, perfect
  foresight, daily SOC neutrality) are inherited untouched — close them there
  before publishing numbers from here.
- Everything is single-day. Annual sweeps are a loop over `--date` away, but
  mind the runtime of the iterative methods.
