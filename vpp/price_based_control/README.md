# Method E — Market-Mediated / Indirect Price Control

**The cautionary baseline.** No envelope allocation, no iteration: a price
signal is broadcast once and every household responds selfishly ("transactive
energy"). The point of implementing it is to show *why explicit envelopes are
needed* (VPP_EXTENSION.md §7): nothing guarantees the aggregate respects the
feeder limits, and identical batteries responding to an identical signal
**herd** — creating a new aggregate peak at the price trough, the
well-documented demand-response failure mode.

## How it works

The quadratic `h.(net-b)^2` term is deliberately **kept** and the price `mu`
enters as a linear term on grid power (`q`-shift, still a QP). This follows
option (a) from VPP_EXTENSION.md §7: under this project's tariff set (flat
$0.40/kWh export exceeding every import rate) a purely linear dollar objective
degenerates into bang-bang arbitrage — the documented reason Ratnam et al. use
the quadratic surrogate at all. Do not "simplify" this to an LP.

Three signals:

| Signal | What it is | What it demonstrates |
|---|---|---|
| `none` | zero price | the uncoupled baseline |
| `tou` | `gamma * TOU tariff` | a naive retail-shaped signal; raise `--gamma` to watch herding grow |
| `shadow` | `-y` from the centralised solve's coupling duals | the *right* congestion prices — a one-shot broadcast approximately reproduces the centralised optimum because household objectives are strictly convex |

The `shadow` row is the interesting contrast: perfect prices work (this is just
dual decomposition at its fixed point), but a DNSP can only compute them by
already solving the centralised problem — which is the argument for explicit
envelopes or iterative coordination in practice.

## Run

```bash
python vpp/price_based_control/price_based_control.py --save
python vpp/price_based_control/price_based_control.py --gamma 200 --save   # stronger herding
```

Outputs: per-signal table (objective, gap %, envelope violation kW/kWh, peak
export) and `figures/price_response.png` — aggregate profiles against the
(unenforced) envelope.

## Assessment (from VPP_EXTENSION.md §7)

| | |
|---|---|
| Optimality | No constraint guarantee — violation is measured, not prevented |
| Risk | Synchronisation/herding creates new peaks at the price trough |
| Role here | Comparison baseline / counterexample, not a primary method |
