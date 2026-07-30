# FCAS Co-optimisation Under Dynamic Operating Envelopes

**The headline result** (VPP_EXTENSION.md §9): quantify how much aggregate
contingency-raise FCAS capacity the VPP can offer under a **static** vs a
**dynamic** feeder export envelope. Static limits are known to throttle FCAS
delivery; SAPN showed dynamic limits can more than double available export at
key intervals. This script reproduces that comparison on the project's own
ensemble and formulation.

## How it works

The centralised VPP QP (Method A) is extended with per-household enablement
variables `r_i >= 0` (kW of raise the household commits to deliver if called).
Everything stays a convex QP:

```
headroom          b_ik + r_ik <= B_MAX             can physically ramp up by r
energy adequacy   soc_ik >= tau * r_ik             can sustain the response for tau hours
                  (as rows: A_soc b_i + tau r_i <= soc_init)
DOE interaction   sum_i (pi_ik - r_ik) >= D_min_k  a delivered response must still
                                                   fit inside the export envelope
objective         sum h (net-b)^2 - sum_k p_k R_k,   R_k = sum_i r_ik
```

The DOE-interaction row is the mechanism of interest: it is *exactly* how a
tight export limit throttles FCAS enablement, and exactly what a midday-relaxed
dynamic envelope frees up. `P` gains zero diagonal entries for `r` (still
diagonal); the FCAS price enters through `q` only.

Raise-only is implemented deliberately — one service, one price, clean story
(VPP_EXTENSION.md §9's advice). Lower is symmetric (charge headroom + empty
capacity). Aggregation matters: a single 10 kWh battery is far below any NEM
threshold, so `R_k` is the VPP-level offer.

## Simplifications to keep in mind

- Flat enablement price (`--fcas-price`, $/kW per interval). Real FCAS prices
  are extremely spiky; medians-alongside-means reporting is future work.
- 30-minute intervals, not the NEM's 5-minute dispatch. `--tau` (default 5 min)
  only controls the energy-adequacy coupling.
- Enablement, not delivery: the battery reserves capability; actual activation
  energy is not simulated.

## Run

```bash
python vpp/fcas_cooptimisation/fcas_cooptimisation.py --save
python vpp/fcas_cooptimisation/fcas_cooptimisation.py \
    --compare static,dynamic_solar,tight_tou --fcas-price 0.2 --save
```

Outputs: per-envelope table (total enabled raise kWh, midday mean kW, revenue,
energy savings) and `figures/fcas_enablement.png` — aggregate `R_k` profiles
side by side with the envelope shapes. The static-vs-`dynamic_solar` midday
ratio is the number to quote against the SAPN finding.
