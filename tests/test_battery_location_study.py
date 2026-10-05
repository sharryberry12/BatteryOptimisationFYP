"""
Tests for studies/battery_location_study.py.

  * aggregate_injection sums the battery series over the loads actually
    mapped (replicated customers count once per load);
  * the step-by-step daily runner reproduces run_daily()'s monitor
    samples exactly and returns one loss value per interval;
  * an aggregate Generator at the 11 kV bus removes its injection from
    the transformer flow while leaving customer voltages essentially
    untouched -- the property the location study measures.

The last two solve the Elermore Vale circuit (a few seconds each).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conftest import COMMON_DIR, GLM_DIR, requires_glm_sources  # noqa: E402

bl = pytest.importorskip("studies.battery_location_study")

T = bl.T
FLAT_KW = 1.0
N_MONITORS = 10


def test_aggregate_injection_counts_each_mapped_load():
    profiles = {
        1: [{"battery": np.full(T, 2.0)}],
        2: [{"battery": np.full(T, -1.0)}],
        3: [],                                   # customer without this day
    }
    lc_map = {"l1": 1, "l2": 1, "l3": 2, "l4": 3}

    inject = bl.aggregate_injection(lc_map, profiles, day_idx=0)

    np.testing.assert_allclose(inject, np.full(T, 2.0 + 2.0 - 1.0))


def _flat_profiles(ids, kw):
    day = {"date": "2011-02-05", "load": np.full(T, kw), "pv": np.zeros(T),
           "battery": np.zeros(T), "grid": np.full(T, kw),
           "soc": np.zeros(T), "savings": 0.0}
    return {cid: [dict(day)] for cid in ids}


def _prepare_flat_day(ev):
    """Fresh load-only circuit, every load at FLAT_KW all day, monitors on."""
    ev.build_elermorevale(str(GLM_DIR), str(COMMON_DIR), skip_generators=True)
    names = ev.get_network_load_names()
    lc_map = ev.map_customers_to_network_loads([1], names)
    monitored = ev.select_monitored_loads(lc_map, n_monitors=N_MONITORS)
    ev.add_monitors(monitored)
    ev.attach_shapes(lc_map, _flat_profiles([1], FLAT_KW), 0, series="grid")
    return monitored


@requires_glm_sources
def test_stepwise_runner_matches_run_daily_and_records_losses(ev):
    monitored = _prepare_flat_day(ev)
    ev.run_daily()
    tx_ref, _ = ev.collect_tx_power()
    v_ref = ev.collect_voltages(monitored)

    monitored = _prepare_flat_day(ev)
    loss_kw, loss_kvar = bl.run_daily_stepwise(ev)
    tx_step, _ = ev.collect_tx_power()
    v_step = ev.collect_voltages(monitored)

    assert loss_kw.shape == (T,) and loss_kvar.shape == (T,)
    np.testing.assert_allclose(tx_step, tx_ref, rtol=1e-6)
    for name in monitored:
        np.testing.assert_allclose(v_step[name], v_ref[name], rtol=1e-6)
    # flat load all day -> flat losses, at a plausible loss fraction
    assert np.ptp(loss_kw) < 1e-3 * loss_kw.mean()
    assert 0.005 < loss_kw.mean() / tx_ref.mean() < 0.15


@requires_glm_sources
def test_aggregate_generator_offsets_transformer_flow_not_customer_voltages(ev):
    inject_kw = 400.0

    monitored = _prepare_flat_day(ev)
    loss0, _ = bl.run_daily_stepwise(ev)
    tx0, _ = ev.collect_tx_power()
    v0 = np.array(list(ev.collect_voltages(monitored).values()))

    monitored = _prepare_flat_day(ev)
    bl.add_aggregate_generator(ev, np.full(T, inject_kw))
    loss1, _ = bl.run_daily_stepwise(ev)
    tx1, _ = ev.collect_tx_power()
    v1 = np.array(list(ev.collect_voltages(monitored).values()))

    # the transformer sees the injection (minus a small loss change)
    np.testing.assert_allclose(tx0 - tx1, inject_kw, rtol=0.02)
    # the LV network is untouched: losses and customer voltages barely move
    assert abs(loss1.mean() - loss0.mean()) < 0.05 * loss0.mean()
    assert np.abs(v1 - v0).max() < 0.005
