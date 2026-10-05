"""
Tests for studies/doe_day_sweep.py (no data files needed).

  * jesmond_days returns only fully populated days, in kW, keyed by ISO
    date, with the same quarter-hour averaging as load_jesmond_day;
  * select_dates handles ranges, lists and unknown dates;
  * profiles_from_dispatch builds exactly the dict the network stage
    would get from a CSV round trip through vpp_export.profile_frame and
    elermorevale_openDSS.load_profiles_from_csv.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

vc = pytest.importorskip("vpp.vpp_common")
base = pytest.importorskip("dispatch.osqp_daily")
sw = pytest.importorskip("studies.doe_day_sweep")
sr = pytest.importorskip("studies.static_vs_doe_replay")
from test_static_vs_doe_replay import _write_jesmond_csv  # noqa: E402
from test_vpp_methods import synthetic_household  # noqa: E402

T = vc.T


def test_jesmond_days_keeps_only_fully_populated_days(tmp_path):
    csv = tmp_path / "jesmond.csv"
    full = [0.01 * (i + 1) for i in range(96)]
    partial = full[:-1] + [np.nan]
    _write_jesmond_csv(csv, {"04FEB2011": None, "05FEB2011": full,
                             "06FEB2011": partial, "07FEB2011": full})

    days = sw.jesmond_days(csv)

    assert sorted(days) == ["2011-02-05", "2011-02-07"]
    np.testing.assert_allclose(days["2011-02-05"],
                               sr.load_jesmond_day(csv, "2011-02-05"))


def test_select_dates_range_list_and_unknown():
    avail = {"2011-02-01": 0, "2011-02-02": 0, "2011-02-03": 0}
    assert sw.select_dates(avail, None) == sorted(avail)
    assert sw.select_dates(avail, "2011-02-02:2011-02-03") == \
        ["2011-02-02", "2011-02-03"]
    assert sw.select_dates(avail, "2011-02-03,2011-02-01") == \
        ["2011-02-01", "2011-02-03"]
    with pytest.raises(ValueError, match="not in the substation file"):
        sw.select_dates(avail, "2011-02-09")


def test_profiles_from_dispatch_matches_csv_round_trip(tmp_path):
    ev = pytest.importorskip("network.elermorevale_openDSS")
    vexport = pytest.importorskip("vpp.vpp_export")
    tariff = base.build_tariff()
    households = [synthetic_household(i, tariff) for i in range(3)]
    B = np.vstack([hh.b_uncoupled for hh in households])

    direct = sw.profiles_from_dispatch(households, B, "2011-02-05")
    path = tmp_path / "dispatch.csv"
    vexport.profile_frame(households, B, "2011-02-05", tariff, "fit") \
        .to_csv(path, index=False)
    via_csv = ev.load_profiles_from_csv(str(path))

    assert sorted(direct) == sorted(via_csv) == [1, 2, 3]
    for cid in direct:
        assert len(direct[cid]) == len(via_csv[cid]) == 1
        for key in ("load", "pv", "battery", "grid", "soc"):
            np.testing.assert_allclose(direct[cid][0][key],
                                       via_csv[cid][0][key], atol=1e-9)
        assert str(via_csv[cid][0]["date"])[:10] == direct[cid][0]["date"]
