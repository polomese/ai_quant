"""Hand-built PIT examples. If any of these fail, every backtest number is suspect."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from conftest import fund_row, make_fund

from pitlab.config import Arm
from pitlab.pit.asof import asof_join, latest_known_rows
from pitlab.pit.availability import ArmSpec, stream_rows

ARMS = {
    "A": Arm(label="A", version="MR", timing="period_end", lag_trading_days=0),
    "B": Arm(label="B", version="MR", timing="original_filing", lag_trading_days=1),
    "C": Arm(label="C", version="AR", timing="own_filing", lag_trading_days=1),
}


def lookup(fund, arm_name, when, cal, stream="T", extra_lag=0, staleness=None):
    arm = ArmSpec.from_config(arm_name, ARMS[arm_name], extra_lag)
    rows = stream_rows(fund, arm, stream, cal)
    grid = pl.DataFrame({"ticker": ["XYZ"], "date": [when]})
    return asof_join(grid, rows, max_staleness_days=staleness).row(0, named=True)


# --- the three required proofs ------------------------------------------------------------


def test_filing_after_rebalance_is_invisible_to_arm_c(tiny_fund, calendar_2019_2021):
    r = lookup(tiny_fund, "C", date(2020, 4, 30), calendar_2019_2021)
    assert r["netinc"] is None and r["available_at"] is None


def test_arm_a_sees_restated_value_at_period_end(tiny_fund, calendar_2019_2021):
    r = lookup(tiny_fund, "A", date(2020, 4, 30), calendar_2019_2021)
    assert r["netinc"] == 120.0  # restated value, before it was even first filed
    assert r["available_at"] == date(2020, 3, 31)


def test_arm_b_restated_but_only_after_original_filing(tiny_fund, calendar_2019_2021):
    cal = calendar_2019_2021
    assert lookup(tiny_fund, "B", date(2020, 4, 30), cal)["netinc"] is None
    assert lookup(tiny_fund, "B", date(2020, 5, 15), cal)["netinc"] is None  # filing day itself
    r = lookup(tiny_fund, "B", date(2020, 5, 18), cal)  # next trading day (Monday)
    assert r["netinc"] == 120.0
    assert r["available_at"] == date(2020, 5, 18)


def test_arm_c_as_reported_from_next_trading_day(tiny_fund, calendar_2019_2021):
    cal = calendar_2019_2021
    assert lookup(tiny_fund, "C", date(2020, 5, 15), cal)["netinc"] is None
    r = lookup(tiny_fund, "C", date(2020, 5, 18), cal)
    assert r["netinc"] == 100.0  # first-published value, never the restatement


def test_quarterly_stock_stream_follows_same_rules(tiny_fund, calendar_2019_2021):
    cal = calendar_2019_2021
    assert lookup(tiny_fund, "A", date(2020, 4, 30), cal, stream="Q")["equity"] == 1200.0
    assert lookup(tiny_fund, "B", date(2020, 4, 30), cal, stream="Q")["equity"] is None
    assert lookup(tiny_fund, "C", date(2020, 4, 30), cal, stream="Q")["equity"] is None
    assert lookup(tiny_fund, "B", date(2020, 5, 29), cal, stream="Q")["equity"] == 1200.0
    assert lookup(tiny_fund, "C", date(2020, 5, 29), cal, stream="Q")["equity"] == 1000.0


# --- multiple versions / ordering ---------------------------------------------------------


def test_latest_available_version_wins(calendar_2019_2021):
    q1 = date(2020, 3, 31)
    fund = make_fund(
        [
            fund_row("XYZ", "ART", q1, date(2020, 5, 15), netinc=100.0),
            fund_row("XYZ", "ART", q1, date(2020, 6, 10), netinc=105.0),  # amendment
        ]
    )
    cal = calendar_2019_2021
    assert lookup(fund, "C", date(2020, 5, 29), cal)["netinc"] == 100.0
    assert lookup(fund, "C", date(2020, 6, 30), cal)["netinc"] == 105.0


def test_late_amendment_of_older_period_does_not_replace_newer(calendar_2019_2021):
    fund = make_fund(
        [
            fund_row("XYZ", "ART", date(2019, 12, 31), date(2020, 2, 20), netinc=90.0),
            fund_row("XYZ", "ART", date(2020, 3, 31), date(2020, 5, 15), netinc=100.0),
            fund_row("XYZ", "ART", date(2019, 12, 31), date(2020, 6, 15), netinc=95.0),
        ]
    )
    r = lookup(fund, "C", date(2020, 6, 30), calendar_2019_2021)
    assert r["calendardate"] == date(2020, 3, 31) and r["netinc"] == 100.0


def test_arm_b_drops_mr_rows_without_matching_ar_row(calendar_2019_2021):
    q1 = date(2020, 3, 31)
    fund = make_fund([fund_row("XYZ", "MRT", q1, q1, netinc=120.0)])
    assert lookup(fund, "B", date(2020, 12, 31), calendar_2019_2021)["netinc"] is None
    assert lookup(fund, "A", date(2020, 12, 31), calendar_2019_2021)["netinc"] == 120.0


def test_arm_b_uses_earliest_original_filing(calendar_2019_2021):
    q1 = date(2020, 3, 31)
    fund = make_fund(
        [
            fund_row("XYZ", "ART", q1, date(2020, 5, 15), netinc=100.0),
            fund_row("XYZ", "ART", q1, date(2020, 7, 1), netinc=104.0),
            fund_row("XYZ", "MRT", q1, q1, netinc=120.0),
        ]
    )
    r = lookup(fund, "B", date(2020, 5, 29), calendar_2019_2021)
    assert r["netinc"] == 120.0 and r["available_at"] == date(2020, 5, 18)


def test_staleness_cutoff(tiny_fund, calendar_2019_2021):
    cal = calendar_2019_2021
    assert lookup(tiny_fund, "C", date(2021, 3, 31), cal, staleness=456)["netinc"] == 100.0
    assert lookup(tiny_fund, "C", date(2021, 9, 30), cal, staleness=456)["netinc"] is None


def test_lag_sweep_shifts_visibility(tiny_fund, calendar_2019_2021):
    cal = calendar_2019_2021
    # +20 extra trading days on top of C's base +1 -> available 2020-06-15
    assert lookup(tiny_fund, "C", date(2020, 6, 12), cal, extra_lag=20)["netinc"] is None
    assert lookup(tiny_fund, "C", date(2020, 6, 15), cal, extra_lag=20)["netinc"] == 100.0
    # arm A + 5 trading days from 2020-03-31 -> 2020-04-07
    assert lookup(tiny_fund, "A", date(2020, 4, 6), cal, extra_lag=5)["netinc"] is None
    assert lookup(tiny_fund, "A", date(2020, 4, 7), cal, extra_lag=5)["netinc"] == 120.0


def test_lag4_uses_year_ago_value_known_at_the_time(calendar_2019_2021):
    """Year-ago revenue must be the version available when the current row became available."""
    fund = make_fund(
        [
            fund_row("XYZ", "ART", date(2019, 3, 31), date(2019, 5, 10), revenue=1000.0),
            # the year-ago period gets amended AFTER the current period's filing
            fund_row("XYZ", "ART", date(2019, 3, 31), date(2020, 8, 1), revenue=900.0),
            fund_row("XYZ", "ART", date(2020, 3, 31), date(2020, 5, 15), revenue=1100.0),
        ]
    )
    r = lookup(fund, "C", date(2020, 6, 30), calendar_2019_2021, stream="T")
    assert r["revenue"] == 1100.0 and r["revenue_lag4"] == 1000.0


def test_asof_never_returns_future_rows():
    rows = pl.DataFrame(
        {
            "ticker": ["XYZ", "XYZ"],
            "calendardate": [date(2020, 3, 31), date(2020, 6, 30)],
            "available_at": [date(2020, 5, 18), date(2020, 8, 17)],
            "netinc": [1.0, 2.0],
        }
    )
    grid = pl.DataFrame(
        {"ticker": ["XYZ"] * 3, "date": [date(2020, 5, 1), date(2020, 7, 31), date(2020, 8, 31)]}
    )
    out = asof_join(grid, rows).sort("date")
    assert out["netinc"].to_list() == [None, 1.0, 2.0]


def test_latest_known_rows_dedupes_same_availability():
    rows = pl.DataFrame(
        {
            "ticker": ["XYZ", "XYZ"],
            "calendardate": [date(2019, 12, 31), date(2020, 3, 31)],
            "available_at": [date(2020, 5, 18), date(2020, 5, 18)],
            "netinc": [1.0, 2.0],
        }
    )
    out = latest_known_rows(rows)
    assert out.height == 1 and out["netinc"][0] == 2.0


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_other_tickers_do_not_leak(tiny_fund, calendar_2019_2021, arm):
    grid = pl.DataFrame({"ticker": ["OTHER"], "date": [date(2020, 12, 31)]})
    rows = stream_rows(tiny_fund, ArmSpec.from_config(arm, ARMS[arm]), "T", calendar_2019_2021)
    assert asof_join(grid, rows)["netinc"][0] is None


def test_impossible_filings_are_dropped():
    from datetime import date

    import polars as pl

    from pitlab.pipeline import drop_impossible_filings

    f = pl.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "dimension": ["ARQ", "ARQ", "MRQ"],
            "reportperiod": [date(2020, 3, 31)] * 3,
            "datekey": [date(2020, 5, 1), date(2020, 3, 1), date(2020, 3, 31)],
        }
    )
    out = drop_impossible_filings(f)
    assert out.height == 2 and date(2020, 3, 1) not in out["datekey"].to_list()
