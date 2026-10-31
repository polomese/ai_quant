from __future__ import annotations

from datetime import date

import polars as pl

from pitlab.config import load_settings
from pitlab.pit.availability import ArmSpec, original_filing_dates, shift_trading_days, stream_rows


def test_shift_trading_days_weekday_weekend_and_end(calendar_2019_2021):
    s = pl.Series(
        "d", [date(2020, 5, 15), date(2020, 5, 16), date(2021, 12, 31), None], dtype=pl.Date
    )
    out1 = shift_trading_days(s, calendar_2019_2021, 1).to_list()
    assert out1 == [date(2020, 5, 18), date(2020, 5, 19), None, None]
    out0 = shift_trading_days(s, calendar_2019_2021, 0).to_list()
    assert out0[:3] == [date(2020, 5, 15), date(2020, 5, 18), date(2021, 12, 31)]


def test_available_at_per_arm(tiny_fund, calendar_2019_2021):
    s = load_settings()
    got = {}
    for name, arm in s.arms.items():
        rows = stream_rows(tiny_fund, ArmSpec.from_config(name, arm), "Q", calendar_2019_2021)
        got[name] = (rows["available_at"][0], rows["equity"][0])
    assert got["A"][0] == date(2020, 3, 31)
    assert got["B"][0] == date(2020, 5, 18)
    assert got["C"][0] == date(2020, 5, 18)


def test_values_per_arm(tiny_fund, calendar_2019_2021):
    s = load_settings()
    vals = {
        n: stream_rows(tiny_fund, ArmSpec.from_config(n, a), "T", calendar_2019_2021)["netinc"][0]
        for n, a in s.arms.items()
    }
    assert vals == {"A": 120.0, "B": 120.0, "C": 100.0}


def test_original_filing_dates(tiny_fund):
    out = original_filing_dates(tiny_fund, "Q")
    assert out.row(0, named=True)["orig_filing"] == date(2020, 5, 15)


def test_config_arms_match_spec():
    s = load_settings()
    assert (s.arms["A"].version, s.arms["A"].timing, s.arms["A"].lag_trading_days) == (
        "MR",
        "period_end",
        0,
    )
    assert (s.arms["B"].version, s.arms["B"].timing) == ("MR", "original_filing")
    assert (s.arms["C"].version, s.arms["C"].timing, s.arms["C"].lag_trading_days) == (
        "AR",
        "own_filing",
        1,
    )
    assert s.lag_sweep.extra_lags == [0, 1, 5, 20, 60]


def test_clock2_stub_protocol():
    from pitlab.clock2 import DatedScorer, NotImplementedScorer, is_pit_usable

    s = NotImplementedScorer(knowledge_cutoff=date(2024, 6, 1))
    assert isinstance(s, DatedScorer)
    assert is_pit_usable(s, date(2024, 7, 1)) and not is_pit_usable(s, date(2024, 5, 1))
