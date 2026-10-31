"""Arm availability rules: when does each fundamentals row become usable?

Every fundamentals row leaves this module with an explicit ``available_at`` (a trading
day). Rules (see docs/METHODOLOGY.md):

* ``period_end``      (arm A): MR row, ``available_at = shift(calendardate, lag)``
* ``original_filing`` (arm B): MR row, ``available_at = shift(original AR datekey, lag)``,
  where the original filing date is the *earliest* ``datekey`` of the matching ARQ/ART row
  for the same ticker + calendardate. MR rows without a matching AR row are dropped.
* ``own_filing``      (arm C): AR row, ``available_at = shift(own datekey, lag)``.

``shift(d, k)`` = the k-th trading day after the first trading day on/after ``d``; so a
filing on trading day T with k=1 is usable from T+1, and a weekend filing with k=1 is
usable from the second trading day after it (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from pitlab.config import Arm

STOCK_ITEMS = ["equity", "assets", "sharesbas"]
FLOW_ITEMS = ["netinc", "revenue", "gp", "ncfo"]
#: single-quarter flows taken from the Q stream, renamed so they don't clash with TTM values
QUARTER_FLOWS = {"netinc": "netinc_qtr"}
#: per stream, which items get a year-ago (lag-4-quarter) value
LAG_ITEMS = {"Q": ["equity", "assets", "sharesbas", "netinc_qtr"], "T": ["revenue"]}


@dataclass(frozen=True)
class ArmSpec:
    """A concrete arm: rule + total trading-day lag (base lag + sweep shift)."""

    name: str
    version: str  # "MR" | "AR"
    timing: str  # "period_end" | "original_filing" | "own_filing"
    lag: int

    @classmethod
    def from_config(cls, name: str, arm: Arm, extra_lag: int = 0) -> ArmSpec:
        return cls(
            name=name, version=arm.version, timing=arm.timing, lag=arm.lag_trading_days + extra_lag
        )


def trading_calendar(stocks: pl.DataFrame) -> np.ndarray:
    """Sorted unique trading days (datetime64[D]) observed in the price table."""
    return np.sort(stocks.get_column("date").unique().to_numpy().astype("datetime64[D]"))


def shift_trading_days(dates: pl.Series, calendar: np.ndarray, k: int) -> pl.Series:
    """Map each date to the k-th trading day after the first trading day >= date.

    Dates that land beyond the end of the calendar become null (not yet available).
    """
    d = dates.to_numpy().astype("datetime64[D]")
    nat = np.isnat(d)
    idx = np.searchsorted(calendar, np.where(nat, calendar[0], d), side="left") + k
    ok = (~nat) & (idx < len(calendar))
    out = np.where(ok, calendar[np.minimum(idx, len(calendar) - 1)], np.datetime64("NaT", "D"))
    return pl.Series(dates.name, out.astype("datetime64[D]"), dtype=pl.Date)


def _dims(version: str, stream: str) -> str:
    return f"{version}{stream}"  # e.g. ARQ, MRT


def original_filing_dates(fund: pl.DataFrame, stream: str) -> pl.DataFrame:
    """Earliest AR datekey per (ticker, calendardate) for a stream (Q or T)."""
    return (
        fund.filter(pl.col("dimension") == _dims("AR", stream))
        .group_by(["ticker", "calendardate"])
        .agg(pl.col("datekey").min().alias("orig_filing"))
    )


def stream_rows(
    fund: pl.DataFrame, arm: ArmSpec, stream: str, calendar: np.ndarray
) -> pl.DataFrame:
    """Rows of one stream (Q = stock items, T = flow items) with ``available_at`` and lag-4Q values.

    Output columns: ticker, calendardate, datekey, available_at, <items>, <item>_lag4.
    """
    items = STOCK_ITEMS + list(QUARTER_FLOWS) if stream == "Q" else FLOW_ITEMS
    items = [c for c in items if c in fund.columns]
    rows = fund.filter(pl.col("dimension") == _dims(arm.version, stream)).select(
        "ticker", "calendardate", "datekey", *items
    )
    if stream == "Q":
        rows = rows.rename({k: v for k, v in QUARTER_FLOWS.items() if k in rows.columns})
    if arm.timing == "period_end":
        base = rows.get_column("calendardate")
    elif arm.timing == "own_filing":
        base = rows.get_column("datekey")
    elif arm.timing == "original_filing":
        rows = rows.join(
            original_filing_dates(fund, stream), on=["ticker", "calendardate"], how="inner"
        )
        base = rows.get_column("orig_filing")
        rows = rows.drop("orig_filing")
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(arm.timing)
    rows = rows.with_columns(shift_trading_days(base, calendar, arm.lag).alias("available_at"))
    rows = rows.filter(pl.col("available_at").is_not_null())
    return add_lag4(rows, [c for c in LAG_ITEMS[stream] if c in rows.columns])


def add_lag4(rows: pl.DataFrame, items: list[str]) -> pl.DataFrame:
    """Attach the year-ago value of ``items`` *as it was known* when this row became available.

    As-of join keyed on (ticker, calendardate - 1y), backward on available_at: we take the
    latest version of the year-ago period whose available_at <= this row's available_at.
    """
    if not items:
        return rows
    left = rows.with_columns(pl.col("calendardate").dt.offset_by("-1y").alias("_prev_cd")).sort(
        "available_at"
    )
    right = rows.select(
        "ticker",
        pl.col("calendardate").alias("_prev_cd"),
        pl.col("available_at").alias("_prev_avail"),
        *[pl.col(c).alias(f"{c}_lag4") for c in items],
    ).sort("_prev_avail")
    out = left.join_asof(
        right,
        left_on="available_at",
        right_on="_prev_avail",
        by=["ticker", "_prev_cd"],
        strategy="backward",
        check_sortedness=False,
    )
    return out.drop("_prev_cd", "_prev_avail")


def arm_streams(fund: pl.DataFrame, arm: ArmSpec, calendar: np.ndarray) -> dict[str, pl.DataFrame]:
    return {s: stream_rows(fund, arm, s, calendar) for s in ("Q", "T")}
