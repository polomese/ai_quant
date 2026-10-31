"""As-of joins: the latest fundamentals record *known* at each rebalance date.

Never merge fundamentals on period. Every lookup is a backward ``join_asof`` on
``available_at``, grouped by ticker, so a row is visible at date t iff available_at <= t.
"""

from __future__ import annotations

import polars as pl


def latest_known_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """Prepare stream rows for the as-of lookup.

    1. Same (ticker, available_at): keep the newest period (then the last version).
    2. Cumulative-max rule: a row for an *older* period that arrives after a newer period
       was already known (e.g. a late amendment) must not replace the newer period.
       Amendments of the current latest period are kept and win by later available_at.
    """
    rows = rows.sort(["ticker", "available_at", "calendardate"])
    rows = rows.unique(subset=["ticker", "available_at"], keep="last", maintain_order=True)
    prev_max = pl.col("calendardate").cum_max().shift(1).over("ticker")
    rows = rows.filter(prev_max.is_null() | (pl.col("calendardate") >= prev_max))
    return rows


def asof_join(
    grid: pl.DataFrame,
    rows: pl.DataFrame,
    *,
    date_col: str = "date",
    max_staleness_days: int | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Left-join to ``grid`` (ticker, date) the latest row with ``available_at <= date``.

    ``rows`` must contain ticker, calendardate, available_at (+ value columns). Value
    columns are renamed with ``suffix``. Rows whose period ended more than
    ``max_staleness_days`` before the date are treated as missing.
    """
    known = latest_known_rows(rows).sort("available_at")
    value_cols = [c for c in known.columns if c not in ("ticker",)]
    known = known.rename({c: f"{c}{suffix}" for c in value_cols}) if suffix else known
    avail = f"available_at{suffix}"
    cd = f"calendardate{suffix}"
    out = grid.sort(date_col).join_asof(
        known,
        left_on=date_col,
        right_on=avail,
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    )
    # defensive invariant: nothing from the future
    bad = out.filter(pl.col(avail) > pl.col(date_col)).height
    if bad:
        raise AssertionError(f"as-of join produced {bad} rows with available_at > date")
    if max_staleness_days is not None:
        stale = (pl.col(date_col) - pl.col(cd)).dt.total_days() > max_staleness_days
        fields = [c for c in known.columns if c != "ticker"]
        out = out.with_columns(
            [pl.when(stale).then(None).otherwise(pl.col(c)).alias(c) for c in fields]
        )
    return out
