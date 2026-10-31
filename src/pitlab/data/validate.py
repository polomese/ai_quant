"""Data validation. Errors fail loudly (non-zero exit); warnings are reported."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl
from loguru import logger

from pitlab.config import Settings
from pitlab.data.ingest import load_table, utc_stamp

PRIMARY_KEYS = {
    "fundamentals": ["ticker", "dimension", "datekey", "reportperiod"],
    "stocks": ["ticker", "date"],
    "daily": ["ticker", "date"],
    "tickers": ["table", "permaticker", "ticker"],
    "sp500": ["date", "action", "ticker"],
    "events": ["date", "ticker"],
}
REQUIRED_DATES = {
    "fundamentals": ["calendardate", "datekey"],
    "stocks": ["date"],
    "daily": ["date"],
}


#: max share of AR rows dated before their period end that is tolerated (and dropped)
IMPOSSIBLE_FILING_TOLERANCE = 1e-4


class ValidationError(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "error" | "warning" | "info"
    detail: str


def _tables(settings: Settings) -> dict[str, pl.DataFrame]:
    out = {}
    for name in ["fundamentals", "stocks", "daily", "tickers", "sp500", "actions", "events"]:
        with contextlib.suppress(FileNotFoundError):
            out[name] = load_table(settings, name)
    return out


def run_checks(tables: dict[str, pl.DataFrame]) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    for t in ("fundamentals", "stocks", "tickers"):
        if t not in tables:
            checks.append(Check(f"{t}: present", False, "error", "table missing"))
    for t, pk in PRIMARY_KEYS.items():
        if t in tables and all(c in tables[t].columns for c in pk):
            dup = tables[t].height - tables[t].select(pk).unique().height
            checks.append(
                Check(f"{t}: unique primary key {pk}", dup == 0, "error", f"{dup} duplicate keys")
            )
    for t, cols in REQUIRED_DATES.items():
        if t in tables:
            for c in cols:
                if c not in tables[t].columns:
                    checks.append(
                        Check(
                            f"{t}.{c}: column exists",
                            False,
                            "error",
                            "missing - check config/columns.yaml",
                        )
                    )
                    continue
                n = tables[t][c].null_count()
                checks.append(
                    Check(
                        f"{t}.{c}: dates parse (no nulls)", n == 0, "error", f"{n} unparseable/null"
                    )
                )
    stats: dict[str, Any] = {"row_counts": {k: v.height for k, v in tables.items()}}
    if "fundamentals" in tables:
        f = tables["fundamentals"]
        ar = f.filter(pl.col("dimension").str.starts_with("AR"))
        if "reportperiod" in f.columns:
            bad = ar.filter(pl.col("datekey") < pl.col("reportperiod")).height
            # a handful of vendor errors (20 of 1.56M AR rows, mostly 1996-98, on 2026-10-28)
            # are tolerated because the pipeline drops them (pipeline.drop_impossible_filings);
            # a non-trivial share would mean a mapping / parsing problem -> error
            share = bad / max(ar.height, 1)
            checks.append(
                Check(
                    "fundamentals AR*: datekey >= reportperiod",
                    bad == 0,
                    "error" if share > IMPOSSIBLE_FILING_TOLERANCE else "warning",
                    f"{bad} rows ({share:.4%}) filed before their fiscal period ended"
                    + (" - dropped by the pipeline" if bad else ""),
                )
            )
        # calendardate is a *normalized* period: Sharadar rounds the fiscal period end to a
        # calendar quarter-end (forward for off-calendar fiscal quarters, e.g. NKE Q ending
        # Nov 30 -> calendardate Dec 31, filed Dec 30; annual rows -> Dec 31). Filing before
        # calendardate is therefore legitimate; the real invariant is datekey >= reportperiod.
        for dims in (["ARQ", "ART"], ["ARY"]):
            early = ar.filter(
                pl.col("dimension").is_in(dims) & (pl.col("datekey") < pl.col("calendardate"))
            )
            checks.append(
                Check(
                    f"fundamentals {'/'.join(dims)}: datekey >= calendardate "
                    "(info; calendardate is normalized)",
                    True,
                    "info",
                    f"{early.height} rows filed before normalized calendardate"
                    + (
                        f" ({', '.join(sorted(early['ticker'].unique().to_list())[:10])})"
                        if early.height
                        else ""
                    ),
                )
            )
        mr = f.filter(pl.col("dimension").str.starts_with("MR"))
        if "reportperiod" in f.columns and mr.height:
            n_eq = mr.filter(pl.col("datekey") == pl.col("reportperiod")).height
            checks.append(
                Check(
                    "fundamentals MR*: datekey == reportperiod (Sharadar convention)",
                    n_eq == mr.height,
                    "warning",
                    f"{mr.height - n_eq} of {mr.height} MR rows differ",
                )
            )
        lag = ar.filter(pl.col("dimension") == "ARQ").select(
            (pl.col("datekey") - pl.col("calendardate")).dt.total_days().alias("d")
        )
        if lag.height:
            stats["filing_lag_days"] = {
                "p05": float(lag["d"].quantile(0.05)),
                "median": float(lag["d"].median()),
                "p95": float(lag["d"].quantile(0.95)),
                "max": int(lag["d"].max()),
            }
        stats["fundamentals_by_dimension"] = dict(
            f.group_by("dimension").len().sort("dimension").iter_rows()
        )
        stats["fundamentals_tickers"] = int(f["ticker"].n_unique())
        stats["calendardate_range"] = [str(f["calendardate"].min()), str(f["calendardate"].max())]
        if "tickers" in tables:
            missing = (
                f.select("ticker")
                .unique()
                .join(tables["tickers"].select("ticker").unique(), on="ticker", how="anti")
            )
            checks.append(
                Check(
                    "fundamentals tickers exist in tickers table",
                    missing.height == 0,
                    "error",
                    f"{missing.height} missing, e.g. {missing['ticker'].head(5).to_list()}",
                )
            )
        for dim_pair in (("ARQ", "MRQ"), ("ART", "MRT")):
            a = (
                f.filter(pl.col("dimension") == dim_pair[0])
                .select("ticker", "calendardate")
                .unique()
            )
            m = (
                f.filter(pl.col("dimension") == dim_pair[1])
                .select("ticker", "calendardate")
                .unique()
            )
            if a.height:
                stats[f"{dim_pair[1]}_matched_by_{dim_pair[0]}_pct"] = round(
                    100
                    * m.join(a, on=["ticker", "calendardate"], how="semi").height
                    / max(m.height, 1),
                    2,
                )
    if "stocks" in tables:
        s = tables["stocks"]
        pc = [
            c for c in ("open", "high", "low", "close", "closeadj", "closeunadj") if c in s.columns
        ]
        neg = s.filter(pl.any_horizontal([pl.col(c) < 0 for c in pc])).height if pc else 0
        checks.append(
            Check("stocks: prices non-negative", neg == 0, "error", f"{neg} negative rows")
        )
        stats["stocks_tickers"] = int(s["ticker"].n_unique())
        stats["stocks_date_range"] = [str(s["date"].min()), str(s["date"].max())]
    if "tickers" in tables and "category" in tables["tickers"].columns:
        stats["ticker_categories"] = dict(
            tables["tickers"]
            .unique("ticker")
            .group_by("category")
            .len()
            .sort("len", descending=True)
            .head(10)
            .iter_rows()
        )
        if "isdelisted" in tables["tickers"].columns:
            stats["delisted_tickers"] = int(
                tables["tickers"].unique("ticker").filter(pl.col("isdelisted") == "Y").height
            )
    return checks, stats


def validate(settings: Settings, raise_on_error: bool = True) -> dict[str, Any]:
    tables = _tables(settings)
    checks, stats = run_checks(tables)
    report = {"validated_at": utc_stamp(), "checks": [asdict(c) for c in checks], "coverage": stats}
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    (settings.processed_dir / "_validation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    for c in checks:
        log = (
            logger.info if c.passed else (logger.error if c.severity == "error" else logger.warning)
        )
        log(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    failed = [c for c in checks if not c.passed and c.severity == "error"]
    if failed and raise_on_error:
        raise ValidationError(
            f"{len(failed)} validation check(s) failed: {[c.name for c in failed]}"
        )
    return report
