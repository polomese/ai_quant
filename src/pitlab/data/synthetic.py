"""Synthetic Sharadar-shaped universe with *known, injected* look-ahead.

The generator produces the same tables, physical column names and quirks as the real
Sharadar API (e.g. MR rows carry ``date == reportperiod``), so everything downstream runs
unchanged. Two leaks are planted:

* **Timing leak.** Each firm-quarter has an earnings surprise ``s``. It is part of the
  period's reported net income / revenue / gross profit, but the market only learns it on
  the filing date (20-90 calendar days after period end), via a price jump ``k_surprise * s``.
  A strategy that treats data as known at period end (arm A) can front-run that jump.
* **Revision leak.** Each firm-quarter has a post-filing news shock ``d`` that drives a price
  drift over the ~63 trading days after filing. For "restater" firms, a share of quarters are
  later restated by an amount proportional to ``d``. The restated (MR) value therefore
  encodes future returns; the as-reported (AR) value does not. Arms A and B see it, C does not.

Book-to-price, asset growth and accruals carry no planted leak (equity and assets are never
restated and contain no surprise), so they act as controls: A ~= B ~= C expected.
A small genuine premium on persistent profitability gives arm C a little real IC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from loguru import logger

SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Industrials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Utilities",
    "Basic Materials",
    "Real Estate",
    "Communication Services",
]
RESTATEMENT_EVENT_CODE = 42  # Item 4.02 "non-reliance on previously issued financials"


@dataclass(frozen=True)
class SyntheticParams:
    n_stocks: int = 800
    years: int = 25
    end_year: int = 2024
    seed: int = 42
    # planted effects
    k_surprise: float = 0.06  # price jump at filing per 1 sd surprise
    surprise_scale: float = 0.020  # surprise in quarterly NI / GP / revenue, as share of assets
    k_drift: float = 0.15  # cumulative post-filing drift per 1 sd news shock
    drift_days: int = 63
    restater_share: float = 0.30  # firms that ever restate
    restate_prob: float = 0.50  # share of a restater's quarters that are restated
    restate_scale: float = (
        0.030  # restatement delta (NI, GP, revenue) as share of assets per 1 sd of d
    )
    genuine_premium: float = 0.02  # annual return per 1 sd of persistent ROA
    filing_lag_min: int = 20
    filing_lag_max: int = 90
    adr_share: float = 0.04  # non-common-stock names the universe filter must drop


def _business_days(start: str, end: str) -> np.ndarray:
    days = np.arange(
        np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"), dtype="datetime64[D]"
    )
    return days[np.is_busday(days)]


def _quarter_ends(start_year: int, end_year: int) -> np.ndarray:
    out = []
    for y in range(start_year, end_year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            out.append(np.datetime64(f"{y:04d}-{m:02d}-{d:02d}"))
    return np.array(out, dtype="datetime64[D]")


def generate(params: SyntheticParams | None = None) -> dict[str, pl.DataFrame]:
    """Return raw-shaped tables keyed by Sharadar table name."""
    p = params or SyntheticParams()
    rng = np.random.default_rng(p.seed)
    start_year = p.end_year - p.years + 1
    days = _business_days(f"{start_year}-01-01", f"{p.end_year}-12-31")
    n_d, n = len(days), p.n_stocks
    qends = _quarter_ends(start_year, p.end_year)
    n_q = len(qends)
    logger.info(f"synthetic: {n} stocks x {n_d} trading days x {n_q} quarters (seed={p.seed})")

    # ---- firm characteristics ------------------------------------------------------------
    tickers = np.array([f"SYN{i:04d}" for i in range(n)])
    sector = rng.choice(SECTORS, size=n)
    is_adr = rng.random(n) < p.adr_share
    beta = rng.uniform(0.7, 1.3, n)
    idio_vol = rng.uniform(0.015, 0.028, n)
    roa_perm = rng.normal(0.06, 0.03, n)  # annual
    roa_z = (roa_perm - roa_perm.mean()) / roa_perm.std()
    eq_ratio = rng.uniform(0.2, 0.7, n)
    turnover = rng.uniform(0.3, 1.5, n)
    margin = rng.uniform(0.2, 0.6, n)
    restater = rng.random(n) < p.restater_share

    # listing windows: 55% listed from the start, others IPO during the first 75% of sample
    start_idx = np.where(rng.random(n) < 0.55, 0, rng.integers(0, int(n_d * 0.75), n))
    mna_idx = np.where(rng.random(n) < 0.25, rng.integers(0, n_d, n), n_d)  # M&A exit day
    mna_idx = np.maximum(mna_idx, start_idx + 260)

    # ---- quarter-level shocks and filing dates -------------------------------------------
    surprise = rng.normal(0, 1, (n_q, n))
    news = rng.normal(0, 1, (n_q, n))
    lag_days = rng.integers(p.filing_lag_min, p.filing_lag_max + 1, (n_q, n))
    filing_dates = qends[:, None] + lag_days.astype("timedelta64[D]")
    filing_idx = np.searchsorted(days, filing_dates, side="left")  # roll weekends forward
    qend_idx = np.searchsorted(days, qends, side="right") - 1  # last trading day <= q end

    # ---- daily returns with planted effects ------------------------------------------------
    mkt = rng.normal(0.0003, 0.010, n_d)
    ret = mkt[:, None] * beta[None, :] + rng.normal(0, 1, (n_d, n)) * idio_vol[None, :]
    ret += (p.genuine_premium * roa_z / 252.0)[None, :]
    cols = np.arange(n)
    for q in range(n_q):
        fi = filing_idx[q]
        ok = fi < n_d
        ret[fi[ok], cols[ok]] += p.k_surprise * surprise[q, ok]
        per_day = p.k_drift * news[q] / p.drift_days
        for k in range(1, p.drift_days + 1):
            idx = fi + k
            ok = idx < n_d
            ret[idx[ok], cols[ok]] += per_day[ok]
    ret = np.clip(ret, -0.6, 1.0)

    # ---- prices, listing and delisting -----------------------------------------------------
    p0 = rng.uniform(12, 80, n)
    log_px = np.cumsum(np.log1p(ret), axis=0)
    log_px -= log_px[start_idx, cols][None, :]
    close = p0[None, :] * np.exp(log_px)
    day_ix = np.arange(n_d)[:, None]
    alive = day_ix >= start_idx[None, :]
    below = (close < 1.0) & alive
    perf_idx = np.where(below.any(axis=0), below.argmax(axis=0), n_d)
    end_idx = np.minimum(perf_idx, mna_idx)  # last trading day index (inclusive) if < n_d
    delist_reason = np.where(
        end_idx >= n_d, "", np.where(perf_idx <= mna_idx, "delisted", "acquisitionby")
    )
    end_idx = np.minimum(end_idx, n_d - 1)
    alive &= day_ix <= end_idx[None, :]
    is_delisted = delist_reason != ""

    # ---- fundamentals (as-reported) built quarter by quarter ------------------------------
    equity0 = np.exp(rng.normal(np.log(1e9), 1.3, n))
    assets = equity0 / eq_ratio
    mcap0 = equity0 * np.exp(rng.normal(0.4, 0.5, n))
    shares = mcap0 / p0
    roa_dev = np.zeros(n)
    recs: list[dict[str, np.ndarray]] = []
    prev_q_close = close[0]
    for q in range(n_q):
        qi = qend_idx[q]
        q_close = close[qi]
        q_ret = q_close / np.where(prev_q_close > 0, prev_q_close, q_close) - 1
        prev_q_close = q_close
        # assets track the firm's value (so ratios stay stationary) plus noise
        growth = 0.004 + 0.6 * np.clip(q_ret, -0.5, 0.5) + rng.normal(0, 0.03, n)
        assets = assets * np.exp(np.clip(growth, -0.4, 0.4))
        roa_dev = 0.6 * roa_dev + rng.normal(0, 0.004, n)
        roa_q = roa_perm / 4 + roa_dev
        equity = assets * np.clip(eq_ratio + rng.normal(0, 0.01, n), 0.05, 0.95)
        ni_base = assets * roa_q
        shock = assets * p.surprise_scale * surprise[q]
        # the surprise is real cash: it hits NI and operating cash flow alike, so accruals stay clean
        netinc = ni_base + shock
        ncfo = ni_base + shock + assets * rng.normal(0, 0.004, n)
        rev = assets * turnover / 4 * (1 + rng.normal(0, 0.02, n)) + 2 * shock
        gp = rev * margin + shock
        shares = shares * np.exp(rng.normal(0.002, 0.004, n))
        # restatement: proportional to the post-filing news shock (future returns)
        restated = restater & (rng.random(n) < p.restate_prob)
        delta = assets * p.restate_scale * news[q]
        d_ni = np.where(restated, delta, 0.0)
        d_rev = np.where(restated, 2 * delta, 0.0)
        d_gp = np.where(restated, delta, 0.0)
        restate_lag = rng.integers(120, 400, n)
        listed = (start_idx <= qi) & (filing_idx[q] <= end_idx) & (filing_idx[q] < n_d)
        recs.append(
            dict(
                q=np.full(n, q),
                listed=listed,
                netinc=netinc,
                revenue=rev,
                gp=gp,
                ncfo=ncfo,
                equity=equity,
                assets=assets.copy(),
                sharesbas=shares.copy(),
                d_ni=d_ni,
                d_rev=d_rev,
                d_gp=d_gp,
                restated=restated,
                restate_lag=restate_lag,
            )
        )

    tables = _assemble(
        p,
        tickers,
        sector,
        is_adr,
        restater,
        days,
        qends,
        filing_idx,
        start_idx,
        end_idx,
        delist_reason,
        is_delisted,
        close,
        alive,
        recs,
    )
    tables["_truth"] = pl.DataFrame(
        {"param": list(asdict(p)), "value": [str(v) for v in asdict(p).values()]}
    )
    return tables


def _assemble(
    p,
    tickers,
    sector,
    is_adr,
    restater,
    days,
    qends,
    filing_idx,
    start_idx,
    end_idx,
    delist_reason,
    is_delisted,
    close,
    alive,
    recs,
) -> dict[str, pl.DataFrame]:
    n_d, n = close.shape
    days.astype(str)

    # ---- fundamentals: ARQ / MRQ / ART / MRT ------------------------------------------------
    q_frames = []
    for q, r in enumerate(recs):
        m = r["listed"]
        if not m.any():
            continue
        fi = filing_idx[q][m]
        q_frames.append(
            pl.DataFrame(
                {
                    "ticker": tickers[m],
                    "q": np.full(m.sum(), q),
                    "calendardate": np.full(m.sum(), qends[q]),
                    "filing": days[fi],
                    "netinc": r["netinc"][m],
                    "revenue": r["revenue"][m],
                    "gp": r["gp"][m],
                    "ncfo": r["ncfo"][m],
                    "equity": r["equity"][m],
                    "assets": r["assets"][m],
                    "sharesbas": r["sharesbas"][m],
                    "d_ni": r["d_ni"][m],
                    "d_rev": r["d_rev"][m],
                    "d_gp": r["d_gp"][m],
                    "restated": r["restated"][m],
                    "restate_date": days[fi] + r["restate_lag"][m].astype("timedelta64[D]"),
                }
            )
        )
    max_day = days[-1]
    fq = pl.concat(q_frames).sort(["ticker", "q"])
    # a restatement only exists in the data if it happened before the end of the sample
    fq = fq.with_columns(
        (pl.col("restated") & (pl.col("restate_date") <= max_day)).alias("restated")
    )
    fq = fq.with_columns(
        pl.when(pl.col("restated"))
        .then(pl.col("restate_date"))
        .otherwise(pl.col("filing"))
        .alias("lu_mr"),
        *[
            pl.when(pl.col("restated")).then(pl.col(d)).otherwise(0.0).alias(d)
            for d in ("d_ni", "d_rev", "d_gp")
        ],
    )
    flow = ["netinc", "revenue", "gp", "ncfo"]
    fq = fq.with_columns(
        (pl.col("netinc") + pl.col("d_ni")).alias("netinc_mr"),
        (pl.col("revenue") + pl.col("d_rev")).alias("revenue_mr"),
        (pl.col("gp") + pl.col("d_gp")).alias("gp_mr"),
        (pl.col("ncfo") + pl.col("d_ni")).alias("ncfo_mr"),  # restatement reclassifies cash too
    )
    # trailing sums over the last 4 *consecutive* quarters
    consec = (pl.col("q") - pl.col("q").shift(3).over("ticker")) == 3
    ttm = []
    for c in flow:
        ttm.append(pl.col(c).rolling_sum(4).over("ticker").alias(f"{c}_ttm"))
        ttm.append(pl.col(f"{c}_mr").rolling_sum(4).over("ticker").alias(f"{c}_mr_ttm"))
    lu_days = pl.col("lu_mr").cast(pl.Int32).rolling_max(4).over("ticker")
    fq = fq.with_columns(*ttm, consec.alias("has_ttm"), lu_days.cast(pl.Date).alias("lu_mr_ttm"))

    def dim_frame(dim: str) -> pl.DataFrame:
        is_mr = dim.startswith("MR")
        is_ttm = dim.endswith("T")
        src = fq.filter(pl.col("has_ttm")) if is_ttm else fq
        sfx = ("_mr" if is_mr else "") + ("_ttm" if is_ttm else "")
        vals = {c: pl.col(f"{c}{sfx}") for c in flow}
        if is_mr:
            lastupd = pl.col("lu_mr_ttm" if is_ttm else "lu_mr")
            date_col = pl.col("calendardate")  # Sharadar quirk: MR date == reportperiod
        else:
            lastupd = pl.col("filing")
            date_col = pl.col("filing")
        return src.select(
            pl.col("ticker"),
            pl.lit(dim).alias("dimension"),
            pl.col("calendardate"),
            date_col.alias("date"),
            pl.col("calendardate").alias("reportperiod"),
            lastupd.alias("lastupdated"),
            *[v.round(0).cast(pl.Int64).alias(k) for k, v in vals.items()],
            pl.col("equity").round(0).cast(pl.Int64),
            pl.col("assets").round(0).cast(pl.Int64),
            pl.col("sharesbas").round(0).cast(pl.Int64),
        )

    fundamentals = pl.concat([dim_frame(d) for d in ("ARQ", "MRQ", "ART", "MRT")])
    fundamentals = fundamentals.with_columns(
        pl.col("calendardate").dt.strftime("%Y-%m").alias("fiscalperiod")
    ).sort(["ticker", "dimension", "calendardate"])

    # ---- shares outstanding by day (for daily.marketcap) via latest filed quarter ----------
    shares_q = np.full((len(qends), n), np.nan)
    for q, r in enumerate(recs):
        shares_q[q] = np.where(r["listed"], r["sharesbas"], np.nan)
    shares_daily = np.full((n_d, n), np.nan)
    for q in range(len(qends)):
        fi = filing_idx[q]
        for i in np.nonzero(~np.isnan(shares_q[q]) & (fi < n_d))[0]:
            shares_daily[fi[i] :, i] = shares_q[q, i]
    # before the first filing, use the first reported share count
    first = np.where(np.isnan(shares_q), np.inf, shares_q)
    first_valid = np.nanmin(np.where(np.isfinite(first), first, np.nan), axis=0)
    shares_daily = np.where(np.isnan(shares_daily), first_valid[None, :], shares_daily)

    # ---- stocks + daily --------------------------------------------------------------------
    di, si = np.nonzero(alive)
    px = close[di, si]
    rng = np.random.default_rng(p.seed + 1)
    vol = np.round(rng.lognormal(12, 1, len(px)))
    stocks = pl.DataFrame(
        {
            "ticker": tickers[si],
            "date": days[di],
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": vol,
            "closeadj": px,
            "closeunadj": px,
            "lastupdated": days[di],
        }
    ).with_columns(
        pl.col(c).round(4) for c in ("open", "high", "low", "close", "closeadj", "closeunadj")
    )
    daily = pl.DataFrame(
        {
            "ticker": tickers[si],
            "date": days[di],
            "lastupdated": days[di],
            "marketcap": np.round(px * shares_daily[di, si] / 1e6, 1),
        }
    )

    # ---- tickers ----------------------------------------------------------------------------
    first_px = days[start_idx]
    last_px = days[end_idx]
    base = pl.DataFrame(
        {
            "permaticker": np.arange(100000, 100000 + n),
            "ticker": tickers,
            "name": [f"SYNTHETIC CORP {i}" for i in range(n)],
            "exchange": np.where(np.arange(n) % 3 == 0, "NASDAQ", "NYSE"),
            "isdelisted": np.where(is_delisted, "Y", "N"),
            "category": np.where(is_adr, "ADR Common Stock", "Domestic Common Stock"),
            "sector": sector,
            "industry": sector,
            "currency": "USD",
            "firstpricedate": first_px,
            "lastpricedate": last_px,
        }
    )
    tickers_tbl = pl.concat(
        [base.with_columns(pl.lit(t).alias("table")) for t in ("stocks", "fundamentals")]
    ).select("table", *base.columns)

    # ---- actions (delistings) ---------------------------------------------------------------
    dl = np.nonzero(is_delisted)[0]
    actions = pl.DataFrame(
        {
            "date": days[end_idx[dl]],
            "action": delist_reason[dl],
            "ticker": tickers[dl],
            "name": [f"SYNTHETIC CORP {i}" for i in dl],
            "value": np.full(len(dl), np.nan),
            "contraticker": np.where(delist_reason[dl] == "acquisitionby", "SYNACQ", "N/A"),
            "contraname": np.where(
                delist_reason[dl] == "acquisitionby", "SYNTHETIC ACQUIRER", "N/A"
            ),
        }
    )

    # ---- sp500: quarterly 'historical' snapshots of the largest 40% by market cap ----------
    snaps = []
    mcap_q = close * shares_daily
    for q in range(len(qends)):
        qi = int(np.searchsorted(days, qends[q], side="right") - 1)
        live = alive[qi] & ~is_adr
        if live.sum() == 0:
            continue
        cut = np.quantile(mcap_q[qi][live], 0.6)
        members = np.nonzero(live & (mcap_q[qi] >= cut))[0]
        snaps.append(
            pl.DataFrame(
                {
                    "date": np.full(len(members), qends[q]),
                    "action": "historical",
                    "ticker": tickers[members],
                    "name": [f"SYNTHETIC CORP {i}" for i in members],
                    "contraticker": "N/A",
                    "contraname": "N/A",
                    "note": None,
                },
                schema_overrides={"note": pl.String},
            )
        )
    sp500 = pl.concat(snaps)

    # ---- events: filings (22|91) and restatement non-reliance (42) -------------------------
    ev_fil = fq.select(
        "ticker", pl.col("filing").alias("date"), pl.lit("22|91").alias("eventcodes")
    )
    ev_res = fq.filter(pl.col("restated")).select(
        "ticker",
        pl.col("restate_date").alias("date"),
        pl.lit(str(RESTATEMENT_EVENT_CODE)).alias("eventcodes"),
    )
    events = pl.concat([ev_fil, ev_res]).unique(["ticker", "date"], keep="last")

    descriptions = pl.DataFrame(
        {
            "table": ["fundamentals"] * 4,
            "indicator": ["dimension", "date", "calendardate", "reportperiod"],
            "isfilter": ["Y", "Y", "Y", "N"],
            "isprimarykey": ["Y", "Y", "N", "Y"],
            "title": ["Dimension", "Date Key", "Calendar Date", "Report Period"],
            "description": [
                "ARQ/ART as-reported; MRQ/MRT most-recent (restated).",
                "SEC filing date for AR dimensions; equals reportperiod for MR dimensions.",
                "Normalized calendar quarter end.",
                "Actual fiscal period end.",
            ],
            "unittype": ["text", "date", "date", "date"],
        }
    )
    logger.info(
        f"synthetic: fundamentals={fundamentals.height:,} stocks={stocks.height:,} "
        f"delisted={int(is_delisted.sum())} restaters={int(restater.sum())}"
    )
    return {
        "tickers": tickers_tbl,
        "fundamentals": fundamentals,
        "stocks": stocks,
        "daily": daily,
        "actions": actions,
        "sp500": sp500,
        "events": events,
        "descriptions": descriptions,
    }
