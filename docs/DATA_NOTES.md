# Data notes: Sharadar (direct)

The checks below were made on 2026-10-27 against <https://sharadar.com/docs>,
<https://sharadar.com/llms.txt>, the public schema endpoint
`https://api.sharadar.com/v1.0/schema/<table>`, the `descriptions` table, and live calls made
with the documented public `test-api-key`. Where the original project spec disagreed with the
docs, **the docs win**, and the difference is recorded here.

## 1. Column mapping (`config/columns.yaml`)

Internally, every table uses *logical* names. `pitlab.data.ingest.normalize_table` applies the
physical → logical rename at ingest, and nowhere else.

| Table | Logical | Physical (verified) | Note |
|---|---|---|---|
| fundamentals | ticker, dimension, calendardate, reportperiod, lastupdated | same | |
| fundamentals | **datekey** | **`date`** | **Spec discrepancy.** The direct API calls the filing key `date` (Nasdaq Data Link called it `datekey`). Per `descriptions`, it is the SEC filing date for AR* dimensions and **equals `reportperiod` for MR* dimensions**. |
| fundamentals | netinc, revenue, equity, assets, ncfo, gp, sharesbas | same | Large integers arrive as strings in JSON. We download CSV. |
| stocks | ticker, date, close, closeadj, closeunadj, volume | same | `close` is split-adjusted. `closeadj` is also dividend- and spinoff-adjusted. `closeunadj` is raw. |
| daily | ticker, date, marketcap | same | **`marketcap` is in millions of USD** (AAPL ≈ 5,074,761 on 2026-10-23). |
| tickers | table, permaticker, ticker, category, isdelisted, sector, … | same | One row per (table, permaticker, ticker). `table` ∈ {stocks, fundamentals, insiders, …}. |
| sp500 | date, action, ticker | same | See §3. |
| actions | date, action, ticker, value, contraticker | same | |
| events | ticker, date, eventcodes | same | Pipe-separated codes (e.g. `22\|91`). |

## 2. API facts used by the client

- Base URL: `https://api.sharadar.com/v1.0`. The table route is `GET /data/{table}`, and legacy
  codes (`SF1`, `SEP`) also resolve. **This is not Nasdaq Data Link.**
- Auth: the `api_key` query parameter or the `x-api-key` header. **We send the header**, so the
  key never appears in URLs or logs. `401` means a bad key. `403 {"error":"Exceeds free tier"}`
  means the request ran as free tier.
- Parameters: `ticker` (comma list), `from`, `to`, `fields`, `limit` (default 10,000, no stated
  maximum), `skip`/`offset`, `sort` (only the first field applies), `format=csv|json` (CSV is
  the default), and operators `<col>.gt/.gte/.lt/.lte`.
- **`from` defaults to one year ago.** Without it you silently get a single year. The client
  always passes `from=1980-01-01`, *except* for snapshot tables (see quirk Q1).
- Bulk: the same route with `years=5|10|full` returns **302** to a pre-signed zip of CSVs. Bulk is
  subscription-only. `tickers` and `descriptions` publish only a full snapshot. The client
  follows the redirect **without** the API-key header.
- The docs ask not to page whole tables with `limit` + `skip`, and to use `ticker`/`from`/`to`
  slices instead. The REST fallback therefore slices by ticker batches.
- Rate limits are **not documented**. The client throttles to 4 requests/s and backs off
  exponentially on 429/5xx, honoring `Retry-After`.
- JSON responses are `{"count": n, "data": [...]}`.
- Public table schemas: `GET /schema/{table}?format=postgres|sqlite|mysql` (no key needed).

## 3. Quirks found (and how the code handles them)

| # | Quirk | Evidence | Handling |
|---|---|---|---|
| Q1 | **`from=` on `tickers` silently drops rows.** 74,481 → 33,129, including delisted names, which brings back survivorship bias. | live, test key | `SNAPSHOT_TABLES = {tickers, descriptions}` never get a date filter. Regression test: `test_snapshot_tables_never_get_a_date_filter` |
| Q2 | **`skip` paging is unstable.** Pages of 10,000 overlapped (4,276 duplicate keys) and missed rows. | live, test key | Requests use `limit=100000`, so a slice fits in one page. A full page logs a warning. Validation fails on duplicate primary keys. |
| Q3 | MR* rows have `date == reportperiod`, not a filing date. | descriptions + live | Arm B takes the filing date from the matching ARQ/ART row. Validation warns if it ever differs. |
| Q4 | `calendardate` is *normalized*. ARY rows map to Dec 31 even for September fiscal years, so annual `datekey < calendardate` is normal (AAPL: 5 rows). | live | Validation: `datekey >= reportperiod` is an error for all AR rows. `datekey >= calendardate` is an error for ARQ/ART only, and informational for ARY. (The spec asked for `datekey >= calendardate` everywhere.) |
| Q5 | `reportperiod` can precede `calendardate` (AAPL fiscal quarter ending 2026-06-27 → calendardate 2026-06-30). | live | Arm A uses `calendardate` as the spec requires. The gap is a few days. |
| Q6 | The `sp500` table holds quarterly `historical` snapshots plus a `current` snapshot. The test key showed no `added`/`removed` rows. | live | `universe.sp500_membership` supports both encodings, and events win when present. |
| Q7 | The **test key exposes only AAPL**, with about 5 years of history (stocks from 2021-10-26, fundamentals from calendardate 2021-09-30). The spec's "free tier = Dow 30, full history" could not be verified without a signed-up free key. | live | The free tier is detected by probing a non-Dow ticker (403). Accessible tickers come from `fundamentals?dimension=MRY&fields=ticker`. |
| Q8 | The `tickers` table returns the whole market (about 74k rows), even on the free tier. | live | It is used for metadata and categories. Data downloads cover only the accessible tickers. |

## 4. Storage and snapshot discipline

Paths below are relative to the data base: `$PITLAB_DATA_ROOT/pit-two-clocks` if the variable
is set (on the author's PC, `P:\quant_data\pit-two-clocks`), else the repo root.

```
data/raw/<table>/<UTC timestamp>/   raw CSV pages or bulk zip, unchanged
                                     + manifest.json (table, timestamp, row_count, sha256 per file,
                                       combined sha256, license_tier, method, history)
                                     + progress.json (REST resume state)
data/processed/<table>.parquet       typed, logical names, built from the latest snapshot
data/processed/_meta.json            data mode (SYNTHETIC / SHARADAR-FREE / SHARADAR-FULL), counts
data/processed/_validation.json      last validation report (shown in the dashboard)
data/pit.duckdb                      DuckDB views over the Parquet files
```

Snapshots are **never overwritten**. A new download creates a new timestamp folder, so over
time you accumulate your own vintage archive of Sharadar's restated values. A partial REST
snapshot can be resumed with `pitlab data download --resume <timestamp>`.

## 5. Coverage observed (test key, 2026-10-27)

| Table | Rows | Notes |
|---|---|---|
| tickers | 74,481 | whole market metadata |
| fundamentals | 87 | AAPL: ARQ 20, ART 20, ARY 5, MRQ 19, MRT 19, MRY 4 |
| stocks / daily | 1,254 each | AAPL, 2021-10-26 → 2026-10-23 |
| actions / sp500 / events | 20 / 21 / 53 | |
| descriptions | 430 | |

The ARQ filing lag for AAPL is 28 days (p5) / 31.5 days (median) / 35 days (p95). MRQ rows
are 100% matched by ARQ rows.

With a single ticker, no month reaches the 20-name minimum. `pitlab run` therefore completes
with empty statistics and a warning, and the dashboard shows the free-tier warning plus a
"too small" notice. With a signed-up free key (Dow 30), quintile portfolios of about 6 names per
leg will run, and the results are **illustrative only**.

To refresh these numbers: `pitlab data validate` writes `data/processed/_validation.json`
(coverage section), which the **Data health** page displays.

## 6. License

Sharadar data is licensed for **personal, non-professional use, with no redistribution**. This
repository never contains Sharadar rows: `.gitignore` covers `data/`, `*.parquet`, `*.duckdb`,
`.env`, and `outputs/`, and a pre-commit hook (`scripts/check_no_data.py`) blocks anything under
`data/`, data-file extensions, `.env`, and files over 5 MB. Figures in `docs/img/` are produced
by `scripts/make_figures.py`, which refuses to run on non-synthetic runs.

## Free tier, verified 2026-10-27

- 30 tickers: the current Dow 30 (survivors only). History runs from about 2021-10, which
  gives about 47 monthly rebalances after the 12-month warm-up.
- Every table answers for those tickers. `tickers` returns the full universe (about 74k rows).
  Any other ticker returns 403 `Exceeds free tier`. Bulk returns 403 `Bulk download requires
  a subscription that includes this table`.
- **Quirk Q9: calendardate rounds forward.** For off-calendar fiscal quarters (e.g. NKE,
  quarter ending Nov 30), calendardate is the *next* calendar quarter-end (Dec 31), and the
  10-Q can be filed before it (Dec 30). Validation treats this as info. The hard check is
  `datekey >= reportperiod`.

## From Sharadar's llms.txt (docs/sharadder/sharadder_llm.txt), reviewed 2026-10-27

- **Which plan this study needs.** The pipeline reads `fundamentals`, `daily`, `actions`,
  `events`, `sp500`, `tickers` (Fundamentals plan) **and** `stocks` (Prices plan only:
  `closeadj` drives returns, and `daily` has no prices). So it needs the **Bundle**, or
  Fundamentals + Prices. Personal-use prices as listed: Bundle 5Y $29/mo, 10Y $49/mo,
  Full History $69/mo ($499/yr). Fundamentals Full + Prices Full is $39 + $39, which is
  more than the Full Bundle. Check https://sharadar.com/subscribe before buying.
- **Bulk history must match the purchased tier** (`years=5|10|full`). `pitlab data download`
  defaults to `--history full`, so on a 5Y or 10Y plan pass `--history 5` or `--history 10`.
  `tickers` and `descriptions` always come back as a full snapshot.
- **Public schemas without a key:** `GET /schema/{table}?format=postgres`. All physical names in
  `config/columns.yaml` were checked against these schemas (all present). The fundamentals
  primary key is `(ticker, dimension, date, reportperiod)`, which matches validation.
- **Ticker identity:** `permaticker` is the stable security id. In the full tickers snapshot,
  ticker and permaticker are 1:1 within every table (0 conflicts). Reused tickers are
  disambiguated by renaming the *delisted* security with a numeric suffix (e.g. `PINC1`,
  `ARIS2`), so the ticker-keyed joins in this pipeline are safe.
- Legacy codes are accepted as table names (`SF1` = fundamentals, `SEP` = stocks), but the
  `tickers.table` column uses modern names (`fundamentals`, `stocks`, ...). `offset` is an
  alias of `skip`, which we don't use. Auth via query `api_key` or header `x-api-key` (we use
  the header).

## Incremental updates (paid key, verified 2026-10-27)

- `lastupdated.gte=` / `.lte=` filters work on `fundamentals`, `stocks`, and `daily`.
  `actions`, `events`, and `sp500` have no `lastupdated` column. They are small (bulk zips of
  2–12 MB), so they are re-pulled in full every day, along with `tickers` and `descriptions`.
- **Q10: unfiltered queries are capped.** Without a `ticker` parameter, `stocks` and `daily`
  return only rows dated within ~1 year and `fundamentals` within ~5 years, *regardless of*
  `from=` / `date.gte=`. With `ticker=` the full history is returned.
- **Q11: corporate actions rewrite history.** A split or dividend re-stamps `lastupdated` on
  every historical row of that ticker in `stocks` (adjusted prices change). Example: KO
  dividend on 2026-09-15 → all rows since 2015 have `lastupdated` 2026-09-15.
- Typical daily delta: fundamentals 300–1,200 rows, stocks 8–18k, daily 6–10k, plus the
  full histories of ~30–100 tickers with corporate actions (~120k rows). This fits in single
  100k-row requests. On overflow, the client splits the window by day, then by 200-ticker
  batches, and never pages with `skip`.
- Design: `pitlab data update` writes a `kind: delta` raw snapshot. `refresh_processed`
  layers deltas (newest full snapshot first, then deltas in timestamp order, upserted by
  primary key) and records applied deltas in `_meta.json`. The weekly
  `pitlab data update --full` re-bases on a fresh bulk download. That is the only path that
  catches row deletions and restatements outside the 5-year cap.
- Bulk zip sizes: fundamentals 622 MB, stocks 952 MB, daily 757 MB, tickers 4.9 MB, events
  11.5 MB.
- **Q12: 20 AR rows are filed before their period end** (vendor errors, 1996–2016). They are
  dropped by the pipeline. See METHODOLOGY conservative choices.
