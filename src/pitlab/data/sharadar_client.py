"""Sharadar direct API client (https://api.sharadar.com/v1.0) - bulk first, REST fallback.

Follows https://sharadar.com/docs (read 2026-10-27, see docs/DATA_NOTES.md):

* auth: ``x-api-key`` header (also accepted: ``api_key`` query param). We use the header so
  the key never appears in URLs or logs.
* data: ``GET /data/{table}`` with ``ticker`` (comma list), ``from``/``to``, ``fields``,
  ``limit`` (default 10000), ``skip``, ``sort``, ``format=csv|json``, operators
  ``<col>.gte`` etc. ``from`` DEFAULTS TO ONE YEAR AGO, so we always pass it explicitly.
* bulk: same route with ``years=5|10|full`` -> HTTP 302 to a pre-signed zip (subscription
  only). The docs ask not to page whole tables with skip; REST fallback therefore slices by
  ticker batches and pages within each slice.
* free tier: 403 ``Exceeds free tier`` for tickers outside the sample universe.

Rate limits are not documented; we throttle to ``min_interval`` seconds between calls and
back off exponentially on 429 / 5xx / network errors (honouring ``Retry-After``).
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import requests
from loguru import logger

from pitlab.data.ingest import utc_stamp, write_manifest

BASE_URL = "https://api.sharadar.com/v1.0"
EARLIEST = "1980-01-01"
#: snapshot tables: NEVER send a date filter. Verified 2026-10-27: `from=1980-01-01` on
#: `tickers` silently drops 41k of 74k rows (incl. delisted names -> survivorship bias).
SNAPSHOT_TABLES = {"tickers", "descriptions"}
#: tables that are small enough to page whole (no ticker filter needed)
SMALL_TABLES = {"tickers", "descriptions", "sp500", "actions"}
#: tickers per REST request for ticker-sliced tables (rows per ticker differ a lot)
BATCH = {"fundamentals": 10, "stocks": 3, "daily": 3, "events": 50}
#: a large, never-in-the-Dow ticker used to probe the licence tier
TIER_PROBE_TICKER = "TSLA"


class SharadarError(RuntimeError):
    pass


class AuthError(SharadarError):
    pass


class FreeTierError(SharadarError):
    pass


class SharadarClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        min_interval: float = 0.25,
        max_retries: int = 6,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(
            {"x-api-key": api_key, "User-Agent": "pitlab/0.1 (personal research)"}
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last = 0.0

    # -- low level ---------------------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        allow_redirects: bool = True,
        stream: bool = False,
    ) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    allow_redirects=allow_redirects,
                    stream=stream,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == self.max_retries:
                    raise SharadarError(f"network error on {path}: {e}") from e
                self._sleep(attempt, None, f"network error {e.__class__.__name__}")
                continue
            if r.status_code == 401:
                raise AuthError(
                    "401 Unauthorized - check SHARADAR_API_KEY (https://sharadar.com/account)"
                )
            if r.status_code == 403:
                body = r.text[:200]
                if "free tier" in body.lower():
                    raise FreeTierError(body)
                raise SharadarError(f"403 Forbidden on {path}: {body}")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    raise SharadarError(f"{r.status_code} on {path} after {attempt} retries")
                self._sleep(attempt, r.headers.get("Retry-After"), f"HTTP {r.status_code}")
                continue
            return r
        raise SharadarError("unreachable")  # pragma: no cover

    def _sleep(self, attempt: int, retry_after: str | None, why: str) -> None:
        delay = min(2.0**attempt, 60.0)
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        logger.warning(
            f"{why}; retrying in {delay:.0f}s (attempt {attempt + 1}/{self.max_retries})"
        )
        time.sleep(delay)

    # -- tier ----------------------------------------------------------------------------------
    def detect_tier(self) -> str:
        """'free' if a non-sample ticker is refused, else 'paid'."""
        try:
            self.get(
                "data/fundamentals",
                {"ticker": TIER_PROBE_TICKER, "limit": 1, "from": EARLIEST, "format": "csv"},
            )
            return "paid"
        except FreeTierError:
            return "free"

    def free_tier_tickers(self) -> list[str]:
        """Tickers the key can actually see (fundamentals rows exist for them)."""
        r = self.get(
            "data/fundamentals",
            {
                "dimension": "MRY",
                "fields": "ticker",
                "from": EARLIEST,
                "limit": 10000,
                "format": "csv",
            },
        )
        df = pl.read_csv(io.BytesIO(r.content), infer_schema=False)
        return sorted(df["ticker"].unique().to_list()) if df.height else []

    # -- bulk ----------------------------------------------------------------------------------
    def bulk_download(self, table: str, dest: Path, history: str = "full") -> Path | None:
        """Download the bulk zip for ``table``; returns the file or None if not available."""
        try:
            r = self.get(f"data/{table}", {"years": history}, allow_redirects=False)
        except (AuthError, FreeTierError, SharadarError) as e:
            logger.info(f"bulk not available for {table}: {e}")
            return None
        if r.status_code not in (301, 302, 303, 307, 308) or "Location" not in r.headers:
            logger.info(f"bulk not available for {table}: HTTP {r.status_code}")
            return None
        # follow the pre-signed URL WITHOUT our API key header (different host)
        with requests.get(r.headers["Location"], stream=True, timeout=self.timeout) as z:
            z.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in z.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        logger.info(f"bulk {table}: {dest.stat().st_size / 1e6:.1f} MB")
        return dest

    # -- REST ----------------------------------------------------------------------------------
    def fetch_pages(
        self, table: str, params: dict[str, Any], page_size: int = 100_000
    ) -> Iterable[bytes]:
        """Yield raw CSV pages for one slice.

        Requests use a large ``limit`` so a slice normally fits in ONE page: skip-paging was
        observed to be unstable (overlapping / missing rows) on 2026-10-27, and the docs warn
        against it. If a page is full we still page with ``skip`` but log a warning.
        """
        skip = 0
        while True:
            q = {**params, "format": "csv", "limit": page_size, "skip": skip}
            if table not in SNAPSHOT_TABLES:
                q.setdefault("from", EARLIEST)  # the API defaults `from` to one year ago
            r = self.get(f"data/{table}", q)
            body = r.content
            n = max(body.count(b"\n") - 1, 0) if body.strip() else 0
            if body and not body.endswith(b"\n"):
                n += 1
            yield body
            if n < page_size:
                return
            logger.warning(
                f"{table} slice {params.get('ticker', 'all')[:40]} filled a {page_size}-row page; "
                "paging with skip (may be unstable - consider smaller ticker batches)"
            )
            skip += page_size


def _rows_in_csv(body: bytes) -> int:
    if not body.strip():
        return 0
    return len(pl.read_csv(io.BytesIO(body), infer_schema=False))


def _load_progress(snap: Path) -> dict[str, Any]:
    p = snap / "progress.json"
    return (
        json.loads(p.read_text(encoding="utf-8"))
        if p.exists()
        else {"done": [], "skipped": [], "rows": 0}
    )


def _save_progress(snap: Path, prog: dict[str, Any]) -> None:
    (snap / "progress.json").write_text(json.dumps(prog), encoding="utf-8")


def rest_download(client: SharadarClient, table: str, snap: Path, tickers: list[str] | None) -> int:
    """Resumable REST download of one table into ``snap`` (one CSV file per page)."""
    snap.mkdir(parents=True, exist_ok=True)
    prog = _load_progress(snap)
    if table in SMALL_TABLES or not tickers:
        slices = [("all", {})] if table in SMALL_TABLES else []
        if table in SMALL_TABLES and tickers and table != "descriptions":
            slices = [
                (f"t{i:05d}", {"ticker": ",".join(tickers[i : i + 50])})
                for i in range(0, len(tickers), 50)
            ]
            if table == "tickers":
                slices = [("all", {})]  # metadata for the whole market is always useful
    else:
        b = BATCH.get(table, 10)
        slices = [
            (f"t{i:05d}", {"ticker": ",".join(tickers[i : i + b])})
            for i in range(0, len(tickers), b)
        ]
    total = prog["rows"]
    for k, (sid, params) in enumerate(slices):
        if sid in prog["done"]:
            continue
        try:
            pages = list(client.fetch_pages(table, params))
        except FreeTierError:
            prog["skipped"].append(sid)
            prog["done"].append(sid)
            _save_progress(snap, prog)
            continue
        rows = 0
        for j, body in enumerate(pages):
            (snap / f"{table}_{sid}_{j:04d}.csv").write_bytes(body)
            rows += _rows_in_csv(body)
        total += rows
        prog["done"].append(sid)
        prog["rows"] = total
        _save_progress(snap, prog)
        if k % 25 == 0 or k == len(slices) - 1:
            logger.info(
                f"{table}: slice {k + 1}/{len(slices)} ({sid}) +{rows:,} rows, total {total:,}"
            )
    return total


def download_tables(
    client: SharadarClient,
    raw_dir: Path,
    tables: list[str],
    method: str = "auto",
    history: str = "full",
    tickers: list[str] | None = None,
    resume: str | None = None,
    categories: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Download ``tables`` into raw_dir/<table>/<timestamp>/ and write manifests.

    Existing complete snapshots are never overwritten; ``resume`` continues a partial one.
    """
    stamp = resume or utc_stamp()
    tier = client.detect_tier()
    logger.info(f"licence tier detected: {tier}; snapshot {stamp}")
    if tier == "free" and tickers is None:
        tickers = client.free_tier_tickers()
        logger.info(f"free tier: {len(tickers)} accessible tickers: {tickers}")
    manifests = {}
    order = ["tickers"] + [t for t in tables if t != "tickers"]
    for table in [t for t in order if t in tables]:
        snap = raw_dir / table / stamp
        if (snap / "manifest.json").exists():
            logger.info(f"{table}: snapshot {stamp} already complete - skipping")
            continue
        snap.mkdir(parents=True, exist_ok=True)
        used = None
        if method in ("auto", "bulk") and tier == "paid":
            z = client.bulk_download(table, snap / f"{table}.zip", history)
            if z is not None:
                used = "bulk"
                rows = _count_zip_rows(z)
            elif method == "bulk":
                raise SharadarError(f"bulk download unavailable for {table}")
        if used is None:
            if tickers is None and table not in SMALL_TABLES:
                tickers = _universe_from_raw(raw_dir, stamp, categories)
            rows = rest_download(client, table, snap, tickers)
            used = "rest"
        manifests[table] = write_manifest(
            snap, table, rows, tier, {"method": used, "history": history}
        )
        logger.info(f"{table}: {rows:,} rows via {used} -> {snap}")
    return manifests


def _count_zip_rows(path: Path) -> int:
    import zipfile

    n = 0
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                n += max(sum(1 for _ in f) - 1, 0)
    return n


def _universe_from_raw(raw_dir: Path, stamp: str, categories: list[str] | None) -> list[str]:
    """Ticker list for REST slicing, from the tickers snapshot just downloaded."""
    from pitlab.data.ingest import read_raw_snapshot

    snap = raw_dir / "tickers" / stamp
    df = read_raw_snapshot(snap)
    if "table" in df.columns:
        df = df.filter(pl.col("table").is_in(["fundamentals", "SF1", "stocks", "SEP"]))
    if categories and "category" in df.columns:
        df = df.filter(pl.col("category").is_in(categories))
    return sorted(df["ticker"].unique().to_list())
