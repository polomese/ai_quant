"""raw -> typed Parquet (data/processed) -> DuckDB views (data/pit.duckdb), plus manifests.

Internally every table uses *logical* column names from ``config/columns.yaml``; the
physical -> logical rename happens here and nowhere else.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from pitlab.config import Settings, load_columns

TABLES = [
    "tickers",
    "fundamentals",
    "stocks",
    "daily",
    "actions",
    "sp500",
    "events",
    "descriptions",
]
DATE_COLS = {
    "fundamentals": ["calendardate", "datekey", "reportperiod", "lastupdated"],
    "stocks": ["date", "lastupdated"],
    "daily": ["date", "lastupdated"],
    "tickers": [
        "firstpricedate",
        "lastpricedate",
        "lastupdated",
        "firstadded",
        "firstquarter",
        "lastquarter",
    ],
    "actions": ["date"],
    "sp500": ["date"],
    "events": ["date"],
}
FLOAT_COLS = {
    "stocks": ["open", "high", "low", "close", "closeadj", "closeunadj", "volume"],
    "daily": ["marketcap", "ev", "evebit", "evebitda", "pb", "pe", "ps"],
    "fundamentals": ["netinc", "revenue", "equity", "assets", "ncfo", "gp", "sharesbas"],
}
MODES = ("SYNTHETIC", "SHARADAR-FREE", "SHARADAR-FULL")
#: logical primary keys used to upsert incremental (delta) snapshots
UPSERT_KEYS = {
    "fundamentals": ["ticker", "dimension", "datekey", "reportperiod"],
    "stocks": ["ticker", "date"],
    "daily": ["ticker", "date"],
}


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rename_map(table: str, root: Path | None = None) -> dict[str, str]:
    """physical -> logical names for ``table``."""
    cols = load_columns(root).get(table, {})
    return {phys: logical for logical, phys in cols.items() if phys != logical}


def normalize_table(table: str, df: Any, root: Path | None = None) -> Any:
    """Rename to logical names and cast dates / numerics. Unknown columns pass through.

    Works on a DataFrame or a LazyFrame (bulk ingest streams the big tables)."""
    schema = df.collect_schema()
    ren = {k: v for k, v in rename_map(table, root).items() if k in schema}
    df = df.rename(ren)
    schema = df.collect_schema()
    casts = []
    for c in DATE_COLS.get(table, []):
        if c in schema and schema[c] != pl.Date:
            if schema[c] == pl.String:
                casts.append(pl.col(c).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False))
            else:
                casts.append(pl.col(c).cast(pl.Date))
    for c in FLOAT_COLS.get(table, []):
        if c in schema and schema[c] != pl.Float64:
            casts.append(pl.col(c).cast(pl.Float64, strict=False))
    return df.with_columns(casts) if casts else df


def read_raw_snapshot(snapshot_dir: Path) -> pl.DataFrame:
    """Read every CSV (or zipped CSV) file in a raw snapshot directory into one frame."""
    frames: list[pl.DataFrame] = []
    files = sorted(p for p in snapshot_dir.iterdir() if p.suffix.lower() in (".csv", ".zip"))
    for f in files:
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.lower().endswith(".csv"):
                        frames.append(_read_csv_bytes(z.read(name)))
        else:
            frames.append(_read_csv_bytes(f.read_bytes()))
    frames = [f for f in frames if f.height]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _read_csv_bytes(b: bytes) -> pl.DataFrame:
    # read everything as text, cast deliberately in normalize_table (no silent inference)
    return pl.read_csv(io.BytesIO(b), infer_schema=False)


def _cast_text_numbers(df: Any, table: str) -> Any:
    """Raw CSVs are read as text; cast numeric-looking columns that are not dates/ids."""
    keep_text = {
        "ticker",
        "dimension",
        "table",
        "name",
        "exchange",
        "isdelisted",
        "category",
        "cusips",
        "sicsector",
        "sicindustry",
        "figi",
        "famaindustry",
        "sector",
        "industry",
        "scalemarketcap",
        "scalerevenue",
        "relatedtickers",
        "currency",
        "location",
        "secfilings",
        "companysite",
        "action",
        "contraticker",
        "contraname",
        "note",
        "eventcodes",
        "indicator",
        "isfilter",
        "isprimarykey",
        "title",
        "description",
        "unittype",
        "fiscalperiod",
    }
    dates = set(DATE_COLS.get(table, [])) | {"date"}
    casts = []
    for c, dt in df.collect_schema().items():
        if dt == pl.String and c not in keep_text and c not in dates:
            casts.append(pl.col(c).cast(pl.Float64, strict=False))
    return df.with_columns(casts) if casts else df


def write_manifest(
    snapshot_dir: Path, table: str, rows: int, tier: str, extra: dict[str, Any] | None = None
) -> dict:
    files = sorted(p for p in snapshot_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "table": table,
        "timestamp": snapshot_dir.name,
        "row_count": rows,
        "license_tier": tier,
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files
        ],
        **(extra or {}),
    }
    manifest["sha256"] = hashlib.sha256(
        "".join(f["sha256"] for f in manifest["files"]).encode()
    ).hexdigest()
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(snap: Path) -> dict[str, Any]:
    return json.loads((snap / "manifest.json").read_text(encoding="utf-8"))


def snapshot_kind(snap: Path) -> str:
    """'full' (bulk or complete REST pull) or 'delta' (rows changed since a watermark)."""
    return read_manifest(snap).get("kind", "full")


def complete_snapshots(raw_dir: Path, table: str, kind: str | None = None) -> list[Path]:
    d = raw_dir / table
    if not d.exists():
        return []
    snaps = sorted(p for p in d.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    return [p for p in snaps if kind is None or snapshot_kind(p) == kind]


def latest_snapshot(raw_dir: Path, table: str) -> Path | None:
    """Newest complete FULL snapshot (deltas are layered on top by ``refresh_processed``)."""
    snaps = complete_snapshots(raw_dir, table, "full")
    return snaps[-1] if snaps else None


def delta_snapshots(raw_dir: Path, table: str, after: str) -> list[Path]:
    """Complete delta snapshots with a timestamp strictly after ``after`` (oldest first)."""
    return [p for p in complete_snapshots(raw_dir, table, "delta") if p.name > after]


def upsert(base: pl.DataFrame, delta: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Rows of ``delta`` replace rows of ``base`` with the same key; new keys are appended."""
    if delta.height == 0:
        return base
    delta = delta.unique(subset=keys, keep="last", maintain_order=True)
    kept = base.join(delta.select(keys), on=keys, how="anti", nulls_equal=True)
    return pl.concat([kept, delta], how="diagonal_relaxed")


def write_processed(
    settings: Settings, tables: dict[str, pl.DataFrame], mode: str, source: dict[str, Any]
) -> None:
    """Normalize + write Parquet for each table, register DuckDB views, write _meta.json."""
    assert mode in MODES, mode
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in tables.items():
        if name.startswith("_"):
            df.write_parquet(out / f"{name}.parquet")
            continue
        norm = normalize_table(name, df, settings.root)
        norm.write_parquet(out / f"{name}.parquet")
        counts[name] = norm.height
        logger.info(f"processed {name}: {norm.height:,} rows -> {out / (name + '.parquet')}")
    meta = {
        "mode": mode,
        "created_at": utc_stamp(),
        "row_counts": counts,
        "n_fundamentals_tickers": int(tables["fundamentals"]["ticker"].n_unique())
        if "fundamentals" in tables
        else 0,
        "source": source,
    }
    (out / "_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    register_duckdb(settings)


def register_duckdb(settings: Settings) -> None:
    db = settings.duckdb_path
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    try:
        for p in sorted(settings.processed_dir.glob("*.parquet")):
            if p.stem.startswith("_"):
                continue
            path = p.resolve().as_posix().replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {p.stem} AS SELECT * FROM read_parquet('{path}')")
        logger.info(f"registered DuckDB views in {db}")
    finally:
        con.close()


#: tables big enough that a full snapshot is streamed CSV -> Parquet instead of read in memory
STREAM_TABLES = {"fundamentals", "stocks", "daily"}


def _read_normalized(snap: Path, table: str, root: Path) -> pl.DataFrame:
    return normalize_table(table, _cast_text_numbers(read_raw_snapshot(snap), table), root)


def _stream_full_snapshot(snap: Path, table: str, root: Path, dest: Path) -> None:
    """Bulk zip -> temp CSV -> lazy typed scan -> Parquet, without holding text in memory."""
    zips = sorted(snap.glob("*.zip"))
    if not zips:  # REST snapshot: many small CSVs, fine to read eagerly
        _read_normalized(snap, table, root).write_parquet(dest)
        return
    tmp = dest.parent / f"_tmp_{table}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        csvs = []
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".csv"):
                        csvs.append(Path(zf.extract(name, tmp)))
        lf = pl.concat(
            [pl.scan_csv(c, infer_schema=False) for c in csvs], how="diagonal_relaxed"
        )
        lf = normalize_table(table, _cast_text_numbers(lf, table), root)
        lf.sink_parquet(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def refresh_processed(
    settings: Settings, tables: list[str] | None = None, rebuild: bool = False
) -> dict[str, int]:
    """Bring data/processed up to date with data/raw.

    Per table: base = newest FULL snapshot, then every DELTA snapshot taken after it is
    upserted in order (``UPSERT_KEYS``). If the base is unchanged since the last refresh only
    the not-yet-applied deltas are merged into the existing Parquet (cheap daily path);
    ``rebuild`` or a new base re-reads the full snapshot.
    """
    tables = tables or TABLES
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(settings)
    if meta.get("mode") == "SYNTHETIC":
        meta = {}  # never layer real data on a synthetic build
    src = meta.get("source", {})
    bases: dict[str, str] = dict(src.get("snapshots", {}))
    applied: dict[str, list[str]] = {k: list(v) for k, v in src.get("applied_deltas", {}).items()}
    tiers: set[str] = set(src.get("license_tiers", []))
    counts: dict[str, int] = dict(meta.get("row_counts", {}))
    for t in tables:
        snap = latest_snapshot(settings.raw_dir, t)
        if snap is None:
            logger.warning(f"no raw snapshot for {t}; skipping")
            continue
        tiers.add(read_manifest(snap).get("license_tier", "unknown"))
        path = out / f"{t}.parquet"
        deltas = delta_snapshots(settings.raw_dir, t, snap.name) if t in UPSERT_KEYS else []
        if rebuild or bases.get(t) != snap.name or not path.exists():
            if t in STREAM_TABLES:
                _stream_full_snapshot(snap, t, settings.root, path)
                if not deltas:
                    n = pl.scan_parquet(path).select(pl.len()).collect().item()
                    bases[t], applied[t], counts[t] = snap.name, [], n
                    logger.info(f"processed {t} @ {snap.name}: {n:,} rows -> {path} (streamed)")
                    continue
                df = pl.read_parquet(path)
            else:
                df = _read_normalized(snap, t, settings.root)
            logger.info(f"raw {t} @ {snap.name}: {df.height:,} rows")
            done: list[str] = []
        else:
            done = applied.get(t, [])
            deltas = [d for d in deltas if d.name not in done]
            if not deltas:
                continue
            df = pl.read_parquet(path)
        for d in deltas:
            dd = _read_normalized(d, t, settings.root)
            before = df.height
            df = upsert(df, dd, UPSERT_KEYS[t])
            done.append(d.name)
            logger.info(
                f"delta {t} @ {d.name}: {dd.height:,} rows upserted ({df.height - before:+,} net)"
            )
        df.write_parquet(path)
        bases[t], applied[t], counts[t] = snap.name, done, df.height
        logger.info(f"processed {t}: {df.height:,} rows -> {path}")
        del df
    if "fundamentals" not in bases:
        raise RuntimeError("no fundamentals snapshot found - run `pitlab data download` first")
    n_tk = pl.scan_parquet(out / "fundamentals.parquet").select(pl.col("ticker").n_unique())
    n_tk_val = int(n_tk.collect().item())
    mode = "SHARADAR-FREE" if ("free" in tiers or n_tk_val <= 60) else "SHARADAR-FULL"
    new_meta = {
        "mode": mode,
        "created_at": meta.get("created_at", utc_stamp()),
        "updated_at": utc_stamp(),
        "row_counts": counts,
        "n_fundamentals_tickers": n_tk_val,
        "source": {
            "snapshots": bases,
            "applied_deltas": applied,
            "license_tiers": sorted(tiers),
        },
    }
    (out / "_meta.json").write_text(json.dumps(new_meta, indent=2, default=str), encoding="utf-8")
    register_duckdb(settings)
    return counts


def ingest_raw(settings: Settings, tables: list[str] | None = None) -> dict[str, int]:
    """Rebuild processed tables from the newest full snapshot (+ later deltas) of each table."""
    return refresh_processed(settings, tables, rebuild=True)


def read_meta(settings: Settings) -> dict[str, Any]:
    p = settings.processed_dir / "_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_table(settings: Settings, name: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read a processed table; ``columns`` keeps only those present (saves RAM on prices)."""
    p = settings.processed_dir / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing - run `pitlab data synthetic` or `pitlab data download`"
        )
    if columns is None:
        return pl.read_parquet(p)
    have = pl.read_parquet_schema(p)
    return pl.read_parquet(p, columns=[c for c in columns if c in have])


def write_synthetic(
    settings: Settings,
    n_stocks: int | None = None,
    years: int | None = None,
    seed: int | None = None,
) -> dict[str, int]:
    from pitlab.data.synthetic import SyntheticParams, generate

    cfg = settings.synthetic
    params = SyntheticParams(
        n_stocks=n_stocks or cfg.n_stocks,
        years=years or cfg.years,
        end_year=cfg.end_year,
        seed=cfg.seed if seed is None else seed,
    )
    tables = generate(params)
    stamp = utc_stamp()
    # record a manifest like a real download so the Data-health page has something to show
    snap = settings.raw_dir / "synthetic" / stamp
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "params.json").write_text(json.dumps(params.__dict__, indent=2), encoding="utf-8")
    write_manifest(
        snap,
        "synthetic",
        sum(v.height for k, v in tables.items() if not k.startswith("_")),
        "synthetic",
        {"row_counts": {k: v.height for k, v in tables.items()}},
    )
    write_processed(
        settings, tables, "SYNTHETIC", {"synthetic_params": params.__dict__, "snapshot": stamp}
    )
    return {k: v.height for k, v in tables.items()}
