"""`pitlab` command line interface."""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from pitlab.config import EnvSettings, load_settings

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Point-in-time 'two clocks' bias lab."
)
data_app = typer.Typer(no_args_is_help=True, help="Download, generate and validate data.")
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to settings.yaml")]


def _setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )


@data_app.command("synthetic")
def data_synthetic(
    config: ConfigOpt = None,
    n_stocks: Annotated[int | None, typer.Option(help="Number of synthetic stocks")] = None,
    years: Annotated[int | None, typer.Option(help="Years of history")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed")] = None,
) -> None:
    """Build a fake Sharadar-shaped dataset with injected look-ahead (no API key needed)."""
    _setup()
    from pitlab.data.ingest import write_synthetic

    s = load_settings(config)
    counts = write_synthetic(s, n_stocks=n_stocks, years=years, seed=seed)
    typer.echo(f"synthetic data written to {s.processed_dir}: {counts}")


@data_app.command("download")
def data_download(
    config: ConfigOpt = None,
    tables: Annotated[str, typer.Option(help="'all' or comma-separated table names")] = "all",
    method: Annotated[str, typer.Option(help="auto | bulk | rest")] = "auto",
    history: Annotated[str, typer.Option(help="bulk history: 5 | 10 | full")] = "full",
    tickers: Annotated[
        str | None, typer.Option(help="REST only: restrict to these tickers (comma list)")
    ] = None,
    resume: Annotated[
        str | None, typer.Option(help="REST only: resume this snapshot timestamp")
    ] = None,
    ingest: Annotated[bool, typer.Option(help="Normalize to Parquet + DuckDB afterwards")] = True,
) -> None:
    """Download Sharadar tables (bulk preferred, paginated REST fallback) into data/raw/."""
    _setup()
    from pitlab.data.ingest import TABLES, ingest_raw
    from pitlab.data.sharadar_client import SharadarClient, download_tables

    s = load_settings(config)
    key = EnvSettings().sharadar_api_key
    if not key:
        typer.echo(
            "SHARADAR_API_KEY is not set (copy .env.example to .env). Use `pitlab data synthetic` without a key."
        )
        raise typer.Exit(2)
    names = TABLES if tables == "all" else [t.strip() for t in tables.split(",") if t.strip()]
    client = SharadarClient(key)
    tk = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    download_tables(
        client,
        s.raw_dir,
        names,
        method=method,
        history=history,
        tickers=tk,
        resume=resume,
        categories=s.universe.categories,
    )
    if ingest:
        ingest_raw(s, names)


@data_app.command("update")
def data_update(
    config: ConfigOpt = None,
    full: Annotated[
        bool, typer.Option("--full", help="Fresh bulk download of every table (weekly job)")
    ] = False,
    validate: Annotated[bool, typer.Option(help="Run `data validate` afterwards")] = True,
) -> None:
    """Top off the local Sharadar copy: daily incremental by default, `--full` for a bulk refresh.

    Writes a status line to <data>/logs/last_update.json (shown in scheduled-task logs).
    """
    _setup()
    import json
    from datetime import UTC, datetime

    from pitlab.data.sharadar_client import SharadarClient
    from pitlab.data.update import update

    s = load_settings(config)
    key = EnvSettings().sharadar_api_key
    if not key:
        typer.echo("SHARADAR_API_KEY is not set (.env)")
        raise typer.Exit(2)
    status: dict = {"started": datetime.now(UTC).isoformat(timespec="seconds"), "ok": False}
    logs = s.data_dir.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        status["summary"] = update(s, SharadarClient(key), full=full)
        if validate:
            from pitlab.data.validate import validate as run_validate

            rep = run_validate(s)
            status["validation"] = f"passed ({len(rep['checks'])} checks)"
        status["ok"] = True
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        logger.exception("update failed")
        raise typer.Exit(1) from e
    finally:
        status["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        (logs / "last_update.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    typer.echo(json.dumps(status["summary"], indent=2))


@data_app.command("ingest")
def data_ingest(config: ConfigOpt = None) -> None:
    """Re-build data/processed + DuckDB from the latest raw snapshots."""
    _setup()
    from pitlab.data.ingest import ingest_raw

    ingest_raw(load_settings(config))


@data_app.command("validate")
def data_validate(config: ConfigOpt = None) -> None:
    """Validate processed data; exits non-zero on any failed error-level check."""
    _setup()
    from pitlab.data.validate import ValidationError, validate

    try:
        rep = validate(load_settings(config))
    except ValidationError as e:
        logger.error(str(e))
        raise typer.Exit(1) from e
    typer.echo(f"validation passed ({len(rep['checks'])} checks)")


@app.command("run")
def run(
    config: ConfigOpt = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Output folder name under outputs/")
    ] = None,
    lag_sweep: Annotated[
        bool,
        typer.Option(
            "--lag-sweep", help="Run ONLY the lag sweep (adds lag_sweep.parquet to the run)"
        ),
    ] = False,
    no_lag_sweep: Annotated[
        bool, typer.Option("--no-lag-sweep", help="Skip the lag sweep in a full run")
    ] = False,
) -> None:
    """Run the 3-arm experiment (and by default the lag sweep) and write outputs/<run_id>/."""
    _setup()
    from pitlab.pipeline import load_inputs, run_experiment
    from pitlab.report import write_results

    s = load_settings(config)
    rid = run_id or ("lag_sweep" if lag_sweep else "demo")
    inp = load_inputs(s)
    res = run_experiment(inp, lag_sweep=lag_sweep or not no_lag_sweep, main=not lag_sweep)
    out = write_results(s, rid, res)
    typer.echo((out / "summary.md").read_text(encoding="utf-8"))


@app.command("report")
def report(
    config: ConfigOpt = None, run_id: Annotated[str, typer.Option("--run-id")] = "demo"
) -> None:
    """Write a static summary.md + report.html for an existing run."""
    _setup()
    from pitlab.report import render_html, render_markdown, run_dir

    out = run_dir(load_settings(config), run_id)
    if not (out / "summary.json").exists():
        typer.echo(f"no run at {out}; run `pitlab run --run-id {run_id}` first")
        raise typer.Exit(1)
    (out / "summary.md").write_text(render_markdown(out), encoding="utf-8")
    html = render_html(out)
    typer.echo(f"wrote {out / 'summary.md'} and {html}")


@app.command("site")
def site(
    config: ConfigOpt = None,
    run_id: Annotated[str, typer.Option("--run-id")] = "demo",
    out: Annotated[
        Path | None, typer.Option(help="Output file (default: <run>/site/index.html)")
    ] = None,
) -> None:
    """Write a single offline HTML version of the dashboard (aggregate results only)."""
    _setup()
    from pitlab.site import build_site

    path = build_site(load_settings(config), run_id, out)
    typer.echo(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) - open it in any browser")


@app.command("dashboard")
def dashboard(port: Annotated[int, typer.Option(help="Port")] = 8501) -> None:
    """Launch the Streamlit dashboard."""
    from pitlab.config import find_project_root

    app_path = find_project_root() / "dashboard" / "app.py"
    raise typer.Exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
        )
    )


if __name__ == "__main__":
    app()
