"""Integration test: on synthetic data with planted look-ahead, the method must detect it.

Builds a small synthetic universe in a temp project, runs the full pipeline (main arms + lag
sweep), and asserts A > B > C for the leaked signals and zero revision effect for controls.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from pitlab.config import find_project_root, load_settings
from pitlab.data.ingest import write_synthetic
from pitlab.data.validate import validate
from pitlab.pipeline import load_inputs, run_experiment
from pitlab.report import render_html, write_results

LEAKED = ["earnings_yield", "roe", "sales_growth", "gross_profitability"]
CONTROLS = ["book_to_price", "asset_growth", "accruals"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(find_project_root() / "config", root / "config")
    s = load_settings(root / "config" / "settings.yaml", root=root)
    s.sample.start = s.sample.start.replace(year=2000)
    write_synthetic(s, n_stocks=300, years=12, seed=7)
    validate(s)
    res = run_experiment(load_inputs(s), lag_sweep=True)
    out = write_results(s, "e2e", res)
    return s, res, out


def _ic(res, arm, signal):
    t = res.tables["ic_summary"].filter(
        (pl.col("slice_type") == "all") & (pl.col("arm") == arm) & (pl.col("signal") == signal)
    )
    return t["mean_ic"][0]


@pytest.mark.parametrize("signal", LEAKED)
def test_a_gt_b_gt_c_for_leaked_signals(run, signal):
    _, res, _ = run
    a, b, c = (_ic(res, x, signal) for x in "ABC")
    assert a > b > c, (signal, a, b, c)


@pytest.mark.parametrize("signal", LEAKED)
def test_total_bias_significant(run, signal):
    _, res, _ = run
    e = (
        res.tables["effects"]
        .filter(
            (pl.col("slice_type") == "all")
            & (pl.col("signal") == signal)
            & (pl.col("effect") == "total")
        )
        .row(0, named=True)
    )
    assert e["mean_diff"] > 0 and e["t_nw"] > 3


@pytest.mark.parametrize("signal", CONTROLS)
def test_no_revision_effect_for_unrestated_controls(run, signal):
    _, res, _ = run
    b, c = _ic(res, "B", signal), _ic(res, "C", signal)
    assert b == pytest.approx(c, abs=1e-12)


def test_revision_effect_concentrated_in_restaters(run):
    _, res, _ = run
    e = res.tables["effects"].filter(
        (pl.col("slice_type") == "restater")
        & (pl.col("signal") == "roe")
        & (pl.col("effect") == "revision")
    )
    got = dict(zip(e["slice"], e["mean_diff"], strict=True))
    assert got["restater"] > 0.01
    # non-restaters have MR == AR and identical timing; only winsorization bounds (set on the
    # full cross-section, which includes restaters) can move a clipped tie
    assert abs(got["non-restater"]) < 1e-3


def test_contamination_metrics(run):
    _, res, _ = run
    c = res.tables["contamination"]
    roe_ca = c.filter((pl.col("signal") == "roe") & (pl.col("pair") == "C_vs_A"))
    bp_cb = c.filter((pl.col("signal") == "book_to_price") & (pl.col("pair") == "C_vs_B"))
    assert roe_ca["rank_corr"].mean() < 0.999 and roe_ca["share_decile_change"].mean() > 0.01
    assert (
        bp_cb["rank_corr"].min() == pytest.approx(1.0) and bp_cb["share_decile_change"].max() == 0.0
    )


def test_lag_sweep_decays_for_arm_a(run):
    _, res, _ = run
    ls = (
        res.tables["lag_sweep"]
        .filter((pl.col("signal") == "roe") & (pl.col("base_arm") == "A"))
        .sort("extra_lag")
    )
    ic = dict(zip(ls["extra_lag"], ls["mean_ic"], strict=True))
    assert ic[0] > ic[60]


def test_outputs_written(run):
    _, _, out = run
    for f in [
        "config_used.yaml",
        "ic_monthly.parquet",
        "portfolio_returns.parquet",
        "fm_results.parquet",
        "contamination.parquet",
        "summary.json",
        "summary.md",
        "lag_sweep.parquet",
    ]:
        assert (out / f).exists(), f
    assert render_html(out).exists()
    assert "SYNTHETIC" in (out / "summary.md").read_text(encoding="utf-8")


def test_no_future_information_in_fama_macbeth_inputs(run):
    """Fama-MacBeth slope for arm C on leaked signals should be small vs arm A."""
    _, res, _ = run
    fm = res.tables["fm_results"].filter(pl.col("signal") == "roe")
    coef = dict(zip(fm["arm"], fm["coef_mean"], strict=True))
    assert coef["A"] > coef["C"]


def test_repo_gitignore_blocks_data():
    gi = (Path(find_project_root()) / ".gitignore").read_text(encoding="utf-8")
    for pat in ["data/", "*.parquet", "*.duckdb", ".env", "outputs/cache/"]:
        assert pat in gi


def test_experiment_writeup_and_offline_site(run):
    """The write-up grades every hypothesis, and the offline site is self-contained and
    carries no ticker (aggregate results only)."""
    import json

    from pitlab.narrative import experiment_md, hypotheses
    from pitlab.site import build_site, load_tables

    s, _, out = run
    T, meta = load_tables(out)
    hyp = hypotheses(T)
    assert {h["id"] for h in hyp} >= {"H1", "H2", "H4"}
    assert all(h["verdict"] for h in hyp)
    assert "## Hypotheses and results" in experiment_md(T, meta)
    # price signals are identical across arms, so the control hypothesis must pass
    assert next(h for h in hyp if h["id"] == "H4")["verdict"].startswith("✅")

    path = build_site(s, "e2e")
    html = path.read_text(encoding="utf-8")
    assert "Plotly" in html and "__PAYLOAD__" not in html and "/*__PLOTLY__*/" not in html
    assert "<script src=" not in html  # nothing fetched from the network
    payload = html[
        html.index('type="application/json">') + 24 : html.index(
            "</script>", html.index('type="application/json">')
        )
    ]
    data = json.loads(payload)  # "<\/" in the page is a valid JSON escape for "</"
    tickers = set(pl.read_parquet(s.processed_dir / "tickers.parquet")["ticker"])
    blob = json.dumps(data)
    assert not [t for t in tickers if f'"{t}"' in blob]
    assert set(data["per_signal"]) == set(data["signals"])
