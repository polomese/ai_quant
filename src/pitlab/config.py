"""Configuration: YAML experiment settings + column map + environment secrets."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default CWD) to the folder holding ``config/settings.yaml``."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "settings.yaml").exists():
            return p
    # fall back to the source checkout (src/pitlab/config.py -> repo root)
    return Path(__file__).resolve().parents[2]


class EnvSettings(BaseSettings):
    """Secrets from the environment / ``.env`` (never from YAML)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    sharadar_api_key: str | None = Field(default=None, alias="SHARADAR_API_KEY")


PROJECT_NAME = "pit-two-clocks"
DATA_ROOT_ENV = "PITLAB_DATA_ROOT"


def resolve_data_root(root: Path) -> Path:
    """Base folder for data/ and outputs/: ``$PITLAB_DATA_ROOT/pit-two-clocks`` if set, else the repo.

    Keeping data outside the checkout (e.g. ``P:/quant_data``) keeps it out of git and cloud sync;
    each PC sets the variable once (``scripts/bootstrap.ps1``).
    """
    env = os.environ.get(DATA_ROOT_ENV)
    return Path(env) / PROJECT_NAME if env else root


class Paths(BaseModel):
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    duckdb_file: str = "data/pit.duckdb"


class Sample(BaseModel):
    start: date = date(1998, 1, 1)
    end: date | None = None


class Universe(BaseModel):
    categories: list[str] = ["Domestic Common Stock"]
    min_price: float = 5.0
    min_history_months: int = 12
    sp500_only: bool = False
    max_fundamental_staleness_days: int = 456
    comparison_universe: Literal["intersection", "per_arm"] = "intersection"


class Returns(BaseModel):
    delisting_return_unknown: float = -0.30
    delisting_return_mna: float = 0.0
    mna_actions: list[str] = ["acquisitionby", "mergerfrom"]


class Arm(BaseModel):
    label: str
    version: Literal["MR", "AR"]
    timing: Literal["period_end", "original_filing", "own_filing"]
    lag_trading_days: int = 0


class LagSweep(BaseModel):
    arms: list[str] = ["A", "C"]
    extra_lags: list[int] = [0, 1, 5, 20, 60]


class Preprocess(BaseModel):
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99


class Backtest(BaseModel):
    newey_west_lags: int = 6
    n_quantiles: int = 10
    n_quantiles_small: int = 5
    small_universe_threshold: int = 300
    min_names_per_date: int = 20
    fama_macbeth_sector_dummies: bool = True


class Slices(BaseModel):
    size_terciles: bool = True
    sectors: bool = True
    restaters: bool = True
    eras: dict[str, list[date | None]] = {}


class Update(BaseModel):
    """Bulk + daily incremental refresh of Sharadar data (``pitlab data update``)."""

    history: Literal["5", "10", "full"] = "full"  # must match the purchased plan tier
    small_tables: list[str] = ["tickers", "descriptions", "actions", "events", "sp500"]
    delta_tables: list[str] = ["fundamentals", "stocks", "daily"]
    overlap_days: int = 3  # re-request this many days before the stored max(lastupdated)
    rewrite_lookback_days: int = 10  # delta rows older than this => history was rewritten
    adjustment_actions: list[str] = ["split", "dividend", "spinoff", "spinoffdividend"]
    keep_full_snapshots: dict[str, int] = {"stocks": 2, "daily": 2}  # others: keep all


class Synthetic(BaseModel):
    n_stocks: int = 800
    years: int = 25
    end_year: int = 2024
    seed: int = 42


class Settings(BaseModel):
    paths: Paths = Paths()
    sample: Sample = Sample()
    universe: Universe = Universe()
    returns: Returns = Returns()
    arms: dict[str, Arm]
    lag_sweep: LagSweep = LagSweep()
    signals: dict[str, int]
    preprocess: Preprocess = Preprocess()
    backtest: Backtest = Backtest()
    slices: Slices = Slices()
    synthetic: Synthetic = Synthetic()
    update: Update = Update()

    # resolved at load time, not part of the YAML
    root: Path = Field(default_factory=find_project_root, exclude=True)
    data_root: Path | None = Field(default=None, exclude=True)

    @property
    def base_dir(self) -> Path:
        return self.data_root or self.root

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.paths.data_dir

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / self.paths.outputs_dir

    @property
    def duckdb_path(self) -> Path:
        return self.base_dir / self.paths.duckdb_file

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def load_settings(
    path: str | Path | None = None, root: Path | None = None, data_root: Path | None = None
) -> Settings:
    root = root or find_project_root(Path(path).parent.parent if path else None)
    path = Path(path) if path else root / "config" / "settings.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(**raw, root=root, data_root=data_root or resolve_data_root(root))


@lru_cache(maxsize=4)
def _load_columns(path: str) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_columns(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Logical -> physical column map per table (``config/columns.yaml``)."""
    root = root or find_project_root()
    return _load_columns(str(root / "config" / "columns.yaml"))
