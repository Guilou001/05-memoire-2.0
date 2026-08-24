"""Lecture des données brutes : prix quotidiens Yahoo, macro LCDMA (Canada) et FRED-MD (États-Unis).

Conventions de tampon, fixées une fois pour toutes et testées dans ``tests/test_panel.py`` :

- prix mensuel au 1er du mois M = dernier prix quotidien observé avant ou au 1er ;
- rendement tamponné M = variation du prix entre le 1er du mois M-1 et le 1er du mois M
  (il couvre donc le mois M-1, entièrement passé au 1er du mois M) ;
- ligne macro tamponnée M = valeurs du mois M, publiées au début du mois M+1 (convention FRED-MD,
  vérifiée sur UNRATE d'avril 2020) ; le décalage de disponibilité est géré dans ``panel.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "canada": "canadian_stocks_2000-01-01_to_2024-06-01.csv",
    "usa": "us_stocks_2000-01-01_to_2024-06-01.csv",
}
MACRO_FILES = {"canada": ("macro_data.csv", ","), "usa": ("Fred-MD.csv", ";")}
BENCHMARKS = {"canada": "TSX60_2000-01-01_to_2024-06-01.csv", "usa": "SP500_2000-01-01_to_2024-06-01.csv"}


def read_prices_daily(country: str, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / FILES[country], index_col=0, parse_dates=True).sort_index()
    return df.ffill()


def read_benchmark_daily(country: str, raw_dir: Path) -> pd.Series:
    df = pd.read_csv(raw_dir / BENCHMARKS[country], index_col=0, parse_dates=True).sort_index()
    return df.iloc[:, 0].ffill()


def read_macro(country: str, raw_dir: Path) -> pd.DataFrame:
    """Panel macro mensuel, index au 1er du mois couvert, colonnes numériques seulement."""
    fname, sep = MACRO_FILES[country]
    df = pd.read_csv(raw_dir / fname, sep=sep)
    date_col = "sasdate" if "sasdate" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.index = df.index.to_period("M").to_timestamp()  # 1er du mois couvert
    return df


def month_start_prices(prices_daily: pd.DataFrame) -> pd.DataFrame:
    """Prix au 1er de chaque mois : dernier prix observé avant ou au 1er (aucune information future)."""
    grid = pd.date_range(prices_daily.index.min().to_period("M").to_timestamp(),
                         prices_daily.index.max(), freq="MS")
    return prices_daily.reindex(prices_daily.index.union(grid)).ffill().loc[grid]


def monthly_returns(prices_monthly: pd.DataFrame) -> pd.DataFrame:
    """Rendement tamponné M = ce que le titre a fait pendant le mois M-1 (voir docstring du module)."""
    return prices_monthly.pct_change(fill_method=None).iloc[1:]
