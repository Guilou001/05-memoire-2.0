"""Construction du panel (date de formation, titre) : caractéristiques, macro disponible, cible.

Le panel est la table sur laquelle tout le reste travaille. Une ligne = un titre à une date de formation t
(un 1er de mois). La cible est le rendement du mois qui COMMENCE en t. Chaque caractéristique n'utilise que
des prix antérieurs ou égaux à t ; la macro attachée à t est la dernière ligne PUBLIÉE avant t
(paramètre ``macro_lag``, par défaut 2 : au 1er juin, la dernière ligne publiée est celle d'avril,
puisque la ligne de mai paraît dans les premiers jours de juin). C'est la correction du biais d'alignement
mesuré dans la version 1 du mémoire.

Les caractéristiques sont normalisées en rangs transversaux dans [-0,5 ; 0,5] à chaque date (convention
Gu, Kelly et Xiu, 2020) : aucune information d'une autre date n'entre dans la normalisation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from memoire2.data import month_start_prices, monthly_returns, read_macro, read_prices_daily

CHARACTERISTICS = ("mom_1", "mom_3", "mom_6", "mom_12_2", "vol_12m", "vol_1m", "maxret_1m")


def characteristics(prices_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sept caractéristiques par titre, tamponnées à la date de formation t (info <= t seulement)."""
    pm = month_start_prices(prices_daily)
    r = monthly_returns(pm)                      # rendement tamponné t = mois [t-1, t]
    daily_ret = prices_daily.pct_change(fill_method=None)
    month_of = daily_ret.index.to_period("M")
    vol_1m = daily_ret.groupby(month_of).std()
    maxret = daily_ret.groupby(month_of).max()
    # tamponner fin de mois -> 1er du mois suivant (disponible à la formation t)
    vol_1m.index = (vol_1m.index + 1).to_timestamp()
    maxret.index = (maxret.index + 1).to_timestamp()
    out = {
        "mom_1": r,
        "mom_3": pm.pct_change(3, fill_method=None),
        "mom_6": pm.pct_change(6, fill_method=None),
        "mom_12_2": pm.shift(1).pct_change(11, fill_method=None),   # 12 mois en sautant le dernier
        "vol_12m": r.rolling(12).std(),
        "vol_1m": vol_1m,
        "maxret_1m": maxret,
    }
    return {k: v.reindex(pm.index) for k, v in out.items()}


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rangs transversaux dans [-0,5 ; 0,5] ligne par ligne (chaque date se normalise seule)."""
    n = df.notna().sum(axis=1)
    return df.rank(axis=1).sub(1).div(n - 1, axis=0).sub(0.5).where(df.notna())


def build_panel(country: str, raw_dir: Path, macro_lag: int = 2) -> pd.DataFrame:
    """Panel MultiIndex (date de formation, titre) : caractéristiques normalisées, macro brute, cible."""
    prices = read_prices_daily(country, raw_dir)
    pm = month_start_prices(prices)
    r = monthly_returns(pm)
    target = r.shift(-1)                          # cible à t = rendement du mois [t, t+1]
    chars = {k: rank_normalize(v) for k, v in characteristics(prices).items()}

    macro = read_macro(country, raw_dir)
    # ligne macro attachée à t : mois t - macro_lag (la dernière publiée avant t)
    macro_avail = macro.copy()
    macro_avail.index = macro_avail.index + pd.DateOffset(months=macro_lag)
    macro_avail = macro_avail.reindex(pm.index)

    frames = []
    for k, v in chars.items():
        s = v.stack()
        s.name = k
        frames.append(s)
    t = target.stack()
    t.name = "target"
    frames.append(t)
    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.dropna(subset=[*CHARACTERISTICS, "target"])

    macro_cols = macro_avail.add_prefix("m_")
    panel = panel.join(macro_cols, on="date")
    keep = [c for c in panel.columns if not c.startswith("m_") or panel[c].notna().mean() > 0.95]
    panel = panel[keep].dropna()
    return panel.sort_index()


def macro_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("m_")]
