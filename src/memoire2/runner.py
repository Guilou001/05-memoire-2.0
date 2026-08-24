"""Le protocole complet, dans l'ordre, sans fuite :

1. Panel (date, titre) avec alignement temps réel (``panel.py``).
2. Choix des hyperparamètres UNE FOIS, sur la fenêtre de validation 2004-2007 (entraînement 2000-2003),
   puis gel : la période de test 2008-2024 ne sert jamais à choisir un réglage (la fuite de sélection de la
   v1 disparaît).
3. Évaluation walk-forward : réentraînement annuel sur fenêtre croissante, prédictions mensuelles,
   portefeuilles nets de coûts (comparable au protocole du mémoire, en honnête).
4. Évaluation CPCV : les mêmes configurations gelées jugées sur les chemins purgés, d'où la matrice
   configurations x chemins qui donne la PBO ; le Sharpe déflaté corrige du nombre total d'essais
   (grille de validation + familles).
5. L'ensemble = moyenne des prédictions standardisées des familles retenues.

Sorties : ``results/predictions_<pays>.parquet``, ``results/tables/*.csv``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from memoire2 import metrics as mx
from memoire2 import portfolio as pf
from memoire2.models import grid, model_zoo
from memoire2.panel import build_panel
from memoire2.validation import cpcv_splits, walk_forward_splits

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
TRAIN_END, VALID_END = "2004-01-01", "2008-01-01"
FEE = 0.001
SEED = 123


def _dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index.get_level_values("date")


def _fit_predict(pipe, panel: pd.DataFrame, train_dates, test_dates) -> pd.Series:
    train = panel.loc[_dates(panel).isin(train_dates)]
    test = panel.loc[_dates(panel).isin(test_dates)]
    x_cols = [c for c in panel.columns if c != "target"]
    model = clone(pipe).fit(train[x_cols], train["target"].values)
    return pd.Series(model.predict(test[x_cols]), index=test.index)


def select_hyperparameters(panel: pd.DataFrame, verbose: bool = True) -> tuple[dict, int]:
    """Grilles évaluées sur 2004-2007 (entraînement 2000-2003) ; retourne les pipelines gelés et n_essais."""
    dates = _dates(panel)
    train_dates = pd.DatetimeIndex(sorted(set(dates[dates < TRAIN_END])))
    valid_dates = pd.DatetimeIndex(sorted(set(dates[(dates >= TRAIN_END) & (dates < VALID_END)])))
    zoo, grids = model_zoo(), grid()
    frozen, n_trials = {}, 0
    for name, pipe in zoo.items():
        if pipe is None:
            continue
        best_r2, best_params = -np.inf, {}
        for params in grids.get(name, [{}]):
            n_trials += 1
            candidate = clone(pipe).set_params(**params)
            pred = _fit_predict(candidate, panel, train_dates, valid_dates)
            r2 = mx.r2_oos_gkx(panel.loc[pred.index, "target"].values, pred.values)
            if r2 > best_r2:
                best_r2, best_params = r2, params
        frozen[name] = clone(pipe).set_params(**best_params)
        if verbose:
            print(f"  {name}: {best_params} (R2 validation {best_r2:.4f})")
    return frozen, n_trials


def walk_forward_predictions(panel: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Prédictions hors échantillon 2008-2024, réentraînement annuel, une colonne par modèle + ensemble."""
    splits = walk_forward_splits(_dates(panel), first_test=VALID_END, step_months=12)
    preds = {}
    for name, pipe in frozen.items():
        t0 = time.perf_counter()
        parts = [_fit_predict(pipe, panel, tr, te) for tr, te in splits]
        preds[name] = pd.concat(parts)
        print(f"  {name}: walk-forward en {time.perf_counter() - t0:.0f} s")
    out = pd.DataFrame(preds)
    # ensemble : moyenne des prédictions standardisées par date (chaque modèle vote à poids égal)
    z = out.groupby(level="date").transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    out["ensemble"] = z.mean(axis=1)
    return out


def cpcv_performance(panel: pd.DataFrame, frozen: dict, n_groups: int = 8, k_test: int = 2,
                     embargo_months: int = 1) -> pd.DataFrame:
    """Sharpe net par (configuration, chemin CPCV) sur 2008-2024 : la matrice de la PBO."""
    dates = _dates(panel)
    eval_panel = panel.loc[dates >= VALID_END]
    splits = cpcv_splits(pd.DatetimeIndex(sorted(set(_dates(eval_panel)))), n_groups, k_test, embargo_months)
    rows = {}
    for name, pipe in frozen.items():
        sharpes = []
        for tr, te in splits:
            pred = _fit_predict(pipe, eval_panel, tr, te)
            wide_p = pred.unstack("ticker")
            wide_r = eval_panel.loc[pred.index, "target"].unstack("ticker")
            perf = pf.long_short_returns(wide_p, wide_r, fee=FEE)
            sharpes.append(mx.sharpe_monthly(perf["net"]))
        rows[name] = sharpes
        print(f"  {name}: CPCV {len(splits)} chemins, Sharpe médian {np.nanmedian(sharpes):.2f}")
    return pd.DataFrame(rows).T


def run_country(country: str, raw_dir: Path = RAW, results_dir: Path = RESULTS, fast: bool = False) -> pd.DataFrame:
    print(f"=== {country} : panel ===")
    panel = build_panel(country, raw_dir)
    n_dates = panel.index.get_level_values("date").nunique()
    print(f"  {len(panel)} observations, {n_dates} dates, {panel.index.get_level_values('ticker').nunique()} titres")

    print(f"=== {country} : hyperparamètres sur validation 2004-2007 (gelés ensuite) ===")
    frozen, n_trials = select_hyperparameters(panel)
    if fast:
        frozen = {k: v for k, v in frozen.items() if k in ("ridge", "hist_gb")}

    print(f"=== {country} : walk-forward 2008-2024 ===")
    preds = walk_forward_predictions(panel, frozen)
    results_dir.mkdir(parents=True, exist_ok=True)
    joined = preds.join(panel["target"])
    joined.to_parquet(results_dir / f"predictions_{country}.parquet")

    realized = panel["target"].unstack("ticker")
    perf_rows = []
    for name in preds.columns:
        wide = preds[name].unstack("ticker")
        perf = pf.long_short_returns(wide, realized.loc[wide.index], fee=FEE)
        r2 = mx.r2_oos_gkx(joined["target"].values, joined[name].values) if name != "ensemble" else np.nan
        net = perf["net"]
        perf_rows.append({
            "model": name, "r2_oos": r2,
            "cagr_net": mx.cagr_monthly(net), "sharpe_net": mx.sharpe_monthly(net),
            "sharpe_brut": mx.sharpe_monthly(perf["gross"]), "max_drawdown": mx.max_drawdown(net),
            "turnover_mensuel": float(perf["turnover"].mean()), "t_newey_west": mx.newey_west_tstat(net),
            "dsr": mx.deflated_sharpe(mx.sharpe_monthly(net), n_trials=n_trials + len(preds.columns),
                                      n_obs=len(net), skew=float(net.skew()), kurt=float(net.kurt() + 3)),
        })
    # les deux repères sans apprentissage
    naive = pf.momentum_vol_benchmark(panel.loc[_dates(panel) >= VALID_END], fee=FEE)
    perf_rows.append({"model": "controle_momentum_vol (Nagel)", "cagr_net": mx.cagr_monthly(naive["net"]),
                      "sharpe_net": mx.sharpe_monthly(naive["net"]), "max_drawdown": mx.max_drawdown(naive["net"]),
                      "turnover_mensuel": float(naive["turnover"].mean()),
                      "t_newey_west": mx.newey_west_tstat(naive["net"])})
    ew = realized.loc[realized.index >= VALID_END].mean(axis=1)
    perf_rows.append({"model": "equipondere_long_only", "cagr_net": mx.cagr_monthly(ew),
                      "sharpe_net": mx.sharpe_monthly(ew), "max_drawdown": mx.max_drawdown(ew)})

    table = pd.DataFrame(perf_rows).set_index("model")
    (results_dir / "tables").mkdir(exist_ok=True)
    table.to_csv(results_dir / "tables" / f"walk_forward_{country}.csv")
    print(table.round(3).to_string())

    if not fast:
        print(f"=== {country} : CPCV et PBO ===")
        matrix = cpcv_performance(panel, frozen)
        matrix.to_csv(results_dir / "tables" / f"cpcv_sharpe_{country}.csv")
        pbo = mx.pbo_cscv(matrix)
        (results_dir / "tables" / f"pbo_{country}.json").write_text(
            json.dumps({"pbo": pbo, "n_configs": len(matrix), "n_chemins": matrix.shape[1]}, indent=1))
        print(f"  PBO ({len(matrix)} configurations x {matrix.shape[1]} chemins) : {pbo:.2f}")
    return table
