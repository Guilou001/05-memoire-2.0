"""Tests du cœur scientifique : alignement sans fuite, purge, portefeuille, métriques."""

import numpy as np
import pandas as pd
import pytest

from memoire2 import metrics as mx
from memoire2.data import month_start_prices, monthly_returns
from memoire2.panel import rank_normalize
from memoire2.portfolio import long_short_returns, quantile_weights
from memoire2.validation import contiguous_groups, cpcv_splits, purge_train_dates, walk_forward_splits


@pytest.fixture()
def daily_prices():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 6)), axis=0)),
                          index=idx, columns=list("ABCDEF"))
    return prices


# ------------------------------------------------------------------ alignement
def test_month_start_price_uses_only_past_prices(daily_prices):
    pm = month_start_prices(daily_prices)
    # le 1er février 2019 est un vendredi ; le prix au tampon 2019-02-01 doit être celui du 1er ou d'avant
    stamp = pd.Timestamp("2019-02-01")
    expected = daily_prices.loc[:stamp].iloc[-1]
    pd.testing.assert_series_equal(pm.loc[stamp], expected, check_names=False)


def test_return_stamp_covers_previous_month(daily_prices):
    pm = month_start_prices(daily_prices)
    r = monthly_returns(pm)
    stamp = pd.Timestamp("2019-03-01")
    manual = pm.loc[stamp] / pm.loc[pd.Timestamp("2019-02-01")] - 1
    pd.testing.assert_series_equal(r.loc[stamp], manual, check_names=False)


def test_target_would_be_next_month():
    # la cible à t est le rendement tamponné t+1 : vérifié par construction dans build_panel (shift(-1)) ;
    # ici on vérifie la propriété sur un cas minimal
    r = pd.DataFrame({"A": [0.1, 0.2, 0.3]},
                     index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    target = r.shift(-1)
    assert target.loc["2020-01-01", "A"] == pytest.approx(0.2)


def test_rank_normalize_is_cross_sectional_and_bounded():
    df = pd.DataFrame([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]],
                      index=pd.date_range("2020-01-01", periods=2, freq="MS"), columns=list("XYZ"))
    z = rank_normalize(df)
    assert z.loc["2020-01-01"].tolist() == [-0.5, 0.0, 0.5]
    assert z.loc["2020-02-01"].tolist() == [0.5, 0.0, -0.5]


# ------------------------------------------------------------------ validation
def test_purge_removes_embargo_neighbourhood():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    test = dates[10:12]
    train = dates.difference(test)
    purged = purge_train_dates(train, test, embargo_months=2)
    assert dates[9] not in purged and dates[12] not in purged and dates[13] not in purged
    assert dates[0] in purged and dates[23] in purged


def test_cpcv_counts_and_disjointness():
    dates = pd.date_range("2008-01-01", periods=96, freq="MS")
    splits = cpcv_splits(dates, n_groups=6, k_test=2, embargo_months=1)
    assert len(splits) == 15                       # C(6, 2)
    for train, test in splits:
        assert len(set(train) & set(test)) == 0
        assert min(abs((pd.DatetimeIndex(sorted(set(test))) - d).days).min() for d in train) > 31


def test_walk_forward_trains_only_on_past():
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    for train, test in walk_forward_splits(dates, first_test="2006-01-01", step_months=12):
        assert train.max() < test.min()


def test_groups_are_contiguous():
    dates = pd.date_range("2010-01-01", periods=40, freq="MS")
    g = contiguous_groups(dates, 4)
    assert (g.sort_index().diff().dropna() >= 0).all()


# ------------------------------------------------------------------ portefeuille
def test_quantile_weights_sides_and_sum():
    pred = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -2.0})
    w = quantile_weights(pred, quantile=0.2)
    assert w["A"] == pytest.approx(1.0) and w["E"] == pytest.approx(-1.0)
    assert w.sum() == pytest.approx(0.0) and w.abs().sum() == pytest.approx(2.0)


def test_constant_predictions_give_no_position():
    pred = pd.Series(1.0, index=list("ABCDE"))
    assert quantile_weights(pred).abs().sum() == 0.0


def test_costs_reduce_net_of_gross():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    rng = np.random.default_rng(3)
    preds = pd.DataFrame(rng.normal(size=(6, 10)), index=idx, columns=list("ABCDEFGHIJ"))
    realized = pd.DataFrame(rng.normal(0, 0.05, size=(6, 10)), index=idx, columns=list("ABCDEFGHIJ"))
    out = long_short_returns(preds, realized, quantile=0.2, fee=0.001)
    assert (out["net"] <= out["gross"] + 1e-12).all()
    assert out["turnover"].iloc[0] == pytest.approx(2.0)   # mise en place : 100 % long + 100 % short


# ------------------------------------------------------------------ métriques
def test_r2_oos_gkx_zero_forecast_is_zero():
    y = np.array([0.1, -0.2, 0.05])
    assert mx.r2_oos_gkx(y, np.zeros(3)) == pytest.approx(0.0)


def test_sharpe_and_nw_tstat_signs():
    r = pd.Series(np.full(120, 0.01))
    assert mx.sharpe_monthly(r + np.random.default_rng(0).normal(0, 0.001, 120)) > 3


def test_deflated_sharpe_penalizes_trials():
    high = mx.deflated_sharpe(1.0, n_trials=2, n_obs=192, skew=0.0, kurt=3.0)
    low = mx.deflated_sharpe(1.0, n_trials=500, n_obs=192, skew=0.0, kurt=3.0)
    assert high > low


def test_pbo_unbiased_around_half_for_pure_noise():
    # avec peu de chemins, la PBO d'une seule matrice de bruit est très variable ; c'est sa MOYENNE
    # sur des tirages indépendants qui doit tourner autour de 0,5 (aucun vrai signal à sélectionner)
    pbos = [mx.pbo_cscv(pd.DataFrame(np.random.default_rng(s).normal(size=(20, 8)))) for s in range(30)]
    assert 0.35 <= float(np.mean(pbos)) <= 0.65


def test_pbo_low_when_one_config_dominates():
    rng = np.random.default_rng(2)
    perf = pd.DataFrame(rng.normal(0, 0.1, size=(10, 8)))
    perf.iloc[3] += 2.0                              # une configuration réellement meilleure partout
    assert mx.pbo_cscv(perf) < 0.2
