"""Figures : croissance nette d'un dollar par modèle, et Sharpe CPCV par configuration (boîtes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memoire2 import metrics as mx  # noqa: E402
from memoire2 import portfolio as pf  # noqa: E402
from memoire2.runner import FEE, RESULTS  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000", "#999999"]
LABELS = {"canada": "Canada (50 titres TSX)", "usa": "États-Unis (50 titres S&P 500)"}


def _style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8.0, 4.5), "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO), "figure.constrained_layout.use": True,
    })


def figures_for_country(country: str, results_dir: Path = RESULTS) -> list[Path]:
    _style()
    made = []
    pred_file = results_dir / f"predictions_{country}.parquet"
    if not pred_file.exists():
        return made
    joined = pd.read_parquet(pred_file)
    realized = joined["target"].unstack("ticker")
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for name in [c for c in joined.columns if c != "target"]:
        perf = pf.long_short_returns(joined[name].unstack("ticker"), realized, fee=FEE)
        ax.plot((1 + perf["net"]).cumprod(), lw=1.2,
                label=f"{name} (Sharpe {mx.sharpe_monthly(perf['net']):.2f})")
    ew = realized.mean(axis=1)
    ax.plot((1 + ew).cumprod(), lw=1.6, color="#999999", ls="--",
            label=f"équipondéré long only (Sharpe {mx.sharpe_monthly(ew):.2f})")
    ax.set_title(f"{LABELS[country]}, portefeuilles long short nets de 10 pb, 2008-2024")
    ax.set_ylabel("Valeur d'un dollar investi")
    ax.legend(fontsize=7, ncol=2)
    out = out_dir / f"richesse_nette_{country}.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    made.append(out)

    cpcv_file = results_dir / "tables" / f"cpcv_sharpe_{country}.csv"
    if cpcv_file.exists():
        matrix = pd.read_csv(cpcv_file, index_col=0)
        fig, ax = plt.subplots()
        ax.boxplot([row.dropna().values for _, row in matrix.iterrows()], tick_labels=matrix.index, vert=False)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_title(f"{LABELS[country]} : Sharpe net sur les {matrix.shape[1]} chemins CPCV (purge et embargo)")
        ax.set_xlabel("Ratio de Sharpe net, par chemin hors échantillon")
        out = out_dir / f"cpcv_{country}.png"
        fig.savefig(out, dpi=160)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        made.append(out)
    return made
