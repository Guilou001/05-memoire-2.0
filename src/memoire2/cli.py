"""Ligne de commande : ``m2 run`` (protocole complet), ``m2 figures`` (graphiques)."""

from __future__ import annotations

import typer

app = typer.Typer(help="Mémoire 2.0 : protocole complet sans fuite, coûts inclus, PBO et Sharpe déflaté.")


@app.command()
def run(country: str = typer.Option("both", help="canada, usa ou both"),
        fast: bool = typer.Option(False, help="ridge et hist_gb seulement, sans CPCV (essai rapide)")) -> None:
    from memoire2.runner import run_country

    for c in (["canada", "usa"] if country == "both" else [country]):
        run_country(c, fast=fast)


@app.command()
def figures(country: str = typer.Option("both")) -> None:
    from memoire2.report import figures_for_country

    for c in (["canada", "usa"] if country == "both" else [country]):
        for p in figures_for_country(c):
            typer.echo(str(p))


if __name__ == "__main__":
    app()
