# Prérequis : uv (https://docs.astral.sh/uv/). Les données macro (LCDMA, FRED-MD) se déposent à la main : voir scripts/fetch_data.py.
UV ?= uv

setup:            ## environnement verrouillé
	$(UV) sync --locked --all-extras

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts

data:             ## télécharge les prix Yahoo (50 titres TSX, 50 titres S&P 500, indices)
	$(UV) run python scripts/fetch_data.py

run:              ## protocole complet : panel, sélection sur validation, walk-forward, CPCV, PBO/DSR, figures
	$(UV) run m2 run --country both

report:
	$(UV) run m2 report --country both
