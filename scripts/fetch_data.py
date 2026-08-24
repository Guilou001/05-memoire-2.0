"""Télécharge les prix bruts dans data/raw/ (idempotent) ; la macro se dépose à la main.

- 50 titres du TSX et 50 titres du S&P 500, prix quotidiens ajustés Yahoo (usage personnel), 2000-2024,
  plus les indices S&P/TSX composite et S&P 500. Univers de la version 1.1 du mémoire (CNQ.TO et GIB-A.TO
  remplacent le doublon ENB.TO et le FNB XIU.TO de 2024).
- LCDMA (Fortin-Gagnon, Leroux, Stevanovic et Surprenant, 2022) : déposer ``balanced_can_md.csv`` sous
  ``data/raw/macro_data.csv`` (https://www.stevanovic.uqam.ca/DS_LCMD.html).
- FRED-MD (McCracken et Ng, 2016) : déposer le millésime mensuel sous ``data/raw/Fred-MD.csv``
  (séparateur « ; », colonne ``sasdate``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
START, END = "2000-01-01", "2024-06-01"

CANADIAN_TICKERS = [
    "ABX.TO", "AEM.TO", "ATD.TO", "BB.TO", "BBD-B.TO", "BCE.TO", "BMO.TO", "BN.TO", "BLDP.TO", "BNS.TO",
    "CAE.TO", "CCA.TO", "CCL-B.TO", "CCO.TO", "CM.TO", "CNR.TO", "CTC.TO", "CTC-A.TO", "EMA.TO", "EMP-A.TO",
    "ENGH.TO", "ENB.TO", "FTS.TO", "FTT.TO", "GIL.TO", "HR-UN.TO", "IMO.TO", "L.TO", "MFC.TO", "MFI.TO",
    "MRU.TO", "MTL.TO", "NA.TO", "ONEX.TO", "POW.TO", "RCI-B.TO", "RY.TO", "SAP.TO", "SJ.TO", "STN.TO",
    "SU.TO", "T.TO", "TCL-A.TO", "TECK-B.TO", "TRP.TO", "TD.TO", "WN.TO", "WDO.TO", "CNQ.TO", "GIB-A.TO",
]
US_TICKERS = [
    "AAPL", "ABT", "ACN", "AMGN", "AMZN", "AOS", "BA", "CAT", "CL", "CMCSA", "COST", "CRM", "CSCO", "CVX",
    "DHR", "DIS", "GE", "GIS", "GLW", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "LIN",
    "LLY", "LMT", "LOW", "MCD", "MDT", "MMM", "MRK", "MSFT", "NKE", "PEP", "PFE", "PG", "QCOM", "SBUX", "T",
    "TXN", "UNH", "UPS", "VZ", "WMT", "XOM",
]
BENCHMARKS = {"TSX60": "^GSPTSE", "SP500": "^GSPC"}


def download(tickers: list[str], name: str) -> None:
    import yfinance as yf

    out = RAW / f"{name}_{START}_to_{END}.csv"
    if out.exists():
        print(f"déjà présent : {out.name}")
        return
    data = yf.download(tickers=tickers, start=START, end=END, interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        frame = pd.DataFrame({t: data[t]["Close"] for t in tickers})
    else:
        frame = data[["Close"]].rename(columns={"Close": name})
    frame.index = pd.to_datetime(frame.index).normalize()
    frame.index.name = "Date"
    frame.to_csv(out)
    print(f"écrit : {out.name} {frame.shape}")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    download(CANADIAN_TICKERS, "canadian_stocks")
    download(US_TICKERS, "us_stocks")
    for name, ticker in BENCHMARKS.items():
        download([ticker], name)
    for fname in ("macro_data.csv", "Fred-MD.csv"):
        status = "présent" if (RAW / fname).exists() else "MANQUANT (voir docstring)"
        print(f"{fname} : {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
