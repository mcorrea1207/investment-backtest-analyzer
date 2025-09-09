import os
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Set

import pandas as pd
import yfinance as yf

DAILY_DIR = "stock_prices_daily"
FAILED_TICKERS_PATH = os.path.join(DAILY_DIR, "no_data_tickers_daily.json")
ALIASES_PATH = "symbol_aliases.json"

# Mapas de normalización específicos para algunos tickers problemáticos
SPECIAL_MAP = {
    "BRKA": "BRK-A",
    "BRKB": "BRK-B",
    "FB": "META",
}


def _load_alias_map() -> dict:
    """Carga un mapa de alias desde JSON (si existe). Claves y valores en mayúsculas."""
    if not os.path.exists(ALIASES_PATH):
        return {}
    try:
        with open(ALIASES_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # normalizar a mayúsculas
            return {str(k).strip().upper(): str(v).strip().upper() for k, v in data.items() if v}
        return {}
    except Exception:
        return {}


ALIAS_MAP = _load_alias_map()

def normalize_to_yf_symbol(ticker: str) -> str:
    """Normaliza un ticker al formato esperado por yfinance (BRK.B→BRK-B, quita -OLD, etc.)."""
    if not isinstance(ticker, str):
        return ticker
    t = ticker.strip().upper().replace(" ", "")
    # quitar sufijo -OLD si existe (símbolos antiguos en scraped)
    if t.endswith("-OLD"):
        t = t[:-4]
    # normalizaciones básicas: punto o slash a guión
    t = t.replace(".", "-").replace("/", "-")
    # aplicar alias si existe
    if t in ALIAS_MAP:
        return ALIAS_MAP[t]
    # aplicar mapa especial (p.ej., BRKA -> BRK-A)
    return SPECIAL_MAP.get(t, t)


def _yf_symbol_candidates(ticker: str) -> List[str]:
    """Genera candidatos de símbolo para yfinance, útil con warrants o sufijos especiales.
    Orden de prueba: alias/normalizado → variantes de warrant (WT/WTA/WTB) si termina en WS/W.
    """
    t_raw = str(ticker).strip().upper()
    base_norm = normalize_to_yf_symbol(t_raw)
    cands = [base_norm]
    # si el original termina con WS o W, generar variantes de warrant comunes
    if t_raw.endswith("WS") or t_raw.endswith("W"):
        # remover sufijo simple para base
        b = t_raw[:-2] if t_raw.endswith("WS") else t_raw[:-1]
        b = b.rstrip("-")
        if b:
            for suf in ("WT", "WTA", "WTB"):
                cands.append(normalize_to_yf_symbol(f"{b}-{suf}"))
    # deduplicar preservando orden
    seen = set()
    out = []
    for s in cands:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _load_failed_tickers() -> Set[str]:
    if not os.path.exists(FAILED_TICKERS_PATH):
        return set()
    try:
        with open(FAILED_TICKERS_PATH, "r") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set(data.get("tickers", []))
    except Exception:
        return set()


def _save_failed_tickers(tickers: Set[str]):
    os.makedirs(DAILY_DIR, exist_ok=True)
    with open(FAILED_TICKERS_PATH, "w") as f:
        json.dump(sorted(list(tickers)), f, indent=2)


def read_universe_from_transactions(path: str = "backtest_transactions.csv") -> List[str]:
    """Lee tickers únicos desde el historial de transacciones del backtest."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    tickers = (
        df["ticker"].astype(str).str.strip().str.upper().dropna()
    )
    uniq = sorted(set([t for t in tickers if t and t != "NAN"]))
    return uniq


def read_universe_from_positions(path: str) -> List[str]:
    """Lee tickers únicos desde un CSV de posiciones con columna 'accion'."""
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    col = None
    for c in ["accion", "ticker", "symbol"]:
        if c in df.columns:
            col = c
            break
    if not col:
        return []
    vals = df[col].astype(str).str.strip().str.upper().dropna()
    uniq = sorted({v for v in vals if v and v != "NAN"})
    return uniq


def infer_date_range_from_transactions(path: str = "backtest_transactions.csv") -> tuple[str, str]:
    """Infere rango de fechas [start, end] a partir de las transacciones. Fallback: últimos 20 años."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "fecha" in df.columns:
            # Algunas filas pueden tener valores no fecha (p.ej. "2025-ACTUAL").
            dates = pd.to_datetime(df["fecha"], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                start = (dates.min() - pd.Timedelta(days=5)).date().isoformat()
                end = (datetime.now() + timedelta(days=1)).date().isoformat()
                return start, end
    # Fallback genérico: últimos 20 años
    start = (datetime.now() - timedelta(days=365 * 20)).date().isoformat()
    end = (datetime.now() + timedelta(days=1)).date().isoformat()
    return start, end


def download_daily_prices(ticker: str, start: str | None, end: str | None, from_earliest: bool = False) -> bool:
    """Descarga precios diarios para un ticker y guarda CSV estándar.

    - Cuando from_earliest=True, usa el máximo historial disponible por ticker (period="max").
    - Si es False, usa el rango [start, end].
    """
    cand_symbols = _yf_symbol_candidates(ticker)
    try:
        # Usar history del objeto Ticker para respetar posibles ajustes y throttling
        hist = None
        used_sym = None
        for yf_symbol in cand_symbols:
            t = yf.Ticker(yf_symbol)
            hist = t.history(period="max", auto_adjust=False) if from_earliest else t.history(start=start, end=end, auto_adjust=False)
            if hist is not None and not hist.empty:
                used_sym = yf_symbol
                break
        if hist is None or hist.empty:
            print(f"   ⚠️ {ticker}: sin datos diarios en yfinance (intentado como {', '.join(cand_symbols)})")
            return False
        # Asegurar índice de fechas diario y columna Close
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)
        close_col = "Close" if "Close" in hist.columns else ("Adj Close" if "Adj Close" in hist.columns else None)
        if close_col is None:
            print(f"   ⚠️ {ticker}: no hay columna Close/Adj Close")
            return False
        adj_col = "Adj Close" if "Adj Close" in hist.columns else close_col
        out = pd.DataFrame({
            "fecha": pd.to_datetime(hist.index.date),
            "precio_cierre": hist[close_col].astype(float).round(4),
            "precio_ajustado": hist[adj_col].astype(float).round(4),
        })
        out["ticker"] = ticker
        os.makedirs(DAILY_DIR, exist_ok=True)
        fp = os.path.join(DAILY_DIR, f"{ticker}_daily_prices.csv")
        out.to_csv(fp, index=False)
        # informar si se usó alias
        used_alias = ""
        t_upper = ticker.strip().upper()
        alias_info = []
        if t_upper in ALIAS_MAP:
            alias_info.append(f"alias→{ALIAS_MAP[t_upper]}")
        if used_sym and used_sym != normalize_to_yf_symbol(ticker):
            alias_info.append(f"yf→{used_sym}")
        if alias_info:
            used_alias = " (" + ", ".join(alias_info) + ")"
        print(f"   ✅ {ticker}{used_alias}: {len(out)} filas guardadas")
        return True
    except Exception as e:
        print(f"   ⚠️ {ticker}: error descargando - {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="Fecha inicio YYYY-MM-DD (default: inferida de transacciones)")
    parser.add_argument("--end", type=str, default=None, help="Fecha fin YYYY-MM-DD (default: mañana)")
    parser.add_argument("--overwrite", action="store_true", help="Reescribir archivos existentes")
    parser.add_argument("--sleep", type=float, default=0.3, help="Segundos entre descargas para evitar rate limiting")
    parser.add_argument("--extra", type=str, default="SPY,QQQ", help="Tickers extra separados por coma (benchmarks)")
    parser.add_argument("--positions-csv", type=str, default=None, help="CSV de posiciones (usa columna 'accion' para universo)")
    parser.add_argument("--from-earliest", action="store_true", help="Descargar desde la fecha más antigua disponible por cada ticker (period=max)")
    parser.add_argument("--only", type=str, default=None, help="Descargar solo esta lista de tickers separados por coma (omite universo)")
    args = parser.parse_args()

    # Universo de tickers (prioridad: positions-csv > transacciones)
    if args.only:
        tickers = [t.strip().upper() for t in args.only.split(",") if t.strip()]
        pos_tickers = []
        tx_tickers = []
        extras = []
    else:
        pos_tickers: List[str] = []
        if args.positions_csv:
            pos_tickers = read_universe_from_positions(args.positions_csv)
        tx_tickers = read_universe_from_transactions()
        extras = [t.strip().upper() for t in args.extra.split(",") if t.strip()]
        base = pos_tickers if pos_tickers else tx_tickers
        tickers = sorted(set(base + extras))

    if not tickers:
        print("⚠️ No se encontraron tickers en positions ni transacciones. Solo se usarán extras (SPY/QQQ por defecto).")
        tickers = extras

    # Rango de fechas (solo si NO se usa from-earliest)
    if args.from_earliest:
        start, end = None, None
    else:
        if args.start and args.end:
            start, end = args.start, args.end
        else:
            start, end = infer_date_range_from_transactions()

    print("📥 Descarga de precios diarios")
    src = "manual-only" if args.only else ("positions" if pos_tickers else ("transactions" if tx_tickers else "extras"))
    print(f"   🔎 Fuente universo: {src}")
    print(f"   🧾 Tickers: {len(tickers)}")
    if args.from_earliest:
        print(f"   🗓️  Rango: period=max (fecha más antigua por ticker)")
    else:
        print(f"   🗓️  Rango: {start} → {end}")

    os.makedirs(DAILY_DIR, exist_ok=True)
    existing = {fn.replace("_daily_prices.csv", "") for fn in os.listdir(DAILY_DIR) if fn.endswith("_daily_prices.csv")}
    failed = _load_failed_tickers()

    to_fetch = tickers if args.overwrite else [t for t in tickers if t not in existing or t in failed]
    print(f"   📌 Ya existen: {len(existing)} | Fallidos previos: {len(failed)} | A descargar: {len(to_fetch)}")

    ok_count = 0
    fail_count = 0

    for i, t in enumerate(to_fetch, 1):
        print(f"[{i}/{len(to_fetch)}] {t}")
        ok = download_daily_prices(t, start, end, from_earliest=args.from_earliest)
        if ok:
            ok_count += 1
            if t in failed:
                failed.remove(t)
                _save_failed_tickers(failed)
        else:
            fail_count += 1
            failed.add(t)
            _save_failed_tickers(failed)
        time.sleep(max(0.0, args.sleep))

    print("\n🎉 Descarga completada")
    print(f"   ✅ Nuevos OK: {ok_count}")
    print(f"   ❌ Nuevos fallidos: {fail_count}")
    print(f"   🚫 Total marcados como sin datos: {len(failed)} (ver {FAILED_TICKERS_PATH})")
    print(f"   📁 Archivos: {DAILY_DIR}/<TICKER>_daily_prices.csv")


if __name__ == "__main__":
    main()
