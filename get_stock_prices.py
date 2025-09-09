import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
import time
from typing import Iterable, Set, Tuple, List
import argparse

FAILED_TICKERS_PATH = "stock_prices/no_data_tickers.json"


def _load_failed_tickers() -> Set[str]:
    if not os.path.exists(FAILED_TICKERS_PATH):
        return set()
    try:
        with open(FAILED_TICKERS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        if isinstance(data, dict) and "tickers" in data:
            return set(data["tickers"])
    except Exception:
        pass
    return set()


def _save_failed_tickers(tickers: Set[str]):
    os.makedirs(os.path.dirname(FAILED_TICKERS_PATH), exist_ok=True)
    with open(FAILED_TICKERS_PATH, "w") as f:
        json.dump(sorted(list(tickers)), f, indent=2)


# Mapas de normalización específicos para algunos tickers problemáticos
SPECIAL_MAP = {
    "BRKA": "BRK-A",
    "BRKB": "BRK-B",
    "FB": "META",
}


def normalize_to_yf_symbol(ticker: str) -> str:
    """Normaliza un ticker al formato esperado por yfinance (p.ej., BRK.B → BRK-B, BRKA→BRK-A, quita -OLD)."""
    if not isinstance(ticker, str):
        return ticker
    t = ticker.strip().upper().replace(" ", "")
    if t.endswith("-OLD"):
        t = t[:-4]
    t = t.replace(".", "-").replace("/", "-")
    return SPECIAL_MAP.get(t, t)


def _load_source_df(last_n_years: int) -> pd.DataFrame:
    """Carga el DataFrame fuente para Top-1. Usa top_positions.csv si existe; si no, usa top_positions_all_clean.csv filtrando rank==1.
    Filtra por los últimos N años.
    """
    df = None
    if os.path.exists("top_positions.csv"):
        df = pd.read_csv("top_positions.csv")
    elif os.path.exists("top_positions_all_clean.csv"):
        df = pd.read_csv("top_positions_all_clean.csv")
        # normalizar: quedarnos solo con Top-1
        if "rank" in df.columns:
            df = df[df["rank"] == 1]
    else:
        raise FileNotFoundError("No se encontró ni top_positions.csv ni top_positions_all_clean.csv")

    with pd.option_context('mode.chained_assignment', None):
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    current_year = datetime.now().year
    cutoff = current_year - (last_n_years - 1)
    df = df[df["anio"] >= cutoff]

    # Mantener solo columnas necesarias
    keep = [c for c in ["anio", "quarter", "accion"] if c in df.columns]
    df = df[keep].copy()
    # Limpiar accion
    df["accion"] = df["accion"].astype(str).str.strip().str.upper()
    return df


def get_unique_stocks(last_n_years: int = 10) -> List[str]:
    """Extrae tickers únicos de Top-1 para los últimos N años (fuente: top_positions*.csv)."""
    df = _load_source_df(last_n_years)
    unique_stocks = df["accion"].dropna().unique().tolist()
    unique_stocks = [s for s in unique_stocks if s and s != "NAN"]
    return sorted(set(unique_stocks))


def get_unique_stocks_all() -> List[str]:
    """Extrae TODOS los tickers únicos del archivo top_positions*.csv (todas las filas, todos los años)."""
    if os.path.exists("top_positions.csv"):
        df = pd.read_csv("top_positions.csv")
    elif os.path.exists("top_positions_all_clean.csv"):
        df = pd.read_csv("top_positions_all_clean.csv")
    else:
        raise FileNotFoundError("No se encontró ni top_positions.csv ni top_positions_all_clean.csv")

    if "accion" not in df.columns:
        raise ValueError("El archivo de posiciones no tiene columna 'accion'")
    with pd.option_context('mode.chained_assignment', None):
        df["accion"] = df["accion"].astype(str).str.strip().str.upper()
    unique = [s for s in df["accion"].dropna().unique().tolist() if s and s != "NAN"]
    return sorted(set(unique))


def get_quarter_end_date(year: int | str, quarter: str) -> str | None:
    """Devuelve la fecha del último día del quarter"""
    try:
        y = int(year)
    except Exception:
        return None
    quarter_end_dates = {
        "Q1": f"{y}-03-31",
        "Q2": f"{y}-06-30",
        "Q3": f"{y}-09-30",
        "Q4": f"{y}-12-31",
    }
    return quarter_end_dates.get(quarter, None)


def get_all_quarters_from_data(last_n_years: int = 10) -> List[Tuple[int, str]]:
    """Extrae todos los (año, quarter) únicos para los últimos N años (Top-1)."""
    df = _load_source_df(last_n_years)
    quarters = set()
    for _, row in df.iterrows():
        y = row.get("anio")
        q = str(row.get("quarter", "")).strip()
        if pd.isna(y) or not q:
            continue
        try:
            y = int(y)
        except Exception:
            continue
        if q in {"Q1", "Q2", "Q3", "Q4"}:
            quarters.add((y, q))
    return sorted(quarters)


def get_stock_price_for_date(yf_symbol: str, date_str: str):
    """Obtiene el precio de cierre para una fecha específica usando yfinance (símbolo normalizado)."""
    try:
        stock = yf.Ticker(yf_symbol)
        # Obtener datos desde unos días antes y después para asegurar encontrar precio
        target = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (target - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target + timedelta(days=5)).strftime("%Y-%m-%d")

        hist = stock.history(start=start_date, end=end_date)
        if hist is None or hist.empty:
            return None

        # Indice a fechas sin hora
        idx = pd.to_datetime(hist.index.date)
        target_date = pd.to_datetime(target.date())
        # Encontrar fecha más cercana
        closest_ix = int((abs(idx - target_date)).argmin())
        price = float(hist.iloc[closest_ix]["Close"])  # type: ignore[index]
        return round(price, 2)
    except Exception:
        return None


def create_stock_price_file(original_ticker: str, quarters: Iterable[Tuple[int, str]]):
    """Crea CSV con precios trimestrales de un ticker. Usa símbolo normalizado para yfinance,
    pero conserva el ticker original en el archivo y en el nombre del archivo.
    """
    print(f"📊 Procesando {original_ticker}...")
    yf_symbol = normalize_to_yf_symbol(original_ticker)

    data = []
    for year, quarter in quarters:
        date_str = get_quarter_end_date(year, quarter)
        if not date_str:
            continue
        price = get_stock_price_for_date(yf_symbol, date_str)
        if price is not None:
            data.append({
                "anio": int(year),
                "quarter": quarter,
                "fecha": date_str,
                "precio_cierre": price,
                "ticker": original_ticker,
            })

    if data:
        df = pd.DataFrame(data)
        os.makedirs("stock_prices", exist_ok=True)
        filename = f"stock_prices/{original_ticker}_quarterly_prices.csv"
        df.to_csv(filename, index=False)
        print(f"✅ {original_ticker}: {len(data)} precios guardados en {filename}")
        return True
    else:
        print(f"⚠️ {original_ticker}: No se pudieron obtener precios")
        return False


def create_stock_price_file_full_history(original_ticker: str) -> bool:
    """Descarga el histórico completo del ticker vía yfinance y genera precios trimestrales (cierre a fin de trimestre).
    Es más eficiente: una sola descarga por ticker y luego se remuestrea a fin de trimestre.
    """
    print(f"📊 (Full) Procesando {original_ticker}...")
    yf_symbol = normalize_to_yf_symbol(original_ticker)
    try:
        t = yf.Ticker(yf_symbol)
        hist = t.history(period="max")
        if hist is None or hist.empty:
            print(f"⚠️ {original_ticker}: sin histórico en yfinance")
            return False
        # Asegurar índice de fechas diario
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)
        # Usar cierre sin ajustes (Close). Si no existe, intentar 'Adj Close'
        close_col = "Close" if "Close" in hist.columns else ("Adj Close" if "Adj Close" in hist.columns else None)
        if close_col is None:
            print(f"⚠️ {original_ticker}: no hay columna Close ni Adj Close")
            return False
        # Remuestrear a fin de trimestre, tomando el último cierre disponible del trimestre
        q = hist[close_col].resample('QE').last().dropna()
        if q.empty:
            print(f"⚠️ {original_ticker}: remuestreo trimestral vacío")
            return False
        # Preparar filas
        records = []
        for ts, price in q.items():
            dt = pd.Timestamp(ts).to_pydatetime()
            year = dt.year
            month = dt.month
            quarter = "Q1" if month == 3 else ("Q2" if month == 6 else ("Q3" if month == 9 else "Q4"))
            fecha = dt.strftime("%Y-%m-%d")
            try:
                price_f = float(price)
            except Exception:
                continue
            records.append({
                "anio": int(year),
                "quarter": quarter,
                "fecha": fecha,
                "precio_cierre": round(price_f, 2),
                "ticker": original_ticker,
            })
        if not records:
            print(f"⚠️ {original_ticker}: sin registros trimestrales")
            return False
        df = pd.DataFrame(records)
        os.makedirs("stock_prices", exist_ok=True)
        filename = f"stock_prices/{original_ticker}_quarterly_prices.csv"
        df.to_csv(filename, index=False)
        print(f"✅ {original_ticker}: {len(df)} precios trimestrales (histórico completo) guardados en {filename}")
        return True
    except Exception as e:
        print(f"⚠️ {original_ticker}: error descargando histórico completo: {e}")
        return False


def main():
    """Descarga precios trimestrales para tickers.
    Modos:
      - Por defecto: Top-1 de los últimos 10 años y solo esos quarters.
      - --all-stocks: todos los tickers del dataset (todas las filas, todos los años).
      - --all-time: usa histórico completo y remuestrea a fin de trimestre.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-stocks", action="store_true", help="Usar todos los tickers de top_positions (no solo Top-1 últimos años)")
    parser.add_argument("--all-time", action="store_true", help="Descargar histórico completo y generar todos los trimestres disponibles")
    parser.add_argument("--years", type=int, default=10, help="Últimos N años (si no usa --all-time)")
    parser.add_argument("--sleep", type=float, default=0.4, help="Segundos de espera entre tickers para evitar rate limiting")
    parser.add_argument("--overwrite", action="store_true", help="Regenerar CSVs aunque existan y reintentar tickers fallidos")
    args = parser.parse_args()

    last_n_years = args.years

    # Elegir universo de tickers
    if args.all_stocks:
        print("🔍 Extrayendo TODOS los tickers únicos del dataset...")
        stocks = get_unique_stocks_all()
    else:
        print(f"🔍 Extrayendo stocks únicos (Top-1, últimos {last_n_years} años)...")
        stocks = get_unique_stocks(last_n_years=last_n_years)

    # Excluir tickers con archivo ya existente (el nombre del archivo usa el ticker original)
    os.makedirs("stock_prices", exist_ok=True)
    existing = {fn.replace("_quarterly_prices.csv", "") for fn in os.listdir("stock_prices") if fn.endswith("_quarterly_prices.csv")}

    failed = _load_failed_tickers()

    # Permitir reintento si la normalización cambia el ticker (p.ej., '-OLD' → activo)
    def should_retry(s: str) -> bool:
        norm = normalize_to_yf_symbol(s)
        return (s in failed) and (norm != s)

    if args.overwrite:
        base_candidates = stocks
        to_fetch = stocks  # reintentar todo, incluidos fallidos
    else:
        base_candidates = [s for s in stocks if s not in existing]
        to_fetch = [s for s in base_candidates if s not in failed or should_retry(s)]

    print(f"📈 Tickers únicos: {len(stocks)} | ya existentes: {len(existing)} | fallidos previos: {len(failed)} | a descargar: {len(to_fetch)}")

    if not args.all_time:
        print(f"📅 Extrayendo quarters únicos (últimos {last_n_years} años)...")
        quarters = get_all_quarters_from_data(last_n_years=last_n_years)
        print(f"📅 Quarters únicos: {len(quarters)}")

    print("💰 Descargando precios...")
    successful = 0
    failed_now = 0

    for i, ticker in enumerate(to_fetch, 1):
        print(f"[{i}/{len(to_fetch)}] ", end="")
        if args.all_time:
            ok = create_stock_price_file_full_history(ticker)
        else:
            ok = create_stock_price_file(ticker, quarters)
        if ok:
            successful += 1
            if ticker in failed:
                failed.remove(ticker)
                _save_failed_tickers(failed)
        else:
            failed_now += 1
            failed.add(ticker)
            _save_failed_tickers(failed)
        time.sleep(max(0.0, args.sleep))

    print("\n🎉 Proceso completado!")
    print(f"✅ Exitosos nuevos: {successful}")
    print(f"❌ Fallidos nuevos: {failed_now}")
    print(f"🚫 Total marcados como sin datos: {len(failed)} (ver {FAILED_TICKERS_PATH})")
    print("📁 Archivos guardados en el directorio 'stock_prices/'")


if __name__ == "__main__":
    main()
