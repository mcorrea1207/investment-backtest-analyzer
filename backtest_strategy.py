import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import sys
from io import StringIO
import random
import matplotlib.pyplot as plt
import numpy as np

# ========================================
# CONFIGURACIÓN - EDITA AQUÍ TUS PARÁMETROS
# ========================================
NUM_INVESTORS = 5        # Número de inversores a seguir (se elegirán aleatoriamente)
START_YEAR = 2012          # Año de inicio del backtest (permite ver Antonio desde 2019)
END_YEAR = 2026            # Año final (inclusive para compras; la venta final se hace después). Usado para análisis.
HAS_RELOCATION = True     # True = Con rebalanceo | False = Sin rebalanceo
REDISTRIBUTE_FAILED = True # True = Redistribuir capital de inversores fallidos | False = Mantener en efectivo
RANDOM_SEED = 1207         # Semilla para reproducibilidad (cambia para obtener diferentes inversores aleatorios)
MIN_YEARS_DATA = 1         # Mínimo de años con datos Q3 para incluir un inversor
# Modo de selección de inversores
# 0 = aleatorio
# 1 = top NUM_INVESTORS por rendimiento previo (backtest interno)
# 2 = top por valor de portafolio (Q3 año previo)
# 3 = ranking por CAGR 3y (usa cagr_merged.csv) -> requiere columnas cagr_raw_3y / cagr_gated_3y
# 4 = ranking por CAGR acumulada (usa cagr_merged.csv) -> columnas cagr_raw_all / cagr_gated_all
SELECTION_MODE = 3

# Config para modos 3 y 4
CAGR_SOURCE_CSV = 'cagr_merged.csv'   # Archivo con métricas CAGR combinadas
# CAGR_METRIC controla si usas RAW (sin gate de cobertura) o GATED (con gate) y si es 3y o acumulada.
# Valores válidos:
#  - 'cagr_raw_3y'      -> CAGR trienal usando year_return_raw
#  - 'cagr_gated_3y'    -> CAGR trienal usando year_return (gated)
#  - 'cagr_raw_all'     -> CAGR acumulada (todos los años previos) RAW
#  - 'cagr_gated_all'   -> CAGR acumulada (todos los años previos) GATED
CAGR_METRIC = 'cagr_gated_3y'
CAGR_MIN_YEARS = 3                    # Mínimo años en ventana (para acumulada usa years_in_window)
CAGR_DESCENDING = True                # True = mayor CAGR es mejor
SHOW_ALPHA = False                    # Si True mantiene impresión de alpha anual; False = sólo comparación CAGR
PRIOR_END_QUARTER = 'Q4'   # Quarter de salida para evaluar el año previo (Q3 por defecto)
# Timing de compras y ventas
BUY_QUARTER = 'Q1'         # Quarter en que se compran las acciones (Q1, Q2, Q3, Q4)
SELL_QUARTER = 'Q4'        # Quarter en que se venden las acciones del año anterior (Q1, Q2, Q3, Q4)
DATA_SOURCE_QUARTER = 'Q4' # Quarter del año anterior del cual se toman las posiciones para comprar
MIN_TOP1_PCT = 15          # Umbral (% del portafolio) mínimo para la Top-1; ej. 10.0. None para desactivar
BENCHMARK_TICKER = 'SPY'  # Proxy del S&P 500 para benchmark (usa CSV local stock_prices/SPY_quarterly_prices.csv)
TOP_PICKS_PER_INVESTOR = 3  # Número de top picks por inversor (si hay menos, se reparte entre las que existan)
GENERATE_INVESTOR_TRACK_RECORD = False # Generar track record por inversor (trimestral y anual)
# Ventana configurable para el track record histórico relativo a un año de compra
LOOKBACK_BUY_YEAR = 2019   # Año de compra (ej.: 2019 => ventana 2016-2018 si LOOKBACK_YEARS=3)
LOOKBACK_YEARS = 3         # Años hacia atrás para evaluar el track record
# Opciones de curvas extra en el gráfico
ALWAYS_INCLUDE_RANDOM_BASELINE = True   # Ejecuta una simulación adicional modo aleatorio para graficar como baseline
INCLUDE_ANTONIO_LINE = True             # Agrega curva sintética Antonio (tickers definidos abajo)
ANTONIO_YEAR_TICKER_MAP = {
    2018: 'BSMX', 2019: 'META', 2020: 'SIG',
    2021: 'MSFT', 2022: 'MSFT', 2023: 'META', 2024: 'GOOGL'
}
BRANCH_YEAR = 2019                 # Año en el que empieza a mostrarse la rama Antonio
MAURICIO_COLOR = 'blue'
ANTONIO_COLOR = '#ff69b4'          # Rosa
RANDOM_BASELINE_COLOR = '#8B4513'  # Marrón / brown
# ========================================

from investor_track_record import (
    compute_investor_track_record,
    save_investor_track_record,
    build_lookback_track_record,
)

def get_stock_price_for_date(ticker, date_str, mode: str = 'nearest'):
    """Obtiene el precio de cierre desde CSV local para una fecha objetivo.
    mode:
      - 'on_or_after': primer registro con fecha >= objetivo
      - 'on_or_before': último registro con fecha <= objetivo
      - 'nearest' (default): fecha más cercana al objetivo
    """
    try:
        csv_path = os.path.join("stock_prices", f"{ticker}_quarterly_prices.csv")
        if not os.path.exists(csv_path):
            # Archivo no disponible
            return None
        df = pd.read_csv(csv_path)
        if df.empty or 'fecha' not in df.columns or 'precio_cierre' not in df.columns:
            return None

        # Convertir fechas
        df['fecha'] = pd.to_datetime(df['fecha'])
        target_date = pd.to_datetime(date_str)

        if mode == 'on_or_after':
            candidates = df[df['fecha'] >= target_date]
            if candidates.empty:
                return None
            row = candidates.sort_values('fecha').iloc[0]
        elif mode == 'on_or_before':
            candidates = df[df['fecha'] <= target_date]
            if candidates.empty:
                return None
            row = candidates.sort_values('fecha').iloc[-1]
        else:  # nearest
            diffs = (df['fecha'] - target_date).abs()
            idx = diffs.idxmin()
            row = df.loc[idx]

        try:
            price = float(row['precio_cierre'])
        except Exception:
            return None
        return round(price, 2)
    except Exception as e:
        print(f"⚠️ Error leyendo precio de CSV para {ticker} en {date_str}: {e}")
        return None


def get_latest_price_for_ticker(ticker: str):
    """Obtiene el último precio disponible desde el CSV local del ticker, priorizando daily prices."""
    try:
        # Intentar primero con daily prices
        daily_csv_path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
        if os.path.exists(daily_csv_path):
            df = pd.read_csv(daily_csv_path)
            if not df.empty and 'precio_cierre' in df.columns:
                price = df['precio_cierre'].iloc[-1]
                try:
                    price = float(price)
                    return round(price, 2)
                except Exception:
                    pass
        
        # Fallback a quarterly prices
        csv_path = os.path.join("stock_prices", f"{ticker}_quarterly_prices.csv")
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if df.empty or 'precio_cierre' not in df.columns:
            return None
        # Tomar el último registro (asumiendo orden cronológico por archivo)
        price = df['precio_cierre'].iloc[-1]
        try:
            price = float(price)
        except Exception:
            return None
        return round(price, 2)
    except Exception as e:
        print(f"⚠️ Error leyendo último precio de CSV para {ticker}: {e}")
        return None

def get_positions_for_year_quarter(year, quarter, num_investors, selected_investors=None, min_top1_pct=None, top_k: int = TOP_PICKS_PER_INVESTOR):
    """Obtiene las posiciones de un quarter específico para un año específico.
    Soporta tomar las Top-K (rank 1..K) de cada inversor si existe el archivo 'top_positions_all_clean.csv'.
    Si un inversor tiene menos de K picks, se devuelven las que existan.
    Si min_top1_pct está definido, se filtra por el % de cada pick dentro de las Top-K; si un inversor no tiene
    ninguna pick >= umbral, se excluye al inversor.
    """
    use_all = os.path.exists("top_positions_all_clean.csv")

    if use_all:
        df = pd.read_csv("top_positions_all_clean.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'anio' in df.columns:
                df['anio'] = pd.to_numeric(df['anio'], errors='coerce')
            if 'porcentaje_portafolio' in df.columns:
                df['porcentaje_portafolio'] = pd.to_numeric(df['porcentaje_portafolio'], errors='coerce')
                # Normalizar si viniera en 0-1
                max_val = df['porcentaje_portafolio'].dropna().max()
                if pd.notna(max_val) and max_val is not None and max_val <= 1.5:
                    df['porcentaje_portafolio'] = df['porcentaje_portafolio'] * 100
        quarter_data = df[(df['anio'] == year) & (df['quarter'] == quarter)]
        if selected_investors is None:
            # Si no se especifican, usar hasta num_investors inversores únicos presentes en el quarter
            available_investors = quarter_data['inversionista'].dropna().unique().tolist()
            selected = available_investors[:min(num_investors, len(available_investors))]
            quarter_data = quarter_data[quarter_data['inversionista'].isin(selected)]
        else:
            quarter_data = quarter_data[quarter_data['inversionista'].isin(selected_investors)]

        # Tomar hasta K mejores por inversor según rank asc
        if 'rank' in quarter_data.columns:
            quarter_data = (quarter_data.sort_values(['inversionista', 'rank'])
                    .groupby('inversionista', group_keys=False)
                    .head(int(max(1, top_k))))

        # Aplicar umbral por pick (no sólo Top-1). Si un inversor no tiene ninguna pick >= umbral, se excluye.
        filtered_out = 0
        before_investors = quarter_data['inversionista'].nunique() if not quarter_data.empty else 0
        if min_top1_pct is not None and 'porcentaje_portafolio' in quarter_data.columns:
            thr = float(min_top1_pct)
            quarter_data = quarter_data[pd.to_numeric(quarter_data['porcentaje_portafolio'], errors='coerce') >= thr]
            after_investors = quarter_data['inversionista'].nunique() if not quarter_data.empty else 0
            filtered_out = max(0, before_investors - after_investors)

        used_count = quarter_data['inversionista'].nunique() if not quarter_data.empty else 0
        # Mensaje: "originales que cumplen" = inversores que quedan con ≥1 pick tras el filtro de % sobre las Top-K
        print(f"    📋 {quarter} {year}: Solicitados {num_investors}, originales que cumplen: {used_count}" + (f" | filtrados por %TopK: {filtered_out}" if filtered_out else ""))
        return quarter_data

    # Fallback: dataset reducido (Top-1 por fila)
    df = pd.read_csv("top_positions.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'porcentaje_portafolio' in df.columns:
            df['porcentaje_portafolio'] = pd.to_numeric(df['porcentaje_portafolio'], errors='coerce')
            max_val = df['porcentaje_portafolio'].dropna().max()
            if pd.notna(max_val) and max_val is not None and max_val <= 1.5:
                df['porcentaje_portafolio'] = df['porcentaje_portafolio'] * 100
    quarter_positions = df[(df['anio'] == year) & (df['quarter'] == quarter)]
    if selected_investors is None:
        available_investors = quarter_positions['inversionista'].nunique()
        max_possible = min(num_investors, available_investors)
        unique_investors = quarter_positions['inversionista'].unique()[:max_possible]
        quarter_positions = quarter_positions[quarter_positions['inversionista'].isin(unique_investors)]
    else:
        quarter_positions = quarter_positions[quarter_positions['inversionista'].isin(selected_investors)]
    filtered_out = 0
    before_investors = quarter_positions['inversionista'].nunique() if not quarter_positions.empty else 0
    if min_top1_pct is not None and 'porcentaje_portafolio' in quarter_positions.columns:
        quarter_positions = quarter_positions[quarter_positions['porcentaje_portafolio'] >= float(min_top1_pct)]
        after_investors = quarter_positions['inversionista'].nunique() if not quarter_positions.empty else 0
        filtered_out = max(0, before_investors - after_investors)
    used_count = quarter_positions['inversionista'].nunique() if not quarter_positions.empty else 0
    if top_k > 1:
        print("    ⚠️ Aviso: top_positions.csv no contiene múltiples ranks; sólo se tomará la Top-1 por inversor")
    print(f"    📋 {quarter} {year}: Solicitados {num_investors}, originales que cumplen: {used_count}" + (f" | filtrados por %Top1: {filtered_out}" if filtered_out else ""))
    return quarter_positions

def get_q3_positions_for_year(year, num_investors, selected_investors=None, min_top1_pct=None, top_k: int = TOP_PICKS_PER_INVESTOR):
    """Backward compatibility function - delegates to get_positions_for_year_quarter with Q3"""
    return get_positions_for_year_quarter(year, 'Q3', num_investors, selected_investors, min_top1_pct, top_k)

def select_random_investors(num_investors, start_year, random_seed=None):
    """Selecciona aleatoriamente inversores del dataset que tengan suficientes datos históricos"""
    if random_seed is not None:
        random.seed(random_seed)
    
    df = pd.read_csv("top_positions.csv")
    
    # Obtener años que vamos a necesitar (desde start_year-1 hasta 2024)
    required_years = list(range(start_year - 1, 2025))
    
    # Analizar qué inversores tienen datos Q3 para los años requeridos
    investor_coverage = {}
    for investor in df['inversionista'].unique():
        investor_data = df[df['inversionista'] == investor]
        q3_data = investor_data[investor_data['quarter'] == 'Q3']
        years_with_data = sorted(q3_data['anio'].unique())
        
        # Calcular cobertura de años requeridos
        years_covered = [year for year in required_years if year in years_with_data]
        coverage_percentage = len(years_covered) / len(required_years) * 100
        
        investor_coverage[investor] = {
            'years_with_data': years_with_data,
            'years_covered': years_covered,
            'missing_years': [year for year in required_years if year not in years_with_data],
            'coverage_percentage': coverage_percentage,
            'total_years': len(years_with_data)
        }
    
    # Filtrar inversores con cobertura mínima
    min_coverage = MIN_YEARS_DATA / len(required_years) * 100
    eligible_investors = [
        investor for investor, data in investor_coverage.items() 
        if data['coverage_percentage'] >= min_coverage
    ]
    
    print(f"🔍 Análisis de cobertura de datos (años {start_year-1}-2024):")
    print(f"   📊 Años requeridos: {len(required_years)} ({required_years[0]}-{required_years[-1]})")
    print(f"   ✅ Inversores elegibles (≥{min_coverage:.0f}% cobertura): {len(eligible_investors)}")
    print(f"   ❌ Inversores excluidos: {len(df['inversionista'].unique()) - len(eligible_investors)}")
    
    # Mostrar detalles de cobertura
    print(f"\n📋 Cobertura por inversor:")
    for investor in sorted(df['inversionista'].unique()):
        data = investor_coverage[investor]
        status = "✅" if investor in eligible_investors else "❌"
        print(f"   {status} {investor}: {data['coverage_percentage']:.0f}% ({len(data['years_covered'])}/{len(required_years)} años)")
        if data['missing_years']:
            print(f"      Años faltantes: {data['missing_years']}")
    
    # Si pedimos más inversores de los elegibles, usar todos los elegibles
    actual_num = min(num_investors, len(eligible_investors))
    
    if actual_num < num_investors:
        print(f"⚠️ AVISO: Se solicitaron {num_investors} inversores, pero solo {actual_num} tienen suficientes datos.")
    
    # Seleccionar aleatoriamente de los elegibles
    selected = random.sample(eligible_investors, actual_num)
    
    print(f"\n🎲 Inversores seleccionados aleatoriamente (semilla: {random_seed}):")
    for i, investor in enumerate(selected, 1):
        data = investor_coverage[investor]
        print(f"   {i:2d}. {investor} ({data['coverage_percentage']:.0f}% cobertura)")
    print()
    
    return selected, investor_coverage

def _quarter_end_date(year: int, quarter: str) -> str:
    mapping = {
        'Q1': '03-31',
        'Q2': '06-30',
        'Q3': '09-30',
        'Q4': '12-31',
    }
    day = mapping.get(quarter.upper(), '12-31')
    return f"{year}-{day}"

def select_investors_by_prior_performance(start_year: int, top_n: int, end_quarter: str = 'Q3', mode: str = 'prior_top', random_seed: int | None = None):
    """
    Selecciona inversionistas según rendimiento del año previo usando su posición top de Q3 del año (start_year-2),
    comprada el 1/1/(start_year-1) y vendida al cierre de end_quarter de (start_year-1).

    Args:
        start_year: Año objetivo del backtest (ej. 2020)
        top_n: Número de inversionistas a seleccionar
        end_quarter: Quarter de salida para evaluar el año previo (por defecto 'Q3')
        mode: 'prior_top' para ranking por rendimiento, 'random' para selección aleatoria
        random_seed: Semilla para la aleatoriedad

    Returns:
        (selected_investors, ranking_df)
        - selected_investors: lista de nombres de inversionistas seleccionados
        - ranking_df: DataFrame con columnas [inversionista, ticker, buy_price, sell_price, ret_pct]
    """
    if mode == 'random':
        selected, _coverage = select_random_investors(top_n, start_year, random_seed)
        return selected, None

    df = pd.read_csv('top_positions.csv')
    # Asegurar tipos numéricos
    if df['anio'].dtype != 'int64' and df['anio'].dtype != 'int32':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce')
    
    q_year = start_year - 2            # p.ej. 2018 si start_year=2020
    buy_year = start_year - 1          # p.ej. 2019
    buy_date = _quarter_end_date(q_year, 'Q3')  # comprar al cierre de Q3 del año q_year
    sell_date = _quarter_end_date(buy_year, end_quarter)  # vender al cierre del end_quarter del año buy_year

    # Tomar la top position Q3 del año q_year
    q3_positions = df[(df['anio'] == q_year) & (df['quarter'] == 'Q3')]
    if q3_positions.empty:
        print(f"⚠️ No hay posiciones Q3 para el año {q_year}")
        return [], None

    results = []
    print(f"🔎 Evaluando desempeño previo: comprar {buy_date} la top Q3 {q_year}, vender {sell_date}")
    for _, row in q3_positions.iterrows():
        investor = row['inversionista']
        ticker = row['accion']
        try:
            buy_price = get_daily_price_for_date(ticker, buy_date, mode='on_or_before') or get_stock_price_for_date(ticker, buy_date, mode='on_or_before')
            sell_price = get_daily_price_for_date(ticker, sell_date, mode='on_or_before') or get_stock_price_for_date(ticker, sell_date, mode='on_or_before')
            if buy_price and sell_price and buy_price > 0:
                ret_pct = (sell_price - buy_price) / buy_price * 100.0
                results.append({
                    'inversionista': investor,
                    'ticker': ticker,
                    'buy_price': round(buy_price, 4),
                    'sell_price': round(sell_price, 4),
                    'ret_pct': round(ret_pct, 4)
                })
        except Exception as e:
            print(f"   ⚠️ Error evaluando {investor} - {ticker}: {e}")
            continue

    if not results:
        print("⚠️ No se pudo calcular rendimiento previo para ningún inversionista")
        return [], None

    ranking_df = pd.DataFrame(results).sort_values('ret_pct', ascending=False).reset_index(drop=True)

    # Selección top N
    selected_investors = list(ranking_df.head(top_n)['inversionista'])

    print("🏆 Ranking previo (Top 10 mostrados):")
    for i, r in ranking_df.head(10).iterrows():
        print(f"  {i+1:2d}. {r['inversionista']}: {r['ticker']}  {r['ret_pct']:.2f}%")

    return selected_investors, ranking_df

def select_investors_by_portfolio_value(start_year: int, top_n: int, quarter: str = 'Q3'):
    """Selecciona los inversores con mayor 'portfolio_value_usd' en el quarter indicado del año (start_year-1).
    Requiere el archivo 'top_positions_all_clean.csv'.

    Retorna (selected_investors, ranking_df)
    """
    csv_path = 'top_positions_all_clean.csv'
    if not os.path.exists(csv_path):
        print("⚠️ No existe top_positions_all_clean.csv; no se puede seleccionar por valor de portafolio")
        return [], None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Error leyendo {csv_path}: {e}")
        return [], None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'anio' in df.columns:
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce')
        if 'portfolio_value_usd' in df.columns:
            df['portfolio_value_usd'] = pd.to_numeric(df['portfolio_value_usd'], errors='coerce')

    y = start_year - 1
    dfq = df[(df['anio'] == y) & (df['quarter'] == quarter)]
    if dfq.empty:
        print(f"⚠️ No hay datos en {quarter} {y} para seleccionar por valor de portafolio")
        return [], None

    # portfolio_value_usd puede repetirse por fila; tomar métrica agregada por inversor (máximo)
    agg = (dfq.groupby('inversionista', as_index=False)['portfolio_value_usd']
              .max().sort_values('portfolio_value_usd', ascending=False).reset_index(drop=True))
    
    # Seleccionar los top N inversores por valor de portafolio (ranking natural)
    selected = agg.head(top_n)['inversionista'].tolist()

    print("🏦 Top inversores por valor de portafolio:")
    for i, row in enumerate(agg.head(top_n).itertuples(index=False), 1):
        try:
            print(f"  {i:2d}. {row.inversionista}: ${row.portfolio_value_usd:,.0f}")
        except Exception:
            print(f"  {i:2d}. {row.inversionista}: {row.portfolio_value_usd}")

    return selected, agg

def select_investors_by_cagr(mode: int, top_n: int, metric: str, source_csv: str, min_years: int = 3, descending: bool = True):
    """Selecciona inversionistas usando un archivo de métricas CAGR (cagr_merged.csv).
    mode: 3 (3y) o 4 (acumulada) sólo usado para logs.
    metric: nombre de la columna a ordenar (ej: cagr_raw_3y, cagr_gated_all)
    min_years: para acumulada filtra years_in_window >= min_years; para 3y fuerza que existan columnas window3_start/window3_end.
    descending: True => mayor CAGR mejor.
    Retorna (lista_inversores, df_ranking)
    """
    if not os.path.exists(source_csv):
        print(f"⚠️ No existe {source_csv}; no se puede seleccionar por CAGR")
        return [], None
    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        print(f"⚠️ Error leyendo {source_csv}: {e}")
        return [], None
    if metric not in df.columns:
        print(f"⚠️ Métrica {metric} no existe en {source_csv}")
        print(f"   Columnas disponibles: {list(df.columns)}")
        return [], None
    # Filtrado por años mínimos
    if metric.endswith('_all') and 'years_in_window' in df.columns:
        df = df[df['years_in_window'] >= int(min_years)]
    elif metric.endswith('_3y'):
        # asegurar que el 3y sea válido (las columnas window3_start y end presentes)
        if 'window3_start' in df.columns and 'window3_end' in df.columns:
            df = df[~df['window3_start'].isna() & ~df['window3_end'].isna()]
    # Quitar filas NaN en la métrica
    df = df[~df[metric].isna()]
    if df.empty:
        print(f"⚠️ No hay filas válidas para la métrica {metric}")
        return [], None
    # Para cada inversionista, tomar la fila más reciente (buy_year máximo) para ranking
    df_latest = (df.sort_values(['inversionista','buy_year'])
                   .groupby('inversionista', as_index=False)
                   .tail(1))
    df_rank = df_latest.sort_values(metric, ascending=not descending).reset_index(drop=True)
    selected = df_rank.head(top_n)['inversionista'].tolist()
    label = 'CAGR 3y' if mode == 3 else 'CAGR acumulada'
    print(f"🏁 Ranking por {label} usando {metric} (top {top_n}):")
    for i, row in enumerate(df_rank.head(top_n).itertuples(index=False), 1):
        try:
            val = getattr(row, metric)
        except Exception:
            val = None
        print(f"  {i:2d}. {row.inversionista}: {val:.4f}")
    return selected, df_rank

def backtest_strategy(num_investors, start_year, has_relocation=None, selected_investors=None, investor_coverage=None, min_top1_pct=None, end_year: int | None = None, generate_chart: bool = True):
    """
    Ejecuta la estrategia de backtesting
    
    Args:
        num_investors: Número de inversores a seguir
        start_year: Año de inicio (int)
        has_relocation: Si permite rebalanceo o no
        selected_investors: Lista de inversores específicos a usar (opcional)
        investor_coverage: Datos de cobertura de inversores (opcional)
        min_top1_pct: Si no es None, mínimo % del portafolio en la Top-1 para incluir al inversor
    """
    
    # Si no se especifica, usar la configuración global
    if has_relocation is None:
        has_relocation = HAS_RELOCATION
    if min_top1_pct is None:
        min_top1_pct = MIN_TOP1_PCT

    # Verificar cuántos inversores hay en total en el dataset
    df_check = pd.read_csv("top_positions.csv")
    total_available_investors = df_check['inversionista'].nunique()
    
    print(f"📊 Total de inversores disponibles en el dataset: {total_available_investors}")
    
    # Si no se especificaron inversores, seleccionar con análisis de cobertura
    if selected_investors is None:
        selected_investors, investor_coverage = select_random_investors(num_investors, start_year, RANDOM_SEED)
    
    # Usar exactamente el número de inversores seleccionados
    actual_num_investors = len(selected_investors)
    
    print(f"🚀 Iniciando backtest con {actual_num_investors} inversores desde {start_year}")
    print(f"💰 Capital inicial: $10,000,000")
    print(f"🔄 Rebalanceo: {'Sí' if has_relocation else 'No'}")
    print(f"💸 Redistribuir inversores fallidos: {'Sí' if REDISTRIBUTE_FAILED else 'No (mantener en efectivo)'}")
    print(f"🔎 Filtro mínimo % Top-1: {'N/A' if min_top1_pct is None else f'>= {min_top1_pct}%'}\n")
    
    initial_capital = 10_000_000
    current_capital = initial_capital
    
    # Inicializar capital individual por inversor (para modo sin rebalanceo)
    investor_capitals = {}
    capital_per_investor = initial_capital / actual_num_investors
    for investor in selected_investors:
        investor_capitals[investor] = capital_per_investor
    
    print(f"💰 Capital inicial por inversor: ${capital_per_investor:,.2f}\n")
    
    # Comenzar comprando posiciones del DATA_SOURCE_QUARTER del año anterior al start_year
    buy_year = start_year - 1
    current_year = start_year
    
    portfolio = {}  # {(ticker, investor): {'shares': shares, 'investor': investor, 'buy_price': buy_price}}
    transactions = []  # Historial de transacciones
    yearly_performance = []  # Performance anual
    # Seguimiento para alpha: capital inicial del año y precio de benchmark de compra (Q4 previo)
    start_capital_by_year = {}
    bench_buy_by_year = {}
    
    if end_year is None:
        end_year = END_YEAR
    while current_year <= end_year:  # Incluir end_year para comprar posiciones
        print(f"📅 Procesando año {current_year}")
        
        # ===== VENTA (al final del SELL_QUARTER del año anterior) =====
        sell_date = _quarter_end_date(current_year-1, SELL_QUARTER)
        total_sale_value = 0
        
        if portfolio:  # Si tenemos posiciones
            print(f"  💸 Vendiendo posiciones el {sell_date} ({SELL_QUARTER} {current_year-1})")
            
            if has_relocation:
                # CON REBALANCEO: Todas las ventas van a un pool común
                total_sale_value = 0
                for (ticker, inv), position in portfolio.items():
                    # Use daily prices first, fallback to quarterly
                    sell_price = get_daily_price_for_date(ticker, sell_date, mode='on_or_before')
                    if not sell_price:
                        sell_price = get_stock_price_for_date(ticker, sell_date, mode='on_or_before')
                    if sell_price:
                        sale_value = position['shares'] * sell_price
                        total_sale_value += sale_value
                        transactions.append({
                            'fecha': sell_date,
                            'accion': 'VENTA',
                            'ticker': ticker,
                            'precio': sell_price,
                            'shares': position['shares'],
                            'valor': sale_value,
                            'inversionista': inv
                        })
                        print(f"    📤 Vendido {position['shares']:.4f} acciones de {ticker} a ${sell_price:.2f} = ${sale_value:,.2f} ({inv})")
                current_capital = total_sale_value
                print(f"  💰 Capital total después de ventas: ${current_capital:,.2f}")
            else:
                # SIN REBALANCEO: Cada venta va al capital individual del inversor
                for (ticker, inv), position in portfolio.items():
                    # Use daily prices first, fallback to quarterly
                    sell_price = get_daily_price_for_date(ticker, sell_date, mode='on_or_before')
                    if not sell_price:
                        sell_price = get_stock_price_for_date(ticker, sell_date, mode='on_or_before')
                    if sell_price:
                        sale_value = position['shares'] * sell_price
                        investor_capitals[inv] = sale_value
                        transactions.append({
                            'fecha': sell_date,
                            'accion': 'VENTA',
                            'ticker': ticker,
                            'precio': sell_price,
                            'shares': position['shares'],
                            'valor': sale_value,
                            'inversionista': inv
                        })
                        print(f"    📤 {inv}: Vendido {position['shares']:.4f} acciones de {ticker} a ${sell_price:.2f} = ${sale_value:,.2f}")
                current_capital = sum(investor_capitals.values())
                print(f"  💰 Capital total después de ventas: ${current_capital:,.2f}")
                print(f"    📊 Capital por inversor:")
                for investor, capital in investor_capitals.items():
                    print(f"      - {investor}: ${capital:,.2f}")
            
            portfolio = {}  # Limpiar portfolio

            # === Rendimiento y alpha del año previo ===
            prev_year = current_year - 1
            if prev_year in start_capital_by_year:
                start_cap = start_capital_by_year.get(prev_year)
                bench_buy = bench_buy_by_year.get(prev_year)
                # Use daily prices first, fallback to quarterly
                bench_sell = get_daily_price_for_date(BENCHMARK_TICKER, sell_date, mode='on_or_before')
                if not bench_sell:
                    bench_sell = get_stock_price_for_date(BENCHMARK_TICKER, sell_date, mode='on_or_before')

                port_ret = None
                bench_ret = None
                alpha = None
                if start_cap and start_cap > 0:
                    port_ret = (current_capital / start_cap) - 1
                if bench_buy and bench_sell and bench_buy > 0:
                    bench_ret = (bench_sell - bench_buy) / bench_buy
                if port_ret is not None and bench_ret is not None:
                    alpha = port_ret - bench_ret

                yearly_performance.append({
                    'year': prev_year,
                    'portfolio_return': None if port_ret is None else round(port_ret, 6),
                    'benchmark_return': None if bench_ret is None else round(bench_ret, 6),
                    'alpha': None if alpha is None else round(alpha, 6)
                })
                pr_s = 'N/A' if port_ret is None else f"{port_ret*100:.2f}%"
                br_s = 'N/A' if bench_ret is None else f"{bench_ret*100:.2f}%"
                al_s = 'N/A' if alpha is None else f"{alpha*100:.2f}%"
                print(f"  📊 {prev_year} | Mauricio: {pr_s} | SPX: {br_s} | Alpha: {al_s}")
        
        # ===== COMPRA (al final del BUY_QUARTER) =====
        buy_date = _quarter_end_date(current_year, BUY_QUARTER)

        # Registrar capital inicial del año y precio de benchmark de compra
        start_capital_by_year[current_year] = current_capital
        # Use daily prices first, fallback to quarterly
        bench_buy_price = get_daily_price_for_date(BENCHMARK_TICKER, buy_date, mode='on_or_after')
        if not bench_buy_price:
            bench_buy_price = get_stock_price_for_date(BENCHMARK_TICKER, buy_date, mode='on_or_before')
        bench_buy_by_year[current_year] = bench_buy_price
        
        # Obtener posiciones del DATA_SOURCE_QUARTER del año anterior (aplicando filtro de % Top-1 si corresponde)
        positions = get_positions_for_year_quarter(buy_year, DATA_SOURCE_QUARTER, actual_num_investors, selected_investors, min_top1_pct=min_top1_pct, top_k=TOP_PICKS_PER_INVESTOR)

        # Intentar reemplazos si faltan inversores tras el filtro
        if min_top1_pct is not None:
            used_investors = set(positions['inversionista'].unique())
            missing_investors = [inv for inv in selected_investors if inv not in used_investors]
            if missing_investors:
                # Buscar candidatos de reemplazo (excluyendo todos los seleccionados para no reciclar)
                candidates = find_replacement_positions(buy_year, exclude_investors=set(selected_investors), needed=len(missing_investors), min_top1_pct=min_top1_pct, quarter=DATA_SOURCE_QUARTER)
                replacements_rows = []
                for i, orig_inv in enumerate(missing_investors):
                    if i >= len(candidates):
                        break
                    rep = candidates.iloc[i].copy()
                    # Marcar a quién reemplaza
                    rep['replacement_from'] = orig_inv
                    # Identidad del inversor en la operación:
                    # - Con rebalanceo: mantener el inversor del candidato (logs más intuitivos)
                    # - Sin rebalanceo: usar el inversor original para que su capital se aplique correctamente
                    if not has_relocation:
                        rep['inversionista'] = orig_inv
                    # Si hay rebalanceo, dejamos el nombre del candidato tal cual
                    replacements_rows.append(rep)
                if replacements_rows:
                    positions = pd.concat([positions, pd.DataFrame(replacements_rows)], ignore_index=True)
                    print(f"    🔁 Reemplazos aplicados: {len(replacements_rows)} inversores no alcanzaron el {min_top1_pct}% y fueron sustituidos en {DATA_SOURCE_QUARTER} {buy_year}")
                else:
                    print(f"    ⚠️ No hay suficientes candidatos para reemplazar a {len(missing_investors)} inversores en {DATA_SOURCE_QUARTER} {buy_year}")
        
        if positions.empty:
            print(f"  ⚠️ No hay posiciones {DATA_SOURCE_QUARTER} para {buy_year} tras aplicar filtros")
            current_year += 1
            buy_year += 1
            continue
        
        print(f"  🛒 Comprando posiciones {DATA_SOURCE_QUARTER} de {buy_year} el {buy_date} ({BUY_QUARTER} {current_year})")
        
        # Verificar qué acciones están disponibles
        valid_positions = []
        failed_positions = []
        
        print(f"    🔍 Analizando {len(positions)} posiciones {DATA_SOURCE_QUARTER} de {buy_year}:")
        for _, position in positions.iterrows():
            ticker = position['accion']
            investor = position['inversionista']
            suffix = ""
            if 'replacement_from' in position and isinstance(position['replacement_from'], str) and position['replacement_from']:
                suffix = f" (reemplazo de {position['replacement_from']})"
            print(f"      - {investor}: {ticker}{suffix}", end="")
            # Use daily prices first, fallback to quarterly
            buy_price = get_daily_price_for_date(ticker, buy_date, mode='on_or_after')
            if not buy_price:
                buy_price = get_stock_price_for_date(ticker, buy_date, mode='on_or_before')
            if buy_price and buy_price > 0:
                # Guardar adicionalmente el rank si existe
                rank = position['rank'] if 'rank' in position and not pd.isna(position['rank']) else None
                valid_positions.append({
                    'ticker': ticker,
                    'investor': investor,
                    'price': buy_price,
                    'rank': rank
                })
                print(f" ✅ ${buy_price:.2f}")
            else:
                failed_positions.append({
                    'ticker': ticker,
                    'investor': investor
                })
                print(f" ❌ Sin precio")
        
        print(f"    📊 Total: {len(positions)} posiciones, {len(valid_positions)} válidas, {len(failed_positions)} fallidas")
        
        # Determinar estrategia de capital
        if valid_positions:
            if has_relocation:
                # CON REBALANCEO: repartir por inversor y luego entre sus picks
                total_capital_to_invert = current_capital
                base_per_investor = total_capital_to_invert / actual_num_investors if actual_num_investors > 0 else 0
                # Agrupar posiciones válidas por inversor
                by_inv = {}
                for vp in valid_positions:
                    by_inv.setdefault(vp['investor'], []).append(vp)
                valid_investors = list(by_inv.keys())
                selected_valid_investors = [inv for inv in valid_investors if inv in selected_investors]
                failed_investors = [inv for inv in selected_investors if inv not in valid_investors]
                n_failed = len(failed_investors)
                n_valid = len(valid_investors)
                n_selected_valid = len(selected_valid_investors)
                redistribution = 0.0
                if n_failed > 0:
                    if REDISTRIBUTE_FAILED and n_valid > 0:
                        # Total capital should be distributed equally among ALL valid investors
                        total_available_capital = current_capital
                        equal_allocation_per_investor = total_available_capital / n_valid
                        print(f"    🔄 REBALANCEO: Redistribuyendo ${total_available_capital:,.2f} de capital total entre {n_valid} inversores válidos")
                        invested_total = total_available_capital
                        print(f"    💰 Capital total a invertir: ${invested_total:,.2f}")
                        
                        for inv in valid_investors:
                            cap_inv = equal_allocation_per_investor
                            if cap_inv <= 0:
                                continue
                            picks = by_inv[inv]
                            per_pick = cap_inv / len(picks) if len(picks) > 0 else 0
                            for p in picks:
                                ticker = p['ticker']
                                buy_price = p['price']
                                shares_to_buy = per_pick / buy_price if buy_price > 0 else 0
                                purchase_value = shares_to_buy * buy_price
                                portfolio[(ticker, inv)] = {
                                    'shares': shares_to_buy,
                                    'investor': inv,
                                    'buy_price': buy_price
                                }
                                transactions.append({
                                    'fecha': buy_date,
                                    'accion': 'COMPRA',
                                    'ticker': ticker,
                                    'precio': buy_price,
                                    'shares': shares_to_buy,
                                    'valor': purchase_value,
                                    'inversionista': inv
                                })
                                print(f"    📥 {inv}: {shares_to_buy:.4f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f}")
                    else:
                        print(f"    💵 {n_failed} inversores sin posición; su capital queda en efectivo (sin redistribuir)")
                        invested_total = base_per_investor * n_selected_valid
                        print(f"    💰 Capital total a invertir: ${invested_total:,.2f}")
                        
                        for inv in valid_investors:
                            cap_inv = base_per_investor if inv in selected_investors else 0.0
                            if cap_inv <= 0:
                                continue
                            picks = by_inv[inv]
                            per_pick = cap_inv / len(picks) if len(picks) > 0 else 0
                            for p in picks:
                                ticker = p['ticker']
                                buy_price = p['price']
                                shares_to_buy = per_pick / buy_price if buy_price > 0 else 0
                                purchase_value = shares_to_buy * buy_price
                                portfolio[(ticker, inv)] = {
                                    'shares': shares_to_buy,
                                    'investor': inv,
                                    'buy_price': buy_price
                                }
                                transactions.append({
                                    'fecha': buy_date,
                                    'accion': 'COMPRA',
                                    'ticker': ticker,
                                    'precio': buy_price,
                                    'shares': shares_to_buy,
                                    'valor': purchase_value,
                                    'inversionista': inv
                                })
                                print(f"    📥 {inv}: {shares_to_buy:.4f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f}")
                else:
                    # No failed investors, use normal allocation
                    invested_total = base_per_investor * n_valid
                    print(f"    💰 Capital total a invertir: ${invested_total:,.2f}")
                    
                    for inv in valid_investors:
                        cap_inv = base_per_investor
                        if cap_inv <= 0:
                            continue
                        picks = by_inv[inv]
                        per_pick = cap_inv / len(picks) if len(picks) > 0 else 0
                        for p in picks:
                            ticker = p['ticker']
                            buy_price = p['price']
                            shares_to_buy = per_pick / buy_price if buy_price > 0 else 0
                            purchase_value = shares_to_buy * buy_price
                            portfolio[(ticker, inv)] = {
                                'shares': shares_to_buy,
                                'investor': inv,
                                'buy_price': buy_price
                            }
                            transactions.append({
                                'fecha': buy_date,
                                'accion': 'COMPRA',
                                'ticker': ticker,
                                'precio': buy_price,
                                'shares': shares_to_buy,
                                'valor': purchase_value,
                                'inversionista': inv
                            })
                            print(f"    📥 {inv}: {shares_to_buy:.4f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f}")
            else:
                # SIN REBALANCEO: cada inversor invierte su capital, dividido entre sus picks
                print(f"    💰 SIN REBALANCEO: Cada inversor invierte su capital individual dividido entre sus picks")
                for investor in selected_investors:
                    capital = investor_capitals.get(investor, 0)
                    print(f"      - {investor}: ${capital:,.2f} disponible")
                # Agrupar posiciones por inversor
                by_inv = {}
                for vp in valid_positions:
                    by_inv.setdefault(vp['investor'], []).append(vp)
                total_invested = 0.0
                investors_with_positions = 0
                for inv, picks in by_inv.items():
                    investor_capital = investor_capitals.get(inv, 0)
                    if investor_capital > 0 and len(picks) > 0:
                        per_pick = investor_capital / len(picks)
                        for p in picks:
                            ticker = p['ticker']
                            buy_price = p['price']
                            shares_to_buy = per_pick / buy_price if buy_price > 0 else 0
                            purchase_value = shares_to_buy * buy_price
                            investor_capitals[inv] = 0  # capital ahora en acciones
                            portfolio[(ticker, inv)] = {
                                'shares': shares_to_buy,
                                'investor': inv,
                                'buy_price': buy_price
                            }
                            transactions.append({
                                'fecha': buy_date,
                                'accion': 'COMPRA',
                                'ticker': ticker,
                                'precio': buy_price,
                                'shares': shares_to_buy,
                                'valor': purchase_value,
                                'inversionista': inv
                            })
                            total_invested += purchase_value
                            print(f"    📥 {inv}: {shares_to_buy:.4f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f}")
                        investors_with_positions += 1
                # Manejar inversores sin posiciones
                uninvested_capital = 0.0
                if REDISTRIBUTE_FAILED and len(portfolio) > 0:
                    # Redistribuir efectivo remanente entre TODAS las posiciones compradas
                    rem = 0.0
                    for investor in selected_investors:
                        if investor not in by_inv:
                            rem += investor_capitals.get(investor, 0.0)
                            investor_capitals[investor] = 0.0
                    if rem > 0:
                        num_positions = len(portfolio)
                        extra_per_pos = rem / num_positions
                        print(f"    🔄 Redistribuyendo efectivo ${rem:,.2f} entre {num_positions} posiciones compradas")
                        for (ticker, inv), position in portfolio.items():
                            extra_shares = extra_per_pos / position['buy_price']
                            position['shares'] += extra_shares
                            transactions.append({
                                'fecha': buy_date,
                                'accion': 'COMPRA_REDISTRIBUIDA',
                                'ticker': ticker,
                                'precio': position['buy_price'],
                                'shares': extra_shares,
                                'valor': extra_per_pos,
                                'inversionista': f"{inv} (redistribuido)"
                            })
                            total_invested += extra_per_pos
                    current_capital = total_invested
                    print(f"    💰 Total invertido: ${total_invested:,.2f}")
                    print(f"    👥 Inversores con posiciones: {investors_with_positions}/{actual_num_investors}")
                else:
                    # Mantener en efectivo el capital de inversores sin posición
                    for investor in selected_investors:
                        if investor not in by_inv:
                            uninvested_capital += investor_capitals.get(investor, 0.0)
                    print(f"    💰 Total invertido: ${total_invested:,.2f}")
                    print(f"    💵 Total sin invertir: ${uninvested_capital:,.2f}")
                    print(f"    👥 Inversores con posiciones: {investors_with_positions}/{actual_num_investors}")
                    current_capital = total_invested + uninvested_capital
        else:
            print(f"    ❌ No se pudieron comprar acciones para ningún inversor en {buy_date}")
        
        # Avanzar al siguiente año
        current_year += 1
        buy_year += 1
    
    # ===== VENTA FINAL DE LAS POSICIONES end_year =====
    # Vender las posiciones compradas en enero end_year usando el precio más reciente del CSV
    if portfolio:
        print(f"\n🏁 Venta final de posiciones {end_year}")
        if has_relocation:
            total_final_value = 0
            for (ticker, inv), position in portfolio.items():
                final_price = get_latest_price_for_ticker(ticker)
                if final_price is not None:
                    final_value = position['shares'] * final_price
                    total_final_value += final_value
                    transactions.append({
                        'fecha': f'{end_year}-ACTUAL',
                        'accion': 'VENTA_FINAL',
                        'ticker': ticker,
                        'precio': final_price,
                        'shares': position['shares'],
                        'valor': final_value,
                        'inversionista': inv
                    })
                    print(f"    📤 {ticker}: {position['shares']:.2f} acciones × ${final_price:.2f} = ${final_value:,.2f} ({inv})")
                else:
                    print(f"    ⚠️ No se pudo obtener precio actual para {ticker} desde CSV")
            current_capital = total_final_value
        else:
            for (ticker, inv), position in portfolio.items():
                final_price = get_latest_price_for_ticker(ticker)
                if final_price is not None:
                    final_value = position['shares'] * final_price
                    investor_capitals[inv] = final_value
                    transactions.append({
                        'fecha': f'{end_year}-ACTUAL',
                        'accion': 'VENTA_FINAL',
                        'ticker': ticker,
                        'precio': final_price,
                        'shares': position['shares'],
                        'valor': final_value,
                        'inversionista': inv
                    })
                    print(f"    📤 {inv}: {ticker} - {position['shares']:.2f} acciones × ${final_price:.2f} = ${final_value:,.2f}")
                else:
                    print(f"    ⚠️ No se pudo obtener precio actual para {ticker} desde CSV")
            current_capital = sum(investor_capitals.values())
        print(f"  💰 Valor final del portfolio: ${current_capital:,.2f}")
    
    # Registrar performance del último año si no se añadió (porque la venta final ocurre fuera del bucle)
    if yearly_performance:
        recorded_years = {r['year'] for r in yearly_performance}
        last_invest_year = end_year - 1
        if last_invest_year not in recorded_years and last_invest_year in start_capital_by_year:
            start_cap = start_capital_by_year.get(last_invest_year)
            sell_date = f"{last_invest_year}-12-31"
            bench_buy = bench_buy_by_year.get(last_invest_year)
            bench_sell = get_daily_price_for_date(BENCHMARK_TICKER, sell_date, mode='on_or_before') or get_stock_price_for_date(BENCHMARK_TICKER, sell_date, mode='on_or_before')
            port_ret = (current_capital / start_cap) - 1 if start_cap and start_cap > 0 else None
            bench_ret = ((bench_sell - bench_buy) / bench_buy) if bench_buy and bench_sell and bench_buy > 0 else None
            alpha = (port_ret - bench_ret) if port_ret is not None and bench_ret is not None else None
            yearly_performance.append({
                'year': last_invest_year,
                'portfolio_return': None if port_ret is None else round(port_ret,6),
                'benchmark_return': None if bench_ret is None else round(bench_ret,6),
                'alpha': None if alpha is None else round(alpha,6)
            })

    # ===== RESULTADOS =====
    final_portfolio_value = current_capital

    # ===== RESULTADOS =====
    total_return = final_portfolio_value - initial_capital
    return_percentage = (total_return / initial_capital) * 100

    print(f"\n🎯 RESULTADOS DEL BACKTEST")
    print(f"💰 Capital inicial: ${initial_capital:,.2f}")
    print(f"💰 Capital final: ${final_portfolio_value:,.2f}")
    print(f"📈 Ganancia/Pérdida: ${total_return:,.2f}")
    print(f"📊 Retorno: {return_percentage:.2f}%")

    # Resumen del benchmark (S&P 500 vía SPY) en el mismo periodo
    bench_growth = 1.0
    if yearly_performance:
        for rec in sorted(yearly_performance, key=lambda r: r['year']):
            bret = rec.get('benchmark_return')
            if bret is not None:
                bench_growth *= (1.0 + bret)
    # Incluir tramo 2025 hasta el último precio disponible (para igualar el horizonte del portfolio)
    if 2025 in bench_buy_by_year:
        bench_buy_2025 = bench_buy_by_year.get(2025)
        bench_sell_latest = get_latest_price_for_ticker(BENCHMARK_TICKER)
        if bench_buy_2025 and bench_sell_latest and bench_buy_2025 > 0:
            bench_growth *= (bench_sell_latest / bench_buy_2025)

    benchmark_final_value = initial_capital * bench_growth
    benchmark_total_return = benchmark_final_value - initial_capital
    benchmark_return_percentage = (benchmark_total_return / initial_capital) * 100.0

    print("\n🆚 S&P 500 (proxy SPY) en el mismo periodo")
    print(f"💰 Capital final SPY: ${benchmark_final_value:,.2f}")
    print(f"📈 Ganancia/Pérdida SPY: ${benchmark_total_return:,.2f}")
    print(f"📊 Retorno SPY: {benchmark_return_percentage:.2f}%")

    # Comparación por CAGR en vez de alpha (si SHOW_ALPHA es False)
    if not SHOW_ALPHA and yearly_performance:
        years_span = (max(r['year'] for r in yearly_performance) - min(r['year'] for r in yearly_performance)) + 1
        port_growth = final_portfolio_value / initial_capital if initial_capital > 0 else None
        bench_growth_total = benchmark_final_value / initial_capital if initial_capital > 0 else None
        port_cagr = (port_growth ** (1/years_span) - 1) if port_growth and port_growth > 0 else None
        bench_cagr = (bench_growth_total ** (1/years_span) - 1) if bench_growth_total and bench_growth_total > 0 else None
        if port_cagr is not None and bench_cagr is not None:
            diff = port_cagr - bench_cagr
            print(f"\n📈 CAGR Portafolio ({years_span}y): {port_cagr*100:.2f}% | CAGR SPY: {bench_cagr*100:.2f}% | Spread: {diff*100:.2f}%")
        else:
            print("\n⚠️ No se pudo calcular CAGR comparativo.")

    # Calcular YTD (Year-to-Date) para 2025
    ytd_start_date = "2025-01-01"
    ytd_portfolio_start = start_capital_by_year.get(2025, current_capital)
    ytd_portfolio_return = ((current_capital - ytd_portfolio_start) / ytd_portfolio_start * 100) if ytd_portfolio_start > 0 else 0.0
    
    # YTD para S&P 500 usando precios diarios
    ytd_spy_start = get_daily_price_for_date(BENCHMARK_TICKER, ytd_start_date, mode='on_or_after')
    ytd_spy_current = get_latest_daily_price_for_ticker(BENCHMARK_TICKER)
    ytd_spy_return = ((ytd_spy_current - ytd_spy_start) / ytd_spy_start * 100) if ytd_spy_start and ytd_spy_current and ytd_spy_start > 0 else 0.0
    
    # YTD para NASDAQ (QQQ) usando precios diarios
    ytd_nasdaq_start = get_daily_price_for_date('QQQ', ytd_start_date, mode='on_or_after')
    ytd_nasdaq_current = get_latest_daily_price_for_ticker('QQQ')
    ytd_nasdaq_return = ((ytd_nasdaq_current - ytd_nasdaq_start) / ytd_nasdaq_start * 100) if ytd_nasdaq_start and ytd_nasdaq_current and ytd_nasdaq_start > 0 else 0.0

    print("\n📅 Rendimiento YTD (2025)")
    print(f"🎯 Portfolio Mauricio: {ytd_portfolio_return:.2f}%")
    print(f"📊 S&P 500 (SPY): {ytd_spy_return:.2f}%")
    print(f"💻 NASDAQ (QQQ): {ytd_nasdaq_return:.2f}%")

    # Resumen anual de alpha
    if SHOW_ALPHA and yearly_performance:
        df_alpha = pd.DataFrame(yearly_performance)
        df_alpha.sort_values('year', inplace=True)
        pr = df_alpha['portfolio_return'].fillna(0.0)
        br = df_alpha['benchmark_return'].fillna(0.0)
        df_alpha['cum_port'] = (1.0 + pr).cumprod()
        df_alpha['cum_bench'] = (1.0 + br).cumprod()
        df_alpha['cum_alpha'] = (df_alpha['cum_port'] / df_alpha['cum_bench']) - 1.0
        print("\n📅 Alpha anual (portafolio vs S&P 500)")
        for _, r in df_alpha.iterrows():
            y = int(r['year'])
            pr = r['portfolio_return']
            br = r['benchmark_return']
            al = r['alpha']
            cal = r['cum_alpha']
            pr_s = 'N/A' if pd.isna(pr) or pr is None else f"{pr*100:.2f}%"
            br_s = 'N/A' if pd.isna(br) or br is None else f"{br*100:.2f}%"
            al_s = 'N/A' if pd.isna(al) or al is None else f"{al*100:.2f}%"
            cal_s = 'N/A' if pd.isna(cal) or cal is None else f"{cal*100:.2f}%"
            print(f"  {y}: Mauricio {pr_s} | SPX {br_s} | Alpha {al_s} | CumAlpha {cal_s}")
        df_alpha.to_csv('yearly_alpha.csv', index=False)
        final_cum_alpha = df_alpha['cum_alpha'].iloc[-1]
        final_cum_alpha_s = 'N/A' if pd.isna(final_cum_alpha) else f"{final_cum_alpha*100:.2f}%"
        print(f"🔢 Alpha acumulada del periodo: {final_cum_alpha_s}")
        print("💾 Alpha anual guardado en yearly_alpha.csv")
    
    # Guardar transacciones
    df_transactions = pd.DataFrame(transactions)
    df_transactions.to_csv("backtest_transactions.csv", index=False)
    print(f"💾 Transacciones guardadas en backtest_transactions.csv")
    
    # Generar gráfico de comparación
    if generate_chart:
        create_performance_comparison_graph(yearly_performance, initial_capital, final_portfolio_value, 
                                          benchmark_final_value, start_year, transactions=transactions)
    
    return {
        'capital_inicial': initial_capital,
        'capital_final': final_portfolio_value,
        'ganancia': total_return,
        'retorno_pct': return_percentage,
        'benchmark_capital_final': benchmark_final_value,
        'benchmark_retorno_pct': benchmark_return_percentage,
        'ytd_portfolio_return': ytd_portfolio_return,
        'ytd_spy_return': ytd_spy_return,
        'ytd_nasdaq_return': ytd_nasdaq_return,
        'transacciones': transactions
    }

def find_replacement_positions(year, exclude_investors, needed, min_top1_pct=None, quarter='Q3'):
    """Busca hasta 'needed' inversores alternativos para el quarter especificado de 'year'.
    - Excluye los nombres en exclude_investors.
    - Aplica el umbral min_top1_pct si se define.
    - Devuelve un DataFrame ordenado por porcentaje_portafolio desc con columnas del CSV original.
    """
    if needed <= 0:
        return pd.DataFrame()

    df = pd.read_csv("top_positions.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'porcentaje_portafolio' in df.columns:
            df['porcentaje_portafolio'] = pd.to_numeric(df['porcentaje_portafolio'], errors='coerce')

    pool = df[(df['anio'] == year) & (df['quarter'] == quarter)]
    if min_top1_pct is not None and 'porcentaje_portafolio' in pool.columns:
        pool = pool[pool['porcentaje_portafolio'] >= float(min_top1_pct)]

    if exclude_investors:
        pool = pool[~pool['inversionista'].isin(list(exclude_investors))]

    if 'porcentaje_portafolio' in pool.columns:
        pool = pool.sort_values('porcentaje_portafolio', ascending=False)

    return pool.head(int(needed)).reset_index(drop=True)

def generate_random_stock_baseline(start_year, end_year, initial_capital, date_index):
    """
    Genera una curva de baseline de selección aleatoria considerando TODOS los inversores y stocks disponibles.
    Para cada año, selecciona aleatoriamente NUM_INVESTORS stocks de TODOS los disponibles en Q3.
    """
    try:
        np.random.seed(RANDOM_SEED if 'RANDOM_SEED' in globals() else 42)
        random.seed(RANDOM_SEED if 'RANDOM_SEED' in globals() else 42)
        
        # Leer todas las posiciones disponibles
        df = pd.read_csv("top_positions_all_clean.csv") if os.path.exists("top_positions_all_clean.csv") else pd.read_csv("top_positions.csv")
        
        curve = pd.Series(initial_capital, index=date_index)
        current_capital = initial_capital
        
        print(f"🎲 Generando baseline aleatorio: selección random de {NUM_INVESTORS} stocks de TODOS los inversores disponibles")
        
        # Helper function to load daily series (copied from main graph function)
        def load_daily_series(ticker: str) -> pd.Series:
            path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
            if os.path.exists(path):
                df_daily = pd.read_csv(path)
                if not df_daily.empty and 'fecha' in df_daily.columns and 'precio_cierre' in df_daily.columns:
                    s = pd.to_datetime(df_daily['fecha'])
                    v = pd.to_numeric(df_daily['precio_cierre'], errors='coerce')
                    return pd.Series(data=v.values, index=s, name=ticker).dropna().sort_index()
            # Fallback to quarterly if no daily data
            path_q = os.path.join("stock_prices", f"{ticker}_quarterly_prices.csv")
            if os.path.exists(path_q):
                df_q = pd.read_csv(path_q)
                if not df_q.empty and 'fecha' in df_q.columns and 'precio_cierre' in df_q.columns:
                    s = pd.to_datetime(df_q['fecha'])
                    v = pd.to_numeric(df_q['precio_cierre'], errors='coerce')
                    return pd.Series(data=v.values, index=s, name=ticker).dropna().sort_index()
            return pd.Series(dtype=float)

        for year in range(start_year, end_year + 1):
            buy_year = year - 1
            buy_date = _quarter_end_date(year, BUY_QUARTER)
            sell_date = _quarter_end_date(year, SELL_QUARTER)
            
            # Obtener TODAS las posiciones del DATA_SOURCE_QUARTER disponibles del año anterior (todos los inversores)
            positions = df[(df['anio'] == buy_year) & (df['quarter'] == DATA_SOURCE_QUARTER)]
            
            if positions.empty:
                print(f"     Año {year}: Sin posiciones {DATA_SOURCE_QUARTER} en {buy_year}")
                continue
            
            # Contar inversores únicos disponibles
            total_investors = positions['inversionista'].nunique()
            total_positions = len(positions)
            
            # Seleccionar aleatoriamente NUM_INVESTORS posiciones de TODAS las disponibles
            if len(positions) > NUM_INVESTORS:
                selected_positions = positions.sample(n=NUM_INVESTORS, random_state=RANDOM_SEED + year)
            else:
                selected_positions = positions
                
            print(f"     Año {year}: {total_investors} inversores, {total_positions} posiciones -> seleccionadas {len(selected_positions)}")
            
            # Build portfolio using actual daily stock prices instead of straight line interpolation
            year_start = pd.Timestamp(_quarter_end_date(year, BUY_QUARTER))
            year_end = pd.Timestamp(_quarter_end_date(year, SELL_QUARTER))
            year_mask = (date_index >= year_start) & (date_index <= year_end)
            
            if year_mask.any():
                year_dates = date_index[year_mask]
                daily_sum = pd.Series(0.0, index=year_dates)
                start_capital = curve.loc[year_dates[0]] if year > start_year else initial_capital
                
                # Calculate how much to invest in each stock (equal weight)
                equal_weight = start_capital / NUM_INVESTORS
                valid_stocks = 0
                
                for _, position in selected_positions.iterrows():
                    ticker = position['accion']
                    investor = position['inversionista']
                    
                    # Get buy price for number of shares calculation
                    buy_price = get_daily_price_for_date(ticker, buy_date, mode='on_or_after')
                    buy_source = "daily"
                    if not buy_price:
                        buy_price = get_stock_price_for_date(ticker, buy_date, mode='on_or_before')
                        buy_source = "quarterly"
                    
                    if buy_price and buy_price > 0:
                        # Calculate shares to buy with equal weight
                        shares = equal_weight / buy_price
                        
                        # Load daily price series for this stock
                        ts = load_daily_series(ticker)
                        if not ts.empty:
                            # Filter to year dates and align with our index
                            ts_year = ts[(ts.index >= year_start) & (ts.index <= year_end)]
                            if not ts_year.empty:
                                # Add this stock's daily value to the portfolio
                                for date in year_dates:
                                    if date in ts_year.index:
                                        daily_sum.loc[date] += ts_year.loc[date] * shares
                                    elif len(ts_year) > 0:
                                        # Use forward fill if no exact date match
                                        closest_price = ts_year[ts_year.index <= date]
                                        if len(closest_price) > 0:
                                            daily_sum.loc[date] += closest_price.iloc[-1] * shares
                                        else:
                                            # Use first available price if no prior price
                                            daily_sum.loc[date] += ts_year.iloc[0] * shares
                                
                                valid_stocks += 1
                                # Calculate return for logging
                                sell_price = get_daily_price_for_date(ticker, sell_date, mode='on_or_before')
                                sell_source = "daily"
                                if not sell_price:
                                    sell_price = get_stock_price_for_date(ticker, sell_date, mode='on_or_before')
                                    sell_source = "quarterly"
                                
                                if sell_price:
                                    stock_return = sell_price / buy_price
                                    print(f"       {ticker} ({investor}): ${buy_price:.2f}({buy_source}) -> ${sell_price:.2f}({sell_source}) = {((stock_return-1)*100):.1f}%")
                
                if valid_stocks > 0:
                    # Update the curve with actual daily values
                    curve.loc[year_dates] = daily_sum.values
                    current_capital = daily_sum.iloc[-1] if len(daily_sum) > 0 else start_capital
                    year_return = ((current_capital / start_capital) - 1) * 100
                    print(f"     Año {year}: {valid_stocks}/{len(selected_positions)} stocks válidos, retorno: {year_return:.1f}%, capital: ${current_capital:,.0f}")
                else:
                    print(f"     Año {year}: Sin stocks válidos")
                    # Keep the same capital if no valid stocks
                    curve.loc[year_dates] = start_capital
        
        print(f"🎲 Baseline aleatorio completado: capital final ${current_capital:,.0f}")
        return curve
        
    except Exception as e:
        print(f"⚠️ Error en generate_random_stock_baseline: {e}")
        return pd.Series(initial_capital, index=date_index)

def create_performance_comparison_graph(yearly_performance, initial_capital, final_portfolio_value, 
                                       benchmark_final_value, start_year=2015, transactions=None):
    """
    Genera un gráfico comparando Portfolio vs S&P 500 (SPY) vs NASDAQ (QQQ) usando precios DIARIOS
    de la carpeta stock_prices_daily. Si no hay diarios, cae a quarterly.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        def load_daily_series(ticker: str) -> pd.Series:
            path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty and 'fecha' in df.columns and 'precio_cierre' in df.columns:
                    s = pd.to_datetime(df['fecha'])
                    v = pd.to_numeric(df['precio_cierre'], errors='coerce')
                    ser = pd.Series(v.values, index=s)
                    return ser.sort_index()
            # fallback quarterly
            path_q = os.path.join("stock_prices", f"{ticker}_quarterly_prices.csv")
            if os.path.exists(path_q):
                dfq = pd.read_csv(path_q)
                if not dfq.empty and 'fecha' in dfq.columns and 'precio_cierre' in dfq.columns:
                    s = pd.to_datetime(dfq['fecha'])
                    v = pd.to_numeric(dfq['precio_cierre'], errors='coerce')
                    ser = pd.Series(v.values, index=s)
                    return ser.sort_index()
            return pd.Series(dtype=float)
        
        # Series diarias para SPY y QQQ
        spy_ser = load_daily_series('SPY')
        qqq_ser = load_daily_series('QQQ')
        if spy_ser.empty or qqq_ser.empty:
            print("⚠️ No se encontraron series diarias para SPY/QQQ; se usará el gráfico anual como fallback.")
            # Fallback al comportamiento anual anterior
            years = []
            portfolio_values = [initial_capital]
            spy_values = [initial_capital]
            nasdaq_values = [initial_capital]
            portfolio_cumulative = 1.0
            spy_cumulative = 1.0
            nasdaq_cumulative = 1.0
            for year_data in sorted(yearly_performance, key=lambda x: x['year']):
                year = year_data['year']
                years.append(year)
                port_ret = year_data.get('portfolio_return', 0.0) or 0.0
                spy_ret = year_data.get('benchmark_return', 0.0) or 0.0
                nas_ret = get_yearly_return('QQQ', year) or 0.0
                portfolio_cumulative *= (1.0 + port_ret)
                spy_cumulative *= (1.0 + spy_ret)
                nasdaq_cumulative *= (1.0 + nas_ret)
                portfolio_values.append(initial_capital * portfolio_cumulative)
                spy_values.append(initial_capital * spy_cumulative)
                nasdaq_values.append(initial_capital * nasdaq_cumulative)
            years.append(2025)
            portfolio_values.append(final_portfolio_value)
            # 2025 YTD con diarios si hay
            ytd_spy_start = get_daily_price_for_date('SPY', '2025-01-01', mode='on_or_after')
            ytd_spy_curr = get_latest_daily_price_for_ticker('SPY')
            ytd_spy_ret = ((ytd_spy_curr - ytd_spy_start) / ytd_spy_start) if ytd_spy_start and ytd_spy_curr and ytd_spy_start>0 else 0.0
            ytd_qqq_start = get_daily_price_for_date('QQQ', '2025-01-01', mode='on_or_after')
            ytd_qqq_curr = get_latest_daily_price_for_ticker('QQQ')
            ytd_qqq_ret = ((ytd_qqq_curr - ytd_qqq_start) / ytd_qqq_start) if ytd_qqq_start and ytd_qqq_curr and ytd_qqq_start>0 else 0.0
            spy_values.append(initial_capital * spy_cumulative * (1.0 + ytd_spy_ret))
            nasdaq_values.append(initial_capital * nasdaq_cumulative * (1.0 + ytd_qqq_ret))
            years_with_start = [start_year - 1] + years
            plt.figure(figsize=(12,8))
            plt.plot(years_with_start, portfolio_values, label='Portfolio Mauricio')
            plt.plot(years_with_start, spy_values, label='S&P 500 (SPY)')
            plt.plot(years_with_start, nasdaq_values, label='NASDAQ (QQQ)')
            plt.title('Comparación de Rendimiento (anual, fallback)')
            plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
            print("\n📊 Gráfico (fallback anual) guardado como 'performance_comparison.png'")
            return
        
        # Rango de fechas
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = min(spy_ser.index.max(), qqq_ser.index.max())
        spy_ser = spy_ser[(spy_ser.index>=start_date) & (spy_ser.index<=end_date)]
        qqq_ser = qqq_ser[(qqq_ser.index>=start_date) & (qqq_ser.index<=end_date)]
        
        # Construir curva diaria del portfolio a partir de transacciones
        port_curve = pd.Series(0.0, index=spy_ser.index)
        if transactions:
            # Agrupar compras por año
            buys = [t for t in transactions if t.get('accion') in ('COMPRA','COMPRA_REDISTRIBUIDA')]
            # Ventas estándar al 31-12, y ventas finales 2025-ACTUAL
            sells = [t for t in transactions if t.get('accion') in ('VENTA','VENTA_FINAL')]
            # Construir periodos por año
            for year in range(start_year, 2026):
                period_start = pd.Timestamp(_quarter_end_date(year, BUY_QUARTER))
                if year < 2025:
                    period_end = pd.Timestamp(_quarter_end_date(year, SELL_QUARTER))
                else:
                    period_end = end_date
                # Filtrar compras del periodo (hechas en el BUY_QUARTER del año)
                buy_date_year = _quarter_end_date(year, BUY_QUARTER)
                year_buys = [b for b in buys if b.get('fecha') == buy_date_year]
                if not year_buys:
                    continue
                # Cargar series de cada ticker y sumar
                daily_sum = pd.Series(0.0, index=port_curve.loc[(port_curve.index>=period_start)&(port_curve.index<=period_end)].index)
                # Sumar por ticker la cantidad total de acciones compradas en el año
                shares_by_ticker = {}
                for b in year_buys:
                    ticker = b.get('ticker')
                    shares_by_ticker[ticker] = shares_by_ticker.get(ticker, 0.0) + float(b.get('shares',0.0))
                for ticker, shares in shares_by_ticker.items():
                    ts = load_daily_series(ticker)
                    if ts.empty:
                        continue
                    ts = ts[(ts.index>=period_start) & (ts.index<=period_end)]
                    if ts.empty:
                        continue
                    aligned = daily_sum.copy()
                    aligned.loc[ts.index] = aligned.loc[ts.index] + (ts * shares).values
                    daily_sum = aligned
                # Asignar al tramo
                port_curve.loc[daily_sum.index] = daily_sum.values
        
        # Normalizar SPY/QQQ a capital inicial
        if not spy_ser.empty and spy_ser.iloc[0] > 0:
            spy_curve = initial_capital * (spy_ser / spy_ser.iloc[0])
        else:
            spy_curve = pd.Series(index=port_curve.index, dtype=float)
        if not qqq_ser.empty and qqq_ser.iloc[0] > 0:
            qqq_curve = initial_capital * (qqq_ser / qqq_ser.iloc[0])
        else:
            qqq_curve = pd.Series(index=port_curve.index, dtype=float)
        
        # Si el portafolio quedó vacío en algún tramo (sin datos), hacer forward-fill con último valor conocido
        if (port_curve == 0).all():
            # Si no se pudo construir, caer a línea simple con valor final
            port_curve[:] = initial_capital
        port_curve = port_curve.replace(0, pd.NA).ffill().fillna(initial_capital)
        
        # Helper para construir curva diaria a partir de transacciones (reutilizable para baseline)
        def _build_curve_from_transactions(transacciones, index, initial_cap):
            curve = pd.Series(0.0, index=index)
            if not transacciones:
                curve[:] = initial_cap
                return curve
            buys = [t for t in transacciones if t.get('accion') in ('COMPRA','COMPRA_REDISTRIBUIDA')]
            for year in range(start_year, END_YEAR+1):
                period_start = pd.Timestamp(_quarter_end_date(year, BUY_QUARTER))
                period_end = pd.Timestamp(_quarter_end_date(year, SELL_QUARTER)) if year < END_YEAR else index.max()
                buy_date_year = _quarter_end_date(year, BUY_QUARTER)
                year_buys = [b for b in buys if b.get('fecha') == buy_date_year]
                if not year_buys:
                    continue
                daily_sum = pd.Series(0.0, index=curve.loc[(curve.index>=period_start)&(curve.index<=period_end)].index)
                shares_by_ticker = {}
                for b in year_buys:
                    ticker = b.get('ticker')
                    shares_by_ticker[ticker] = shares_by_ticker.get(ticker, 0.0) + float(b.get('shares',0.0))
                for ticker, shares in shares_by_ticker.items():
                    ts = load_daily_series(ticker)
                    if ts.empty:
                        continue
                    ts = ts[(ts.index>=period_start) & (ts.index<=period_end)]
                    if ts.empty:
                        continue
                    aligned = daily_sum.copy()
                    aligned.loc[ts.index] = aligned.loc[ts.index] + (ts * shares).values
                    daily_sum = aligned
                curve.loc[daily_sum.index] = daily_sum.values
            # rellenar huecos
            curve = curve.replace(0, pd.NA).ffill().fillna(initial_cap)
            return curve

        # Curva baseline aleatoria (opcional) - selección aleatoria real de stocks
        rand_curve = None
        if ALWAYS_INCLUDE_RANDOM_BASELINE:
            try:
                # Generar curva de selección aleatoria real considerando todos los inversores y stocks
                rand_curve = generate_random_stock_baseline(start_year, END_YEAR, initial_capital, port_curve.index)
            except Exception as e_baseline:
                print(f"⚠️ Error creando baseline aleatorio: {e_baseline}")
                rand_curve = None
        # Curva Antonio sintética - comenzar desde el valor del S&P en BRANCH_YEAR
        antonio_curve = None
        if INCLUDE_ANTONIO_LINE:
            try:
                print(f"🎨 Generando curva Antonio desde BRANCH_YEAR {BRANCH_YEAR}")
                
                # Encontrar el valor del S&P al inicio del BRANCH_YEAR para comenzar desde ahí
                branch_start = pd.Timestamp(f"{BRANCH_YEAR}-01-01")
                sp500_branch_value = initial_capital
                
                if branch_start in spy_curve.index:
                    sp500_branch_value = spy_curve.loc[branch_start]
                elif (spy_curve.index <= branch_start).any():
                    # Tomar el valor más cercano antes del branch
                    sp500_branch_value = spy_curve.loc[spy_curve.index <= branch_start].iloc[-1]
                
                print(f"   Valor del S&P en {BRANCH_YEAR}: ${sp500_branch_value:,.0f}")
                
                # Crear curva completa empezando con valores del S&P
                antonio_curve = port_curve.copy()
                
                # Calcular rendimientos año por año desde BRANCH_YEAR usando Antonio's strategy
                cap = sp500_branch_value
                
                for year in range(BRANCH_YEAR, END_YEAR + 1):
                    ticker = ANTONIO_YEAR_TICKER_MAP.get(year)
                    if not ticker:
                        # Si no hay ticker para el año, usar el último ticker conocido o META como default
                        available_years = [y for y in ANTONIO_YEAR_TICKER_MAP.keys() if y <= year]
                        if available_years:
                            ticker = ANTONIO_YEAR_TICKER_MAP[max(available_years)]
                        else:
                            ticker = 'META'  # Default fallback
                    
                    print(f"   Año {year}: {ticker}")
                    
                    # Usar precios diarios cuando sea posible
                    start_price = get_daily_price_for_date(ticker, f"{year}-01-01", mode='on_or_after') 
                    if not start_price:
                        start_price = get_stock_price_for_date(ticker, f"{year}-01-01", mode='on_or_before')
                    
                    end_price = get_daily_price_for_date(ticker, f"{year}-12-31", mode='on_or_before')
                    if not end_price:
                        end_price = get_stock_price_for_date(ticker, f"{year}-12-31", mode='on_or_before')
                    
                    if start_price and end_price and start_price > 0:
                        year_return = (end_price / start_price) - 1
                        new_cap = cap * (1 + year_return)
                        
                        print(f"     {ticker}: ${start_price:.2f} -> ${end_price:.2f} = {year_return*100:.2f}% | Capital: ${cap:,.0f} -> ${new_cap:,.0f}")
                        
                        # Aplicar interpolación diaria durante el año usando precios diarios del ticker
                        year_start = pd.Timestamp(f"{year}-01-01")
                        year_end = pd.Timestamp(f"{year}-12-31")
                        year_mask = (antonio_curve.index >= year_start) & (antonio_curve.index <= year_end)
                        
                        if year_mask.any():
                            year_dates = antonio_curve.index[year_mask]
                            
                            # Cargar serie diaria del ticker para interpolación más precisa
                            ticker_series = None
                            try:
                                # Intentar cargar precios diarios
                                daily_path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
                                if os.path.exists(daily_path):
                                    ticker_df = pd.read_csv(daily_path)
                                    if not ticker_df.empty and 'fecha' in ticker_df.columns and 'precio_cierre' in ticker_df.columns:
                                        ticker_dates = pd.to_datetime(ticker_df['fecha'])
                                        ticker_prices = pd.to_numeric(ticker_df['precio_cierre'], errors='coerce')
                                        ticker_series = pd.Series(ticker_prices.values, index=ticker_dates)
                                        ticker_series = ticker_series.sort_index()
                                        # Filtrar para el año actual
                                        ticker_series = ticker_series[(ticker_series.index >= year_start) & (ticker_series.index <= year_end)]
                            except Exception:
                                ticker_series = None
                            
                            if ticker_series is not None and not ticker_series.empty:
                                # Usar movimientos diarios del ticker para interpolar
                                first_price = ticker_series.iloc[0] if len(ticker_series) > 0 else start_price
                                if first_price > 0:
                                    # Normalizar la serie del ticker al capital inicial del año
                                    normalized_series = cap * (ticker_series / first_price)
                                    # Alinear con las fechas de antonio_curve
                                    for date in year_dates:
                                        if date in normalized_series.index:
                                            antonio_curve.loc[date] = normalized_series.loc[date]
                                        else:
                                            # Interpolación si la fecha exacta no existe
                                            before = normalized_series[normalized_series.index <= date]
                                            after = normalized_series[normalized_series.index >= date]
                                            if not before.empty and not after.empty:
                                                antonio_curve.loc[date] = (before.iloc[-1] + after.iloc[0]) / 2
                                            elif not before.empty:
                                                antonio_curve.loc[date] = before.iloc[-1]
                                            elif not after.empty:
                                                antonio_curve.loc[date] = after.iloc[0]
                            else:
                                # Fallback: interpolación lineal simple
                                if len(year_dates) > 1:
                                    interp_values = np.linspace(cap, new_cap, len(year_dates))
                                    antonio_curve.loc[year_dates] = interp_values
                                elif len(year_dates) == 1:
                                    antonio_curve.loc[year_dates[0]] = new_cap
                        
                        cap = new_cap
                    else:
                        print(f"     {ticker}: Sin precios válidos")
                
                print(f"   ✅ Curva Antonio generada: {len(antonio_curve)} puntos, valor final: ${cap:,.0f}")
                
            except Exception as e:
                print(f"   ❌ Error generando curva Antonio: {e}")
                antonio_curve = None

        plt.figure(figsize=(12,8))
        
        # Mauricio full curve (base)
        plt.plot(port_curve.index, port_curve.values, label='Mauricio', linewidth=2, color=MAURICIO_COLOR)
        print(f"📊 Curva Mauricio: {len(port_curve)} puntos, rango ${port_curve.min():,.0f} - ${port_curve.max():,.0f}")
        
        # Random baseline (usando todos los inversores disponibles)
        if rand_curve is not None and not rand_curve.empty:
            plt.plot(rand_curve.index, rand_curve.values, label='Random Baseline (All Investors)', linestyle='--', alpha=0.8, color=RANDOM_BASELINE_COLOR, linewidth=2)
            print(f"📊 Curva Random: {len(rand_curve)} puntos, rango ${rand_curve.min():,.0f} - ${rand_curve.max():,.0f}")
        else:
            print("📊 Curva Random: No disponible")
        
        # Antonio branch: show from BRANCH_YEAR onward, starting from Mauricio's value
        if antonio_curve is not None and not antonio_curve.empty:
            # Show only from BRANCH_YEAR
            branch_start = pd.Timestamp(f"{BRANCH_YEAR}-01-01")
            ac_branch = antonio_curve[antonio_curve.index >= branch_start]
            if not ac_branch.empty:
                plt.plot(ac_branch.index, ac_branch.values, label=f'Antonio (desde {BRANCH_YEAR})', linestyle='-.', linewidth=2, color=ANTONIO_COLOR)
                print(f"📊 Curva Antonio (desde {BRANCH_YEAR}): {len(ac_branch)} puntos, rango ${ac_branch.min():,.0f} - ${ac_branch.max():,.0f}")
                
                # Marcar el punto de bifurcación
                if branch_start in port_curve.index:
                    branch_value = port_curve.loc[branch_start]
                    plt.plot(branch_start, branch_value, 'ro', markersize=8, label=f'Bifurcación ({BRANCH_YEAR})')
                    print(f"📊 Punto de bifurcación: {BRANCH_YEAR} = ${branch_value:,.0f}")
            else:
                print(f"📊 Curva Antonio: Sin datos desde {BRANCH_YEAR}")
        else:
            print("📊 Curva Antonio: No disponible")
        
        # Benchmarks
        if not spy_curve.empty:
            plt.plot(spy_curve.index, spy_curve.values, label='S&P 500 (SPY)', linewidth=1.5, alpha=0.8)
            print(f"📊 Curva SPY: {len(spy_curve)} puntos")
        if not qqq_curve.empty:
            plt.plot(qqq_curve.index, qqq_curve.values, label='NASDAQ (QQQ)', linewidth=1.5, alpha=0.8)
            print(f"📊 Curva QQQ: {len(qqq_curve)} puntos")
        plt.title('Comparación de Rendimiento (curva diaria)')
        plt.xlabel('Fecha'); plt.ylabel('Valor ($)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 Gráfico diario guardado como 'performance_comparison.png'")
    except Exception as e:
        print(f"\n❌ Error generando gráfico diario: {e}")

def get_daily_price_for_date(ticker, date_str, mode: str = 'nearest'):
    """Obtiene el precio de cierre desde CSV de precios diarios para una fecha objetivo.
    mode:
      - 'on_or_after': primer registro con fecha >= objetivo
      - 'on_or_before': último registro con fecha <= objetivo
      - 'nearest' (default): fecha más cercana al objetivo
    """
    try:
        csv_path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
        if not os.path.exists(csv_path):
            # Fallback a precios quarterly si no existen daily
            return get_stock_price_for_date(ticker, date_str, mode)
            
        df = pd.read_csv(csv_path)
        if df.empty or 'fecha' not in df.columns or 'precio_cierre' not in df.columns:
            return get_stock_price_for_date(ticker, date_str, mode)

        # Convertir fechas
        df['fecha'] = pd.to_datetime(df['fecha'])
        target_date = pd.to_datetime(date_str)

        if mode == 'on_or_after':
            candidates = df[df['fecha'] >= target_date]
            if candidates.empty:
                return get_stock_price_for_date(ticker, date_str, mode)
            row = candidates.sort_values('fecha').iloc[0]
        elif mode == 'on_or_before':
            candidates = df[df['fecha'] <= target_date]
            if candidates.empty:
                return get_stock_price_for_date(ticker, date_str, mode)
            row = candidates.sort_values('fecha').iloc[-1]
        else:  # nearest
            diffs = (df['fecha'] - target_date).abs()
            idx = diffs.idxmin()
            row = df.loc[idx]

        try:
            price = float(row['precio_cierre'])
        except Exception:
            return get_stock_price_for_date(ticker, date_str, mode)
        return round(price, 2)
    except Exception:
        return get_stock_price_for_date(ticker, date_str, mode)

def get_latest_daily_price_for_ticker(ticker: str):
    """Obtiene el último precio disponible desde el CSV de precios diarios del ticker."""
    try:
        csv_path = os.path.join("stock_prices_daily", f"{ticker}_daily_prices.csv")
        if not os.path.exists(csv_path):
            return get_latest_price_for_ticker(ticker)
            
        df = pd.read_csv(csv_path)
        if df.empty or 'precio_cierre' not in df.columns:
            return get_latest_price_for_ticker(ticker)
            
        # Tomar el último registro (asumiendo orden cronológico por archivo)
        price = df['precio_cierre'].iloc[-1]
        try:
            price = float(price)
        except Exception:
            return get_latest_price_for_ticker(ticker)
        return round(price, 2)
    except Exception:
        return get_latest_price_for_ticker(ticker)

def get_yearly_return(ticker, year):
    """
    Calcula el retorno anual de un ticker desde CSV de precios diarios
    """
    try:
        # Precio al inicio del año (31 dic año anterior o primer día hábil del año)
        start_price = get_daily_price_for_date(ticker, f"{year-1}-12-31", mode='on_or_before')
        if not start_price:
            start_price = get_daily_price_for_date(ticker, f"{year}-01-01", mode='on_or_after')
        
        # Precio al final del año (31 dic o último día hábil del año)
        end_price = get_daily_price_for_date(ticker, f"{year}-12-31", mode='on_or_before')
        
        if start_price and end_price and start_price > 0:
            return (end_price - start_price) / start_price
        return None
        
    except Exception:
        return None

def next_quarter(year: int, quarter: str) -> tuple[int, str]:
    order = ['Q1', 'Q2', 'Q3', 'Q4']
    q = quarter.upper()
    if q not in order:
        return year, 'Q4'
    idx = order.index(q)
    if idx == 3:
        return year + 1, 'Q1'
    return year, order[idx + 1]

# Eliminado: build_investor_track_record
# Ahora el cálculo y guardado del track record por inversor vive en investor_track_record.py
# Usa: compute_investor_track_record, save_investor_track_record y build_lookback_track_record

# ...existing code...
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 CONFIGURACIÓN DEL BACKTEST")
    print("=" * 60)
    print(f"📊 Número de inversores: {NUM_INVESTORS}")
    print(f"📅 Año de inicio: {START_YEAR}")
    print(f"🔄 Rebalanceo: {'Activado' if HAS_RELOCATION else 'Desactivado'}")
    print(f"💸 Redistribuir fallidos: {'Activado' if REDISTRIBUTE_FAILED else 'Desactivado (mantener en efectivo)'}")
    print(f"🎲 Semilla aleatoria: {RANDOM_SEED}")
    mode_label = {0: 'Aleatorio', 1: 'Top previo', 2: 'Valor de portafolio', 3: 'CAGR 3y', 4: 'CAGR acumulada'}.get(SELECTION_MODE, 'Aleatorio')
    print(f"🎯 Modo selección: {mode_label}")
    print(f"🎯 Top picks por inversor: {TOP_PICKS_PER_INVESTOR}")
    print(f"🔎 Filtro mínimo % Top-1: {'N/A' if MIN_TOP1_PCT is None else f'>= {MIN_TOP1_PCT}%'}")
    print(f"🔭 Ventana lookback: buy_year={LOOKBACK_BUY_YEAR}, años={LOOKBACK_YEARS}")
    print("=" * 60)
    print()
    
    # Elegir inversores según el modo seleccionado
    if SELECTION_MODE == 1:
        selected_investors, _ranking = select_investors_by_prior_performance(
            START_YEAR, NUM_INVESTORS, end_quarter=PRIOR_END_QUARTER, mode='prior_top', random_seed=RANDOM_SEED
        )
    elif SELECTION_MODE == 2:
        selected_investors, _ranking = select_investors_by_portfolio_value(
            START_YEAR, NUM_INVESTORS, quarter='Q3'
        )
    elif SELECTION_MODE == 3:
        selected_investors, _ranking = select_investors_by_cagr(
            mode=3,
            top_n=NUM_INVESTORS,
            metric=CAGR_METRIC,
            source_csv=CAGR_SOURCE_CSV,
            min_years=CAGR_MIN_YEARS,
            descending=CAGR_DESCENDING
        )
    elif SELECTION_MODE == 4:
        selected_investors, _ranking = select_investors_by_cagr(
            mode=4,
            top_n=NUM_INVESTORS,
            metric=CAGR_METRIC,
            source_csv=CAGR_SOURCE_CSV,
            min_years=CAGR_MIN_YEARS,
            descending=CAGR_DESCENDING
        )
    else:
        selected_investors, _ = select_investors_by_prior_performance(
            START_YEAR, NUM_INVESTORS, mode='random', random_seed=RANDOM_SEED
        )
    
    # Ejecutar backtest usando la configuración definida arriba
    resultado = backtest_strategy(
        num_investors=NUM_INVESTORS,
        start_year=START_YEAR,
        has_relocation=HAS_RELOCATION,
        selected_investors=selected_investors,
        min_top1_pct=MIN_TOP1_PCT,
    )
    
    # Generar track record por inversionista (trimestral y anual) y CSV de lookback configurable
    if GENERATE_INVESTOR_TRACK_RECORD:
        df_q, df_y = compute_investor_track_record()
        save_investor_track_record(df_q, df_y)
        # CSV adicional con ventana configurable (ej.: 2016-2018 si buy_year=2019 y lookback=3)
        build_lookback_track_record(buy_year=LOOKBACK_BUY_YEAR,
                                    lookback_years=LOOKBACK_YEARS,
                                    df_y=df_y,
                                    out_csv='investor_track_record_lookback.csv')
    
    print(f"\n✅ Backtest completado!")
    print(f"📄 Para cambiar la configuración, edita las variables al inicio del archivo:")
