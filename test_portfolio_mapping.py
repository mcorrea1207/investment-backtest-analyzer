import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ==============================
# Helper: read price from local CSV
# ==============================

def get_price_from_csv(ticker: str, date_str: str, mode: str = 'nearest') -> Optional[float]:
    """
    Read a price from stock_prices/{TICKER}_quarterly_prices.csv for a target date.
    - mode='on_or_after': first date >= target date
    - mode='on_or_before': last date <= target date
    - mode='nearest': whichever date is closest to target
    Returns float price or None.
    """
    path = os.path.join("stock_prices", f"{ticker}_quarterly_prices.csv")
    if not os.path.exists(path):
        print(f"⚠️ CSV no encontrado: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error leyendo {path}: {e}")
        return None

    if df.empty or 'fecha' not in df.columns or 'precio_cierre' not in df.columns:
        print(f"⚠️ CSV inválido para {ticker}: faltan columnas 'fecha' o 'precio_cierre'")
        return None

    # Parse dates
    try:
        df['fecha'] = pd.to_datetime(df['fecha'])
        target = pd.to_datetime(date_str)
    except Exception as e:
        print(f"⚠️ Error parseando fechas para {ticker}: {e}")
        return None

    if mode == 'on_or_after':
        candidates = df[df['fecha'] >= target]
        if candidates.empty:
            return None
        row = candidates.sort_values('fecha').iloc[0]
    elif mode == 'on_or_before':
        candidates = df[df['fecha'] <= target]
        if candidates.empty:
            return None
        row = candidates.sort_values('fecha').iloc[-1]
    else:  # nearest
        idx = (df['fecha'] - target).abs().idxmin()
        row = df.loc[idx]

    try:
        return float(row['precio_cierre'])
    except Exception:
        return None


# ==============================
# Backtest yearly mapping
# ==============================

def backtest_yearly_mapping(mapping: Dict[int, str], initial: float = 1_000_000.0,
                            buy_mode: str = 'on_or_after', sell_mode: str = 'on_or_before'
                            ) -> Tuple[List[Tuple[int, str, str, Optional[float], str, Optional[float], Optional[float]]], float]:
    """
    mapping: {year: ticker}
    Buys on Jan 1 (buy_mode), sells on Dec 31 (sell_mode).
    Returns (rows, final_capital) where rows = [(year, ticker, buy_date, buy, sell_date, sell, ret)]
    and ret is decimal return (e.g., 0.12 for +12%).
    """
    capital = initial
    rows = []

    for year in sorted(mapping.keys()):
        ticker = mapping[year]
        buy_date = f"{year}-01-01"
        sell_date = f"{year}-12-31"

        buy = get_price_from_csv(ticker, buy_date, mode=buy_mode)
        sell = get_price_from_csv(ticker, sell_date, mode=sell_mode)

        ret = None
        if buy is not None and sell is not None and buy > 0:
            ret = (sell - buy) / buy
            capital *= (1 + ret)
        rows.append((year, ticker, buy_date, buy, sell_date, sell, ret))

    return rows, capital


def _print_results(rows, initial: float, final_capital: float):
    print("Year | Ticker | BuyDate | Buy | SellDate | Sell | Return%")
    valid_years = 0
    for y, t, bd, b, sd, s, r in rows:
        r_pct = None if r is None else round(r * 100, 2)
        if r is not None:
            valid_years += 1
        print(f"{y} | {t} | {bd} | {b} | {sd} | {s} | {r_pct}")

    cagr = None
    if valid_years > 0:
        cagr = (final_capital / initial) ** (1 / valid_years) - 1

    print(f"\nInitial capital: {initial:,.2f}")
    print(f"Final capital:   {final_capital:,.2f}")
    if cagr is not None:
        print(f"CAGR ({valid_years} yrs): {cagr * 100:.2f}%")


# ==============================
# Mapping helpers
# ==============================

def shift_eoy_to_next_year(mapping: Dict[int, str], max_year: Optional[int] = None) -> Dict[int, str]:
    """Shift an end-of-year mapping to the next holding year.
    Example: {2019:'META'} -> {2020:'META'}
    If max_year is provided, keep only years <= max_year.
    """
    shifted = {y + 1: t for y, t in mapping.items()}
    if max_year is not None:
        shifted = {y: t for y, t in shifted.items() if y <= max_year}
    return shifted


if __name__ == "__main__":
    # End-of-year (EOY) positions by your friend. You will hold them the next calendar year.
    eoy_positions = {
        # 2018: 'BSMX',  # omitted due to missing data
        2019: 'META',
        2020: 'SIG',
        2021: 'MSFT',
        2022: 'MSFT',
        2023: 'META',
        2024: 'GOOGL',  # would imply holding in 2025 (we drop since 2025 sell not in CSVs)
    }

    # Shift to next-year holdings and cap at 2024 so buy/sell both exist in quarterly CSVs
    portfolio = shift_eoy_to_next_year(eoy_positions, max_year=2024)

    initial = 1_000_000.0
    # Use Q4 prices: buy on/before Jan 1 (prior Q4), sell on/before Dec 31 (current Q4)
    rows, final_capital = backtest_yearly_mapping(
        portfolio,
        initial=initial,
        buy_mode='on_or_before',
        sell_mode='on_or_before'
    )

    _print_results(rows, initial, final_capital)