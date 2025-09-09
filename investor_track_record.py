import os
import warnings
import json
import pandas as pd

# ========================================
# CONFIGURACIÓN
# ========================================
# Cambia este valor para ver cuántos años hacia atrás calcular en el track record de ventana
LOOKBACK_YEARS_DEFAULT = 3
# Archivo de posiciones limpias con pesos por inversor
POSITIONS_CLEAN_CSV = 'top_positions_all_clean.csv'
# Umbral mínimo de cobertura por porcentaje de peso en posiciones con precio disponible (0-1)
COVERAGE_MIN_RATIO = 0.8

# ========================================
# Utilidades de fechas y precios (autónomas; no dependen de backtest_strategy)
# ========================================

def _quarter_end_date(year: int, quarter: str) -> str:
    mapping = {
        'Q1': '03-31',
        'Q2': '06-30',
        'Q3': '09-30',
        'Q4': '12-31',
    }
    day = mapping.get(quarter.upper(), '12-31')
    return f"{year}-{day}"


def _candidate_tickers(ticker: str) -> list:
    """Return plausible alternative tickers for local CSV filenames (handles scraped suffixes).
    Examples: 'SHLDQ-OLD' -> ['SHLDQ-OLD', 'SHLDQ']
    """
    cands = []
    if not isinstance(ticker, str):
        return [ticker]
    t = ticker.strip()
    cands.append(t)
    if t.endswith("-OLD"):
        cands.append(t[:-4])
    # try alias map if present
    try:
        alias_path = os.path.join(os.getcwd(), 'symbol_aliases.json')
        if os.path.exists(alias_path):
            with open(alias_path, 'r') as f:
                amap = json.load(f)
            t_up = t.upper()
            if isinstance(amap, dict) and t_up in amap and amap[t_up]:
                cands.append(str(amap[t_up]).strip().upper())
    except Exception:
        pass
    return list(dict.fromkeys(cands))


def next_quarter(year: int, quarter: str) -> tuple[int, str]:
    order = ['Q1', 'Q2', 'Q3', 'Q4']
    q = quarter.upper()
    if q not in order:
        return year, 'Q4'
    idx = order.index(q)
    if idx == 3:
        return year + 1, 'Q1'
    return year, order[idx + 1]


def get_stock_price_for_date(ticker, date_str, mode: str = 'nearest'):
    """Precio de CSV trimestral local (fallback cuando no hay diarios).
    mode: 'on_or_after' | 'on_or_before' | 'nearest'
    """
    try:
        df = None
        for tk in _candidate_tickers(ticker):
            csv_path = os.path.join('stock_prices', f'{tk}_quarterly_prices.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                break
        if df is None:
            return None
        if df.empty or 'fecha' not in df.columns or 'precio_cierre' not in df.columns:
            return None
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
        else:
            diffs = (df['fecha'] - target_date).abs()
            idx = diffs.idxmin()
            row = df.loc[idx]
        return round(float(row['precio_cierre']), 2)
    except Exception:
        return None


def get_daily_price_for_date(ticker, date_str, mode: str = 'nearest', use_adjusted: bool = True):
    """Precio de CSV diario local, con fallback a trimestral.
    mode: 'on_or_after' | 'on_or_before' | 'nearest'
    """
    try:
        df = None
        for tk in _candidate_tickers(ticker):
            for base_dir in ['manual_prices_daily', 'stock_prices_daily']:
                csv_path = os.path.join(base_dir, f'{tk}_daily_prices.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    break
            if df is not None:
                break
        if df is None:
            return get_stock_price_for_date(ticker, date_str, mode)
        if df.empty or 'fecha' not in df.columns or 'precio_cierre' not in df.columns:
            return get_stock_price_for_date(ticker, date_str, mode)
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
        else:
            diffs = (df['fecha'] - target_date).abs()
            idx = diffs.idxmin()
            row = df.loc[idx]
        price_col = 'precio_ajustado' if use_adjusted and 'precio_ajustado' in df.columns else 'precio_cierre'
        return round(float(row[price_col]), 2)
    except Exception:
        return get_stock_price_for_date(ticker, date_str, mode)


# ========================================
# Cálculo de track records
# ========================================

def _is_warrant(sym: str) -> bool:
    s = str(sym).strip().upper()
    return s.endswith('W') or s.endswith('WS') or '-WT' in s or '-WTA' in s or '-WTB' in s


def _is_bankrupt(sym: str) -> bool:
    s = str(sym).strip().upper()
    return s.endswith('Q')


def compute_investor_track_record(src: str = POSITIONS_CLEAN_CSV,
                                  investor: str | None = None,
                                  investor_code: str | None = None,
                                  use_adjusted: bool = True,
                                  boundary: str = 'on_or_before',
                                  drop_warrants: bool = False,
                                  drop_bankrupt: bool = False,
                                  no_coverage_gate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula el track record por inversor a nivel trimestral y anual.
    Retorna (df_trimestral, df_anual) y NO escribe archivos por sí solo.
    """
    if not os.path.exists(src):
        print(f"⚠️ No existe {src}; no es posible evaluar el portafolio completo por trimestre.")
        return pd.DataFrame(), pd.DataFrame()
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"⚠️ Error leyendo {src}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'anio' in df.columns:
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce').astype('Int64')
        if 'porcentaje_portafolio' in df.columns:
            df['porcentaje_portafolio'] = pd.to_numeric(df['porcentaje_portafolio'], errors='coerce')
            max_val = df['porcentaje_portafolio'].dropna().max()
            if pd.notna(max_val) and max_val is not None and max_val <= 1.5:
                df['porcentaje_portafolio'] = df['porcentaje_portafolio'] * 100

    # Filtros opcionales por inversor
    if investor_code:
        code_norm = investor_code.strip().casefold()
        if 'codigo_inversionista' in df.columns:
            df = df[df['codigo_inversionista'].astype(str).str.strip().str.casefold() == code_norm]
        if df.empty:
            print(f"⚠️ No hay filas para codigo_inversionista '{investor_code}' en {src}")
            return pd.DataFrame(), pd.DataFrame()
    elif investor:
        inv_norm = investor.strip().casefold()
        if 'inversionista' in df.columns:
            df = df[df['inversionista'].astype(str).str.strip().str.casefold() == inv_norm]
        if df.empty:
            print(f"⚠️ No hay filas para inversionista '{investor}' en {src}")
            return pd.DataFrame(), pd.DataFrame()

    req_cols = {'inversionista','anio','quarter','accion'}
    missing = req_cols - set(df.columns)
    if missing:
        print(f"⚠️ Faltan columnas en {src}: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    q_order = {"Q1":1, "Q2":2, "Q3":3, "Q4":4}
    results = []

    investors = sorted(df['inversionista'].dropna().unique().tolist())
    for inv in investors:
        dfi = df[df['inversionista'] == inv].copy()
        dfi = dfi.dropna(subset=['anio','quarter'])
        for (y, q) in sorted(dfi[['anio','quarter']].drop_duplicates().itertuples(index=False, name=None), key=lambda x: (int(x[0]), q_order.get(str(x[1]), 9))):
            try:
                y_int = int(y)
                q_str = str(q)
                start_date = _quarter_end_date(y_int, q_str)
                ny, nq = next_quarter(y_int, q_str)
                end_date = _quarter_end_date(ny, nq)
                snap = dfi[(dfi['anio'] == y_int) & (dfi['quarter'] == q_str)]
                # Exclusiones opcionales de símbolos sin datos razonables
                excluded_mask = pd.Series(False, index=snap.index)
                if drop_warrants:
                    excluded_mask = excluded_mask | snap['accion'].astype(str).map(_is_warrant)
                if drop_bankrupt:
                    excluded_mask = excluded_mask | snap['accion'].astype(str).map(_is_bankrupt)
                excluded_weight_pct = 0.0
                if 'porcentaje_portafolio' in snap.columns and snap['porcentaje_portafolio'].notna().any():
                    excluded_weight_pct = float(snap.loc[excluded_mask, 'porcentaje_portafolio'].fillna(0).sum())
                if excluded_mask.any():
                    snap = snap[~excluded_mask].copy()
                if 'porcentaje_portafolio' in snap.columns and snap['porcentaje_portafolio'].notna().any():
                    weights = snap['porcentaje_portafolio'].fillna(0).clip(lower=0)
                else:
                    weights = pd.Series(1.0, index=snap.index)
                total_w = weights.sum()
                if total_w <= 0:
                    continue
                snap = snap.assign(weight=weights / total_w)
                rets, used_w = [], []
                tickers = snap['accion'].astype(str).tolist()
                for i, t in enumerate(tickers):
                    w = float(snap.iloc[i]['weight'])
                    p0 = get_daily_price_for_date(t, start_date, mode=boundary, use_adjusted=use_adjusted)
                    p1 = get_daily_price_for_date(t, end_date, mode=boundary, use_adjusted=use_adjusted)
                    if p0 and p1 and p0 > 0:
                        rets.append((p1 - p0) / p0)
                        used_w.append(w)
                n_pos = len(tickers)
                n_used = len(used_w)
                coverage_count_ratio = (n_used / n_pos) if n_pos > 0 else 0.0
                if not rets or sum(used_w) == 0:
                    inv_ret = None
                    coverage_weight = 0.0
                else:
                    w_sum = sum(used_w)
                    used_w_norm = [w / w_sum for w in used_w]
                    inv_ret = float(sum(w * r for w, r in zip(used_w_norm, rets)))
                    coverage_weight = w_sum  # proporción de peso con precios disponibles (0-1)
                # Cobertura por peso (no por número de posiciones)
                coverage_ok = coverage_weight >= COVERAGE_MIN_RATIO
                # raw (always computed if possible) vs gated (respect coverage threshold unless disabled)
                raw_ret = inv_ret
                gated_ret = inv_ret if (coverage_ok or no_coverage_gate) else None
                results.append({
                    'inversionista': inv,
                    'anio': y_int,
                    'quarter': q_str,
                    'start_date': start_date,
                    'end_date': end_date,
                    'portfolio_return_raw': None if raw_ret is None else round(raw_ret, 6),
                    'portfolio_return': None if gated_ret is None else round(gated_ret, 6),
                    'positions': n_pos,
                    'positions_used': n_used,
                    'coverage_ratio': round(coverage_count_ratio, 4),
                    'coverage_ok': coverage_ok,
                    'weight_coverage': round(coverage_weight, 4),
                    'excluded_weight_pct': round(excluded_weight_pct, 2)
                })
            except Exception:
                continue

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    df_q = pd.DataFrame(results)
    df_q.sort_values(['inversionista','anio','quarter'], key=lambda s: s.map(q_order).fillna(9) if s.name=='quarter' else s, inplace=True)

    # Anual compuesto (raw y gated) + métricas de cobertura
    yearly_rows = []
    for inv, grp in df_q.groupby('inversionista'):
        for y, g in grp.groupby('anio'):
            g_raw = g.dropna(subset=['portfolio_return_raw'])
            raw_year = float((1.0 + g_raw['portfolio_return_raw']).prod() - 1.0) if not g_raw.empty else None
            g_gated = g.dropna(subset=['portfolio_return'])
            gated_year = float((1.0 + g_gated['portfolio_return']).prod() - 1.0) if not g_gated.empty else None
            quarters_used = int(g_gated.shape[0]) if not g_gated.empty else 0
            avg_cov = float(g['weight_coverage'].mean()) if not g.empty else 0.0
            min_cov = float(g['weight_coverage'].min()) if not g.empty else 0.0
            yearly_rows.append({
                'inversionista': inv,
                'anio': int(y),
                'quarters_used': quarters_used,
                'year_return': None if gated_year is None else round(gated_year, 6),
                'year_return_raw': None if raw_year is None else round(raw_year, 6),
                'year_weight_cov_avg': round(avg_cov,4),
                'year_weight_cov_min': round(min_cov,4)
            })
    df_y = pd.DataFrame(yearly_rows).sort_values(['inversionista','anio'])
    return df_q, df_y


def build_cagr_windows_3y(df_y: pd.DataFrame,
                          min_include_cov: float = 0.3,
                          full_cov_threshold: float = 0.8,
                          normalize: bool = True) -> pd.DataFrame:
    """CAGR rolling de 3 años para cada inversor.
    Ventana para buy_year = (buy_year-3, buy_year-2, buy_year-1).
    Reglas de inclusión de cada año de la ventana:
      - year_return_raw disponible.
      - year_weight_cov_avg >= min_include_cov.
    Normalización: si cobertura < full_cov_threshold y >= min_include_cov y normalize=True, ajusta r = r_raw / cobertura.
    Devuelve filas con CAGRs raw/ajustado y detalles anuales.
    """
    if df_y.empty:
        return pd.DataFrame()
    recs = []
    for inv, g in df_y.groupby('inversionista'):
        g = g.sort_values('anio')
        years = g['anio'].tolist()
        for by in years:
            yset = [by-3, by-2, by-1]
            if not all(y in years for y in yset):
                continue
            sub = g[g['anio'].isin(yset)].set_index('anio')
            if sub.shape[0] != 3:
                continue
            valid = True
            yr_rows = []
            for y in yset:
                row = sub.loc[y]
                cov = row.get('year_weight_cov_avg', 0) or 0.0
                r_raw = row.get('year_return_raw')
                if pd.isna(r_raw) or cov < min_include_cov:
                    valid = False
                    break
                if cov >= full_cov_threshold or not normalize:
                    r_adj = r_raw
                    method = 'raw'
                else:
                    r_adj = r_raw / max(cov,1e-6)
                    method = 'scaled'
                yr_rows.append((y, cov, r_raw, r_adj, method))
            if not valid:
                continue
            comp_raw = 1.0
            comp_adj = 1.0
            for _, _, r_raw, r_adj, _ in yr_rows:
                comp_raw *= (1+r_raw)
                comp_adj *= (1+r_adj)
            comp_raw -= 1.0
            comp_adj -= 1.0
            cagr_raw = (1+comp_raw)**(1/3)-1 if comp_raw > -1 else None
            cagr_adj = (1+comp_adj)**(1/3)-1 if comp_adj > -1 else None
            rec = {
                'inversionista': inv,
                'buy_year': by,
                'window_start_year': yset[0],
                'window_end_year': yset[-1],
                'compounded_return_raw': round(comp_raw,6),
                'compounded_return_adj': round(comp_adj,6),
                'cagr_raw': None if cagr_raw is None else round(cagr_raw,6),
                'cagr_adj': None if cagr_adj is None else round(cagr_adj,6),
                'min_include_cov': min_include_cov,
                'full_cov_threshold': full_cov_threshold
            }
            for (y, cov, r_raw, r_adj, method) in yr_rows:
                rec[f'y{y}_cov'] = round(cov,4)
                rec[f'y{y}_return_raw'] = round(r_raw,6)
                rec[f'y{y}_return_adj'] = round(r_adj,6)
                rec[f'y{y}_method'] = method
            recs.append(rec)
    return pd.DataFrame(recs).sort_values(['inversionista','buy_year'])


def save_investor_track_record(df_q: pd.DataFrame, df_y: pd.DataFrame,
                               out_quarter_csv: str | None = None,
                               out_year_csv: str = 'investor_track_record_yearly.csv'):
    # Solo guardar trimestral si se especifica explícitamente
    if out_quarter_csv:
        if not df_q.empty:
            df_q.to_csv(out_quarter_csv, index=False)
            print(f"💾 Track record trimestral guardado en {out_quarter_csv}")
        else:
            print("⚠️ Track record trimestral vacío; no se guardó CSV")
    # Guardar anual (requerido)
    if not df_y.empty:
        df_y.to_csv(out_year_csv, index=False)
        print(f"💾 Track record anual guardado en {out_year_csv}")
    else:
        print("⚠️ Track record anual vacío; no se guardó CSV")


def build_lookback_track_record(buy_year: int,
                                lookback_years: int = LOOKBACK_YEARS_DEFAULT,
                                out_csv: str = 'investor_track_record_lookback.csv',
                                df_y: pd.DataFrame | None = None,
                                src: str = POSITIONS_CLEAN_CSV,
                                investor: str | None = None):
    """Genera un CSV con el rendimiento de CADA inversor en una ventana de lookback.
    Ej.: buy_year=2019, lookback_years=3 -> usa 2016, 2017, 2018.
    Crea columnas por año, retorno compuesto y anualizado.
    """
    if df_y is None or df_y.empty:
        _dfq, df_y = compute_investor_track_record(src, investor=investor)
    if df_y is None or df_y.empty:
        print("⚠️ No hay datos anuales para generar el lookback.")
        return

    start_year = int(buy_year - lookback_years)
    end_year = int(buy_year - 1)
    years_range = list(range(start_year, end_year + 1))

    rows = []
    for inv, grp in df_y.groupby('inversionista'):
        # Diccionario año->retorno
        yr_map = {int(r['anio']): (None if pd.isna(r['year_return']) else float(r['year_return'])) for _, r in grp.iterrows()}
        per_year = {}
        available_returns = []
        for y in years_range:
            r = yr_map.get(y)
            per_year[y] = r
            if r is not None:
                available_returns.append(r)
        years_available = len([r for r in available_returns if r is not None])
        years_missing = len(years_range) - years_available
        if years_available > 0:
            compounded = float((pd.Series(available_returns).add(1.0)).prod() - 1.0)
            annualized = float((1.0 + compounded) ** (1.0 / years_available) - 1.0)
        else:
            compounded = None
            annualized = None
        row = {
            'inversionista': inv,
            'window_start_year': start_year,
            'window_end_year': end_year,
            'years_in_window': len(years_range),
            'years_available': years_available,
            'years_missing': years_missing,
            'compounded_return': None if compounded is None else round(compounded, 6),
            'annualized_return': None if annualized is None else round(annualized, 6),
        }
        # Agregar columnas por año
        for y in years_range:
            r = per_year[y]
            row[f'y{y}'] = None if r is None else round(r, 6)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(['inversionista'])
    out_df.to_csv(out_csv, index=False)
    print(f"💾 Track record (ventana lookback) guardado en {out_csv} | Ventana: {start_year}-{end_year}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera track records por inversor y ventana de lookback")
    parser.add_argument("--positions-csv", default=POSITIONS_CLEAN_CSV, help="CSV de posiciones limpias (default: top_positions_all_clean.csv)")
    parser.add_argument("--investor", default="", help="Filtrar por nombre de inversionista (case-insensitive). Usa '' para omitir")
    parser.add_argument("--investor-code", default="", help="Filtrar por codigo_inversionista (p.ej. BRK, SEQUX). Tiene prioridad sobre --investor")
    # Por defecto no guardamos el CSV trimestral
    parser.add_argument("--out-quarter", default="", help="Salida CSV trimestral (opcional; vacío para omitir)")
    parser.add_argument("--out-year", default="investor_track_record_yearly.csv", help="Salida CSV anual")
    parser.add_argument("--buy-year", type=int, default=2019, help="Año de compra para ventana lookback (default: 2019)")
    parser.add_argument("--lookback-years", type=int, default=LOOKBACK_YEARS_DEFAULT, help="Años hacia atrás para ventana lookback (default: 3)")
    parser.add_argument("--out-lookback", default="investor_track_record_lookback.csv", help="Salida CSV lookback")
    parser.add_argument("--coverage-min", type=float, default=COVERAGE_MIN_RATIO, help="Umbral mínimo de cobertura por peso (0-1, default: 0.8)")
    parser.add_argument("--use-adjusted", action="store_true", help="Usar precios ajustados (incluyen dividendos/splits) cuando estén disponibles")
    parser.add_argument("--boundary", choices=["on_or_before","on_or_after","nearest"], default="on_or_before", help="Regla para elegir precio en el límite trimestral")
    parser.add_argument("--drop-warrants", action="store_true", help="Excluir símbolos de warrants (WS/W/WT*) del cálculo")
    parser.add_argument("--drop-bankrupt", action="store_true", help="Excluir símbolos con sufijo Q (bancarrota OTC)")
    parser.add_argument("--cagr-last3-out", default="", help="Archivo de salida para ventanas CAGR 3y (todos los inversores, requiere datos completos)")
    parser.add_argument("--cagr-min-cov", type=float, default=0.3, help="Cobertura mínima anual para incluir año en ventana (default 0.3)")
    parser.add_argument("--cagr-full-cov", type=float, default=0.8, help="Cobertura que se considera completa (no se normaliza, default 0.8)")
    parser.add_argument("--debug-year", type=int, default=None, help="Imprime detalle trimestral y cobertura para este año del inversor filtrado")
    args = parser.parse_args()

    # Aplicar umbral de cobertura si se sobreescribe
    COVERAGE_MIN_RATIO = float(args.coverage_min)

    print("="*60)
    print("🧾 Generando track records por inversor (solo anual)")
    print(f"📄 Positions CSV: {args.positions_csv}")
    who = args.investor_code or args.investor or 'ALL'
    print(f"👤 Investor filter: {who}")
    if args.out_quarter:
        print(f"📤 Out quarterly: {args.out_quarter}")
    print(f"📤 Out yearly:    {args.out_year}")
    print(f"🔭 Lookback: buy_year={args.buy_year}, years={args.lookback_years}")
    print(f"📤 Out lookback:  {args.out_lookback}")
    print(f"✅ Cobertura mínima (peso): {COVERAGE_MIN_RATIO*100:.0f}%")
    print("="*60)

    df_q, df_y = compute_investor_track_record(args.positions_csv,
                                               investor=(args.investor or None),
                                               investor_code=(args.investor_code or None),
                                               use_adjusted=bool(args.use_adjusted),
                                               boundary=args.boundary,
                                               drop_warrants=bool(args.drop_warrants),
                                               drop_bankrupt=bool(args.drop_bankrupt))
    if df_q.empty and df_y.empty:
        print("⚠️ No se generaron datos. Verifica el CSV de posiciones.")
        raise SystemExit(1)

    save_investor_track_record(df_q, df_y, out_quarter_csv=(args.out_quarter or None), out_year_csv=args.out_year)

    # Debug anual (opcional)
    if args.debug_year and not df_q.empty:
        dbg = df_q[df_q['anio'] == int(args.debug_year)].copy()
        if not dbg.empty:
            cols = ["anio","quarter","portfolio_return","positions","positions_used","coverage_ok","weight_coverage","excluded_weight_pct"]
            print("\n🔎 Debug anual (trimestres):")
            print(dbg[cols].to_string(index=False))
        else:
            print(f"🔎 No hay trimestres para el año {args.debug_year} con el filtro actual")

    # Generar lookback
    build_lookback_track_record(buy_year=args.buy_year,
                                lookback_years=args.lookback_years,
                                out_csv=args.out_lookback,
                                df_y=df_y,
                                src=args.positions_csv,
                                investor=(args.investor or None))

    # CAGR 3y windows for all investors (ignore single-investor filter) if requested
    if args.cagr_last3_out:
        # recompute without gating to get raw coverage/returns
        df_q_all, df_y_all = compute_investor_track_record(args.positions_csv,
                                                           use_adjusted=bool(args.use_adjusted),
                                                           boundary=args.boundary,
                                                           drop_warrants=False,
                                                           drop_bankrupt=False,
                                                           no_coverage_gate=True)
        cagr_df = build_cagr_windows_3y(df_y_all,
                                        min_include_cov=float(args.cagr_min_cov),
                                        full_cov_threshold=float(args.cagr_full_cov),
                                        normalize=True)
        if not cagr_df.empty:
            cagr_df.to_csv(args.cagr_last3_out, index=False)
            print(f"💾 CAGR 3y windows guardado en {args.cagr_last3_out}")
        else:
            print("⚠️ No se generaron ventanas CAGR 3y (falta de cobertura o datos)")

    print("✅ Listo.")
