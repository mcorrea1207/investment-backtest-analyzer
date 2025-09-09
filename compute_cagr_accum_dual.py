import pandas as pd, argparse, os


def load_yearly(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = {"inversionista", "anio", "year_return", "year_return_raw"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    return df


def compounded_and_cagr(returns):
    """Given a list of yearly returns (floats), compute total compounded return and CAGR.
    Returns (compounded_return, cagr). compounded_return = product(1+r)-1
    CAGR = (1+compounded_return) ** (1/n) - 1
    If any r <= -1 -> None, None
    """
    if not returns:
        return None, None
    comp = 1.0
    for r in returns:
        if r is None or pd.isna(r):
            return None, None
        if r <= -1:
            return None, None
        comp *= (1 + r)
    comp_ret = comp - 1
    n = len(returns)
    cagr = comp ** (1 / n) - 1
    return comp_ret, cagr


def build_accumulated(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for inv, g in df.groupby('inversionista'):
        g = g.sort_values('anio').reset_index(drop=True)
        years = g['anio'].tolist()
        # Pre-build maps for speed
        gated_map = dict(zip(g['anio'], g['year_return']))
        raw_map = dict(zip(g['anio'], g['year_return_raw']))
        earliest = years[0] if years else None
        for y in years:
            # We want all years strictly BEFORE y
            prior_years = [yr for yr in years if yr < y]
            if not prior_years:
                continue  # nothing to accumulate yet
            gated_vals = [gated_map[yr] for yr in prior_years]
            raw_vals = [raw_map[yr] for yr in prior_years]
            # Strict rule: if ANY gated year missing -> gated result None
            gated_complete = all(not pd.isna(v) for v in gated_vals)
            raw_complete = all(not pd.isna(v) for v in raw_vals)
            comp_gated = cagr_gated = None
            comp_raw = cagr_raw = None
            if gated_complete:
                comp_gated, cagr_gated = compounded_and_cagr(gated_vals)
            if raw_complete:
                comp_raw, cagr_raw = compounded_and_cagr(raw_vals)
            rec = {
                'inversionista': inv,
                'buy_year': y,  # anchor year (not included in the window)
                'window_start_year': min(prior_years),
                'window_end_year': max(prior_years),
                'years_in_window': len(prior_years),
                'compounded_gated_all': None if comp_gated is None else round(comp_gated, 6),
                'cagr_gated_all': None if cagr_gated is None else round(cagr_gated, 6),
                'compounded_raw_all': None if comp_raw is None else round(comp_raw, 6),
                'cagr_raw_all': None if cagr_raw is None else round(cagr_raw, 6),
            }
            records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(['inversionista', 'buy_year']).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description='Compute accumulated (from first year up to year-1) CAGR (gated vs raw).')
    ap.add_argument('--yearly-csv', default='investor_track_record_yearly.csv')
    ap.add_argument('--out', default='cagr_accum_dual.csv')
    args = ap.parse_args()
    df = load_yearly(args.yearly_csv)
    out_df = build_accumulated(df)
    if out_df.empty:
        print('⚠️ No accumulated CAGR rows produced.')
        return
    out_df.to_csv(args.out, index=False)
    print(f'💾 Saved {len(out_df)} rows to {args.out}')
    print(out_df.head(12).to_string(index=False))


if __name__ == '__main__':
    main()
