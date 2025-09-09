import pandas as pd, argparse, os

def load_yearly(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # ensure needed cols
    need = {"inversionista","anio","year_return","year_return_raw"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    return df

def rolling_cagr(vals):
    comp = 1.0
    for r in vals:
        if r <= -1:  # impossible / invalid
            return None, None
        comp *= (1 + r)
    comp_ret = comp - 1
    cagr = comp ** (1/3) - 1
    return comp_ret, cagr

def build_cagr(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for inv, g in df.groupby('inversionista'):
        g = g.sort_values('anio')
        years = g['anio'].tolist()
        for y in years:
            window = [y-3, y-2, y-1]
            if not all(w in years for w in window):
                continue
            sub = g[g['anio'].isin(window)].set_index('anio')
            if sub.shape[0] != 3:
                continue
            # gated
            gated_vals = []
            raw_vals = []
            gated_ok = True
            raw_ok = True
            details = {}
            for wy in window:
                yr_ret = sub.at[wy,'year_return'] if 'year_return' in sub.columns else None
                yr_ret_raw = sub.at[wy,'year_return_raw'] if 'year_return_raw' in sub.columns else None
                details[f'y{wy}_year_return'] = yr_ret
                details[f'y{wy}_year_return_raw'] = yr_ret_raw
                if pd.isna(yr_ret):
                    gated_ok = False
                else:
                    gated_vals.append(float(yr_ret))
                if pd.isna(yr_ret_raw):
                    raw_ok = False
                else:
                    raw_vals.append(float(yr_ret_raw))
            comp_gated = cagr_gated = None
            comp_raw = cagr_raw = None
            if gated_ok:
                comp_gated, cagr_gated = rolling_cagr(gated_vals)
            if raw_ok:
                comp_raw, cagr_raw = rolling_cagr(raw_vals)
            rec = {
                'inversionista': inv,
                'buy_year': y,
                'window_start_year': window[0],
                'window_end_year': window[-1],
                'compounded_gated': None if comp_gated is None else round(comp_gated,6),
                'cagr_gated': None if cagr_gated is None else round(cagr_gated,6),
                'compounded_raw': None if comp_raw is None else round(comp_raw,6),
                'cagr_raw': None if cagr_raw is None else round(cagr_raw,6),
            }
            rec.update(details)
            recs.append(rec)
    return pd.DataFrame(recs).sort_values(['inversionista','buy_year'])


def main():
    ap = argparse.ArgumentParser(description='Compute dual 3y CAGR (gated vs raw) from yearly track record file')
    ap.add_argument('--yearly-csv', default='investor_track_record_yearly.csv')
    ap.add_argument('--out', default='cagr3_dual.csv')
    args = ap.parse_args()
    df = load_yearly(args.yearly_csv)
    out_df = build_cagr(df)
    if out_df.empty:
        print('⚠️ No CAGR windows generated.')
    else:
        out_df.to_csv(args.out, index=False)
        print(f'💾 Saved {len(out_df)} rows to {args.out}')
        print(out_df.head(12).to_string(index=False))

if __name__ == '__main__':
    main()
