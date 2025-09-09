import pandas as pd, math, os, argparse

# Parameters (can be overridden via CLI)
MIN_INCLUDE_COV = 0.30   # minimum yearly avg coverage to include year in window
FULL_COV_THRESHOLD = 0.80  # no scaling if coverage >= this
CLIP_MIN = -0.95
CLIP_MAX = 3.0


def load_yearly(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Yearly file not found: {path}")
    df = pd.read_csv(path)
    # Ensure needed columns
    needed = {"inversionista", "anio", "year_return_raw", "year_weight_cov_avg"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in yearly file: {missing}")
    return df


def build_cagr3(df: pd.DataFrame,
                min_include_cov: float = MIN_INCLUDE_COV,
                full_cov_threshold: float = FULL_COV_THRESHOLD,
                normalize: bool = True) -> pd.DataFrame:
    recs = []
    for inv, g in df.groupby('inversionista'):
        g = g.sort_values('anio')
        years = g['anio'].tolist()
        for buy_year in years:
            window = [buy_year - 3, buy_year - 2, buy_year - 1]
            if not all(y in years for y in window):
                continue
            sub = g[g['anio'].isin(window)].set_index('anio')
            if sub.shape[0] != 3:
                continue
            rows = []
            valid = True
            for y in window:
                row = sub.loc[y]
                cov = row['year_weight_cov_avg'] if not pd.isna(row['year_weight_cov_avg']) else 0.0
                r_raw = row['year_return_raw'] if 'year_return_raw' in row and not pd.isna(row['year_return_raw']) else None
                if r_raw is None or cov < min_include_cov:
                    valid = False
                    break
                if normalize and cov < full_cov_threshold:
                    r_adj = r_raw / max(cov, 1e-6)
                    # clip extremes
                    r_adj = max(CLIP_MIN, min(r_adj, CLIP_MAX))
                    method = 'scaled'
                else:
                    r_adj = r_raw
                    method = 'raw'
                rows.append((y, cov, r_raw, r_adj, method))
            if not valid or len(rows) != 3:
                continue
            comp_raw = 1.0
            comp_adj = 1.0
            for _, _, r_raw, r_adj, _ in rows:
                comp_raw *= (1 + r_raw)
                comp_adj *= (1 + r_adj)
            comp_raw -= 1.0
            comp_adj -= 1.0
            cagr_raw = (1 + comp_raw) ** (1/3) - 1 if comp_raw > -1 else None
            cagr_adj = (1 + comp_adj) ** (1/3) - 1 if comp_adj > -1 else None
            rec = {
                'inversionista': inv,
                'buy_year': buy_year,
                'window_start_year': window[0],
                'window_end_year': window[-1],
                'compounded_return_raw': round(comp_raw, 6),
                'compounded_return_adj': round(comp_adj, 6),
                'cagr_raw': None if cagr_raw is None else round(cagr_raw, 6),
                'cagr_adj': None if cagr_adj is None else round(cagr_adj, 6),
                'min_include_cov': min_include_cov,
                'full_cov_threshold': full_cov_threshold
            }
            for (y, cov, r_raw, r_adj, method) in rows:
                rec[f'y{y}_cov'] = round(cov, 4)
                rec[f'y{y}_return_raw'] = round(r_raw, 6)
                rec[f'y{y}_return_adj'] = round(r_adj, 6)
                rec[f'y{y}_method'] = method
            recs.append(rec)
    return pd.DataFrame(recs).sort_values(['inversionista', 'buy_year'])


def main():
    ap = argparse.ArgumentParser(description='Compute 3-year rolling CAGR windows')
    ap.add_argument('--yearly-csv', default='investor_track_record_yearly.csv', help='Input yearly track CSV')
    ap.add_argument('--out', default='cagr3_windows.csv', help='Output CSV for CAGR windows')
    ap.add_argument('--min-cov', type=float, default=MIN_INCLUDE_COV, help='Minimum yearly coverage to include a year (default 0.30)')
    ap.add_argument('--full-cov', type=float, default=FULL_COV_THRESHOLD, help='Coverage threshold considered full (default 0.80)')
    ap.add_argument('--no-normalize', action='store_true', help='Disable scaling for partial coverage years')
    args = ap.parse_args()

    df_y = load_yearly(args.yearly_csv)
    out_df = build_cagr3(df_y, min_include_cov=args.min_cov, full_cov_threshold=args.full_cov, normalize=(not args.no_normalize))
    if out_df.empty:
        print('⚠️ No CAGR windows produced (coverage / data constraints).')
    else:
        out_df.to_csv(args.out, index=False)
        print(f'💾 Saved {len(out_df)} CAGR windows to {args.out}')
        print(out_df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
