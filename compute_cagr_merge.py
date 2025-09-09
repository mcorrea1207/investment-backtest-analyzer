import pandas as pd, argparse, os


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_cagr(cagr3_path: str, accum_path: str) -> pd.DataFrame:
    c3 = load_csv(cagr3_path) if os.path.exists(cagr3_path) else pd.DataFrame()
    ac = load_csv(accum_path) if os.path.exists(accum_path) else pd.DataFrame()
    if c3.empty and ac.empty:
        return pd.DataFrame()
    # Reduce columns
    if not c3.empty:
        keep3 = ['inversionista','buy_year','window_start_year','window_end_year','cagr_gated','cagr_raw','compounded_gated','compounded_raw']
        c3 = c3[[c for c in keep3 if c in c3.columns]].rename(columns={
            'window_start_year':'window3_start','window_end_year':'window3_end',
            'cagr_gated':'cagr_gated_3y','cagr_raw':'cagr_raw_3y',
            'compounded_gated':'compounded_gated_3y','compounded_raw':'compounded_raw_3y'
        })
    if not ac.empty:
        keepa = ['inversionista','buy_year','window_start_year','window_end_year','years_in_window','cagr_gated_all','cagr_raw_all','compounded_gated_all','compounded_raw_all']
        ac = ac[[c for c in keepa if c in ac.columns]].rename(columns={
            'window_start_year':'accum_start','window_end_year':'accum_end'
        })
    if c3.empty:
        merged = ac
    elif ac.empty:
        merged = c3
    else:
        merged = pd.merge(c3, ac, on=['inversionista','buy_year'], how='outer')
    merged = merged.sort_values(['inversionista','buy_year']).reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description='Merge 3y and accumulated CAGR files into one unified table.')
    ap.add_argument('--cagr3', default='cagr3_dual.csv')
    ap.add_argument('--accum', default='cagr_accum_dual.csv')
    ap.add_argument('--out', default='cagr_merged.csv')
    args = ap.parse_args()
    df = merge_cagr(args.cagr3, args.accum)
    if df.empty:
        print('⚠️ No data to merge.')
        return
    df.to_csv(args.out, index=False)
    print(f'💾 Saved merged file with {len(df)} rows to {args.out}')
    print(df.head(12).to_string(index=False))

if __name__ == '__main__':
    main()
