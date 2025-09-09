import os
import sys
import numpy as np
import pandas as pd


def _normalize_percent_column(df: pd.DataFrame, col: str = "porcentaje_portafolio") -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return df
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Replace infinities with NaN explicitly
    df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    max_val = df[col].dropna().max()
    if pd.notna(max_val) and max_val is not None and max_val <= 1.5:
        df[col] = df[col] * 100.0
    return df


def pick_quarter(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    order = ["Q1", "Q2", "Q3", "Q4"]
    qs = [q for q in order if q in set(df['quarter'].astype(str))]
    if not qs:
        return None
    if preferred and preferred in qs:
        return preferred
    # Default to the latest available in the year
    return qs[-1]


def build_table(positions_csv: str,
                investor_code: str,
                year: int,
                quarter: str | None = None,
                normalize: bool = False) -> tuple[pd.DataFrame, dict]:
    if not os.path.exists(positions_csv):
        raise FileNotFoundError(f"No existe el archivo: {positions_csv}")

    df = pd.read_csv(positions_csv)

    # Basic schema checks
    required = {"anio", "quarter", "accion", "porcentaje_portafolio", "codigo_inversionista"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {positions_csv}: {missing}")

    # Normalize columns
    df['anio'] = pd.to_numeric(df['anio'], errors='coerce').astype('Int64')
    df = _normalize_percent_column(df)

    code_norm = str(investor_code).strip().casefold()
    df = df[df['codigo_inversionista'].astype(str).str.strip().str.casefold() == code_norm]
    df = df[df['anio'] == int(year)]
    if df.empty:
        raise ValueError(f"No hay posiciones para codigo_inversionista='{investor_code}' en anio={year}")

    chosen_quarter = pick_quarter(df, preferred=(quarter if quarter else None))
    if not chosen_quarter:
        raise ValueError("No hay 'quarter' válido para ese año")
    dfq = df[df['quarter'].astype(str) == chosen_quarter].copy()
    if dfq.empty:
        raise ValueError(f"No hay posiciones en {year} {chosen_quarter}")

    # Prepare table
    tbl = dfq[["accion", "porcentaje_portafolio"]].rename(columns={"porcentaje_portafolio": "percentage"}).copy()
    tbl['percentage'] = pd.to_numeric(tbl['percentage'], errors='coerce').fillna(0.0)
    tbl = tbl.groupby('accion', as_index=False)['percentage'].sum()

    if normalize:
        s = tbl['percentage'].sum()
        if s > 0:
            tbl['percentage'] = tbl['percentage'] * (100.0 / s)

    tbl['percentage'] = tbl['percentage'].round(2)
    tbl.sort_values('percentage', ascending=False, inplace=True)

    meta = {
        'investor_code': investor_code,
        'investor_name': (dfq['inversionista'].dropna().unique().tolist()[0]
                          if 'inversionista' in dfq.columns and not dfq['inversionista'].dropna().empty else None),
        'year': int(year),
        'quarter': chosen_quarter,
        'positions_count': int(tbl.shape[0]),
        'percent_sum': float(tbl['percentage'].sum())
    }
    return tbl, meta


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Muestra tabla de acciones y porcentajes por codigo de inversor y año")
    parser.add_argument("--positions-csv", default="top_positions_all_clean.csv", help="CSV de posiciones (default: top_positions_all_clean.csv)")
    parser.add_argument("--investor-code", required=True, help="Codigo del inversor (columna codigo_inversionista), p.ej. BRK, SEQUX, fairx")
    parser.add_argument("--year", type=int, required=True, help="Año a mostrar")
    parser.add_argument("--quarter", choices=["Q1","Q2","Q3","Q4"], default=None, help="Quarter específico (por defecto se usa el último disponible del año)")
    parser.add_argument("--normalize", action="store_true", help="Reescalar porcentajes para sumar 100%")
    parser.add_argument("--out", default="", help="Ruta opcional para guardar CSV de salida")
    args = parser.parse_args()

    try:
        tbl, meta = build_table(
            positions_csv=args.positions_csv,
            investor_code=args.investor_code,
            year=args.year,
            quarter=args.quarter,
            normalize=args.normalize,
        )
    except Exception as e:
        print(f"⚠️ {e}")
        sys.exit(1)

    # Pretty print
    title = f"Positions for {meta.get('investor_name') or ''} [{meta['investor_code']}] — {meta['year']} {meta['quarter']}"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(tbl.to_string(index=False))
    print("-" * 40)
    print(f"Total positions: {meta['positions_count']} | Sum%: {meta['percent_sum']:.2f}%" + (" (normalized)" if args.normalize else ""))

    if args.out:
        tbl.to_csv(args.out, index=False)
        print(f"💾 Guardado en {args.out}")


if __name__ == "__main__":
    main()
