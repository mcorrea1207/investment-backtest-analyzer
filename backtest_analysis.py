import argparse, os, pandas as pd, itertools, statistics, json, copy, sys, contextlib, io
from datetime import datetime
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_backtest_module(path='backtest_strategy.py'):
    spec = importlib.util.spec_from_file_location("backtest_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

@contextlib.contextmanager
def silent(quiet: bool):
    if not quiet:
        yield
        return
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')  # type: ignore
        sys.stderr = open(os.devnull, 'w')  # type: ignore
        yield
    finally:
        try:
            sys.stdout.close()  # type: ignore
            sys.stderr.close()  # type: ignore
        except Exception:
            pass
        sys.stdout, sys.stderr = saved_out, saved_err

def run_single(mod, start_year: int, end_year: int, num_investors: int, has_relocation: bool, selection_mode: int, cagr_metric: str, quiet: bool):
    mod.NUM_INVESTORS = num_investors
    mod.START_YEAR = start_year
    mod.END_YEAR = end_year
    mod.SELECTION_MODE = selection_mode
    mod.CAGR_METRIC = cagr_metric
    mod.HAS_RELOCATION = has_relocation
    with silent(quiet):
        result = mod.backtest_strategy(num_investors=num_investors,
                                       start_year=start_year,
                                       has_relocation=has_relocation,
                                       end_year=end_year)
    return result

def _task_runner(params):
    (path, sy, ey, ni, rel, selection_mode, cagr_metric, quiet) = params
    try:
        mod = load_backtest_module(path)
        res = run_single(mod, sy, ey, ni, rel, selection_mode, cagr_metric, quiet)
        return {'ok': True, 'data': res, 'sy': sy, 'ey': ey, 'ni': ni, 'rel': rel}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'sy': sy, 'ey': ey, 'ni': ni, 'rel': rel}

def compute_window_cagr(final_value: float, initial_capital: float, years: int):
    if initial_capital <= 0 or final_value <= 0 or years <= 0:
        return None
    growth = final_value / initial_capital
    return growth ** (1/years) - 1

def main():
    ap = argparse.ArgumentParser(description="Batch analysis: average CAGR of fixed 5y windows across different parameters.")
    ap.add_argument('--start-years', default='2012,2013,2014', help='Comma list of starting years (each will run a 5y window)')
    ap.add_argument('--window-years', type=int, default=5, help='Window length (years) fixed for fairness')
    ap.add_argument('--num-investors', default='3,5,8', help='Comma list of number of investors to test')
    ap.add_argument('--relocation', default='true,false', help='Comma list of relocation flags (true/false)')
    ap.add_argument('--selection-mode', type=int, default=3, help='Selection mode (e.g. 3= Cagr 3y, 4= Cagr acumulada)')
    ap.add_argument('--cagr-metric', default='cagr_raw_all', help='Metric for modes 3/4 (e.g. cagr_raw_all, cagr_gated_all, cagr_raw_3y, cagr_gated_3y)')
    ap.add_argument('--out', default='backtest_analysis_results.csv')
    ap.add_argument('--summary-out', default='backtest_analysis_summary.csv', help='Archivo CSV para guardar el resumen agregado')
    ap.add_argument('--plot', action='store_true', help='Generar gráfico comparando portfolio_cagr por num_investors y relocation')
    ap.add_argument('--quiet', action='store_true', help='Suprime la salida detallada de cada backtest')
    ap.add_argument('--parallel', type=int, default=1, help='Número de procesos en paralelo (1 = secuencial)')
    args = ap.parse_args()

    mod = load_backtest_module()

    start_years = [int(x.strip()) for x in args.start_years.split(',') if x.strip()]
    num_investors_list = [int(x.strip()) for x in args.num_investors.split(',') if x.strip()]
    relocation_flags = []
    for r in args.relocation.split(','):
        v = r.strip().lower()
        if v in ('true','t','1','yes','y'): relocation_flags.append(True)
        elif v in ('false','f','0','no','n'): relocation_flags.append(False)
    win = args.window_years

    tasks = []
    for sy, ni, rel in itertools.product(start_years, num_investors_list, relocation_flags):
        end_year = sy + win - 1
        tasks.append(('backtest_strategy.py', sy, end_year, ni, rel, args.selection_mode, args.cagr_metric, args.quiet))

    records = []
    if args.parallel > 1:
        print(f"🚀 Ejecutando en paralelo con {args.parallel} procesos ({len(tasks)} escenarios)...")
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            future_map = {ex.submit(_task_runner, t): t for t in tasks}
            for fut in as_completed(future_map):
                t = future_map[fut]
                sy, ey, ni, rel = t[1], t[2], t[3], t[4]
                try:
                    res_obj = fut.result()
                except Exception as e:
                    print(f"⚠️ Error (paralelo) {sy}-{ey} inv={ni} rel={rel}: {e}")
                    continue
                if not res_obj.get('ok'):
                    print(f"⚠️ Falló {sy}-{ey} inv={ni} rel={rel}: {res_obj.get('error')}")
                    continue
                res = res_obj['data']
                port_cagr = compute_window_cagr(res.get('capital_final'), res.get('capital_inicial'), win)
                bench_cagr = compute_window_cagr(res.get('benchmark_capital_final'), res.get('capital_inicial'), win)
                spread = (port_cagr - bench_cagr) if (port_cagr is not None and bench_cagr is not None) else None
                records.append({
                    'start_year': sy,
                    'end_year': ey,
                    'years': win,
                    'num_investors': ni,
                    'relocation': rel,
                    'selection_mode': args.selection_mode,
                    'cagr_metric_used': args.cagr_metric,
                    'portfolio_cagr': None if port_cagr is None else round(port_cagr,6),
                    'benchmark_cagr': None if bench_cagr is None else round(bench_cagr,6),
                    'cagr_spread': None if spread is None else round(spread,6),
                    'final_portfolio_value': res.get('capital_final'),
                    'final_benchmark_value': res.get('benchmark_capital_final')
                })
    else:
        for path, sy, ey, ni, rel, selection_mode, cagr_metric, quiet in tasks:
            print(f"➡️ Running window {sy}-{ey} | investors={ni} | relocation={rel}")
            try:
                mod_local = load_backtest_module(path)
                res = run_single(mod_local, sy, ey, ni, rel, selection_mode, cagr_metric, quiet)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"⚠️ Error en ventana {sy}-{ey}: {e}")
                continue
            port_cagr = compute_window_cagr(res.get('capital_final'), res.get('capital_inicial'), win)
            bench_cagr = compute_window_cagr(res.get('benchmark_capital_final'), res.get('capital_inicial'), win)
            spread = (port_cagr - bench_cagr) if (port_cagr is not None and bench_cagr is not None) else None
            records.append({
                'start_year': sy,
                'end_year': ey,
                'years': win,
                'num_investors': ni,
                'relocation': rel,
                'selection_mode': selection_mode,
                'cagr_metric_used': cagr_metric,
                'portfolio_cagr': None if port_cagr is None else round(port_cagr,6),
                'benchmark_cagr': None if bench_cagr is None else round(bench_cagr,6),
                'cagr_spread': None if spread is None else round(spread,6),
                'final_portfolio_value': res.get('capital_final'),
                'final_benchmark_value': res.get('benchmark_capital_final')
            })

    if not records:
        print("⚠️ No se generaron resultados.")
        return
    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"💾 Guardado {len(df)} escenarios en {args.out}")
    # Resumen estadístico
    grp = df.groupby(['num_investors','relocation'])['portfolio_cagr']
    summary = grp.agg(['count','mean','median','std']).reset_index()
    summary = summary.rename(columns={'mean':'avg_portfolio_cagr','std':'std_portfolio_cagr','count':'n'})
    summary.to_csv(args.summary_out, index=False)
    print("\nResumen estadístico por (num_investors, relocation):")
    print(summary.to_string(index=False))
    print(f"💾 Resumen guardado en {args.summary_out}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            piv = summary.pivot(index='num_investors', columns='relocation', values='avg_portfolio_cagr')
            plt.figure(figsize=(8,5))
            for rel in piv.columns:
                plt.plot(piv.index, piv[rel], marker='o', label=f"relocation={rel}")
            plt.title('Avg Portfolio CAGR vs num_investors')
            plt.xlabel('num_investors')
            plt.ylabel('Avg Portfolio CAGR')
            plt.grid(alpha=0.3)
            plt.legend()
            plot_file = 'backtest_analysis_plot.png'
            plt.tight_layout(); plt.savefig(plot_file, dpi=200)
            print(f"🖼️ Gráfico guardado en {plot_file}")
        except Exception as e:
            print(f"⚠️ No se pudo generar gráfico: {e}")

if __name__ == '__main__':
    main()
