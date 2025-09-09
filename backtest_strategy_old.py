import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import sys
from io import StringIO
import random

# ========================================
# CONFIGURACIÓN - EDITA AQUÍ TUS PARÁMETROS
# ========================================
NUM_INVESTORS = 8          # Número de inversores a seguir (se elegirán aleatoriamente)
START_YEAR = 2020          # Año de inicio del backtest
HAS_RELOCATION = True      # True = Con rebalanceo | False = Sin rebalanceo
REDISTRIBUTE_FAILED = True # True = Redistribuir capital de inversores fallidos | False = Mantener en efectivo
RANDOM_SEED = 1207         # Semilla para reproducibilidad (cambia para obtener diferentes inversores aleatorios)
MIN_YEARS_DATA = 3         # Mínimo de años con datos Q3 para incluir un inversor
# ========================================

def get_stock_price_for_date(ticker, date_str):
    """Obtiene el precio de cierre de una acción para una fecha específica"""
    try:
        # Suprimir warnings de yfinance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Capturar stdout para evitar mensajes de error de yfinance
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                stock = yf.Ticker(ticker)
                
                # Si la fecha es futura, usar la fecha actual
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                today = datetime.now()
                
                if target_date > today:
                    # Si la fecha es futura, usar la fecha más reciente disponible
                    hist = stock.history(period="5d")  # Últimos 5 días
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]  # Último precio disponible
                        return round(price, 2)
                
                # Para fechas pasadas, buscar el precio más cercano
                start_date = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")
                end_date = (target_date + timedelta(days=10)).strftime("%Y-%m-%d")
                
                hist = stock.history(start=start_date, end=end_date)
                if hist.empty:
                    return None
                
                # Buscar el precio más cercano a la fecha objetivo
                hist.index = pd.to_datetime(hist.index.date)
                target_date_pd = pd.to_datetime(date_str)
                
                # Encontrar la fecha más cercana
                closest_date = min(hist.index, key=lambda x: abs(x - target_date_pd))
                price = hist.loc[closest_date, 'Close']
                
                return round(price, 2)
                
            finally:
                # Restaurar stdout
                sys.stdout = old_stdout
    
    except Exception as e:
        # Restaurar stdout en caso de error
        if 'old_stdout' in locals():
            sys.stdout = old_stdout
        print(f"⚠️ Error obteniendo precio para {ticker} en {date_str}: {e}")
        return None

def get_q3_positions_for_year(year, num_investors, selected_investors=None):
    """Obtiene las posiciones de Q3 para un año específico y número de inversores"""
    df = pd.read_csv("top_positions.csv")
    
    # Filtrar por Q3 del año especificado
    q3_positions = df[(df['anio'] == year) & (df['quarter'] == 'Q3')]
    
    if selected_investors is None:
        # Verificar cuántos inversores únicos hay disponibles
        available_investors = q3_positions['inversionista'].nunique()
        max_possible = min(num_investors, available_investors)
        
        # Tomar solo el número especificado de inversores (los primeros N únicos)
        unique_investors = q3_positions['inversionista'].unique()[:max_possible]
    else:
        # Usar los inversores previamente seleccionados
        unique_investors = selected_investors
        max_possible = len([inv for inv in selected_investors if inv in q3_positions['inversionista'].values])
    
    q3_positions = q3_positions[q3_positions['inversionista'].isin(unique_investors)]
    
    print(f"    📋 Q3 {year}: Se solicitaron {num_investors} inversores, usando {max_possible}")
    
    return q3_positions

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

def backtest_strategy(num_investors, start_year, has_relocation=True, selected_investors=None, investor_coverage=None):
    """
    Ejecuta la estrategia de backtesting
    
    Args:
        num_investors: Número de inversores a seguir
        start_year: Año de inicio (int)
        has_relocation: Si permite rebalanceo o no
        selected_investors: Lista de inversores específicos a usar (opcional)
        investor_coverage: Datos de cobertura de inversores (opcional)
    """
    
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
    print(f"💸 Redistribuir inversores fallidos: {'Sí' if REDISTRIBUTE_FAILED else 'No (mantener en efectivo)'}\n")
    
    initial_capital = 10_000_000
    current_capital = initial_capital
    
    # Inicializar capital individual por inversor (para modo sin rebalanceo)
    investor_capitals = {}
    capital_per_investor = initial_capital / actual_num_investors
    for investor in selected_investors:
        investor_capitals[investor] = capital_per_investor
    
    print(f"💰 Capital inicial por inversor: ${capital_per_investor:,.2f}\n")
    
    # Comenzar comprando posiciones Q3 del año anterior al start_year
    buy_year = start_year - 1
    current_year = start_year
    
    portfolio = {}  # {ticker: {'shares': shares, 'investor': investor_name}}
    transactions = []  # Historial de transacciones
    yearly_performance = []  # Performance anual
    
    while current_year <= 2025:  # Incluir 2025 para comprar posiciones
        print(f"📅 Procesando año {current_year}")
        
        # ===== VENTA (31 de diciembre del año anterior) =====
        sell_date = f"{current_year-1}-12-31"
        total_sale_value = 0
        
        if portfolio:  # Si tenemos posiciones
            print(f"  💸 Vendiendo posiciones el {sell_date}")
            
            if has_relocation:
                # CON REBALANCEO: Todas las ventas van a un pool común
                total_sale_value = 0
                for ticker, position in portfolio.items():
                    sell_price = get_stock_price_for_date(ticker, sell_date)
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
                            'inversionista': position['investor']
                        })
                        
                        print(f"    📤 Vendido {position['shares']:.0f} acciones de {ticker} a ${sell_price:.2f} = ${sale_value:,.2f}")
                
                current_capital = total_sale_value
                print(f"  💰 Capital total después de ventas: ${current_capital:,.2f}")
                
            else:
                # SIN REBALANCEO: Cada venta va al capital individual del inversor
                for ticker, position in portfolio.items():
                    sell_price = get_stock_price_for_date(ticker, sell_date)
                    if sell_price:
                        sale_value = position['shares'] * sell_price
                        investor = position['investor']
                        
                        # Actualizar capital individual del inversor
                        investor_capitals[investor] = sale_value
                        
                        transactions.append({
                            'fecha': sell_date,
                            'accion': 'VENTA',
                            'ticker': ticker,
                            'precio': sell_price,
                            'shares': position['shares'],
                            'valor': sale_value,
                            'inversionista': investor
                        })
                        
                        print(f"    📤 {investor}: Vendido {position['shares']:.0f} acciones de {ticker} a ${sell_price:.2f} = ${sale_value:,.2f}")
                
                # Calcular capital total sumando todos los capitales individuales
                current_capital = sum(investor_capitals.values())
                print(f"  💰 Capital total después de ventas: ${current_capital:,.2f}")
                print(f"    📊 Capital por inversor:")
                for investor, capital in investor_capitals.items():
                    print(f"      - {investor}: ${capital:,.2f}")
            
            portfolio = {}  # Limpiar portfolio
        
        # ===== COMPRA (1 de enero) =====
        buy_date = f"{current_year}-01-01"
        
        # Obtener posiciones Q3 del año anterior
        q3_positions = get_q3_positions_for_year(buy_year, actual_num_investors, selected_investors)
        
        if q3_positions.empty:
            print(f"  ⚠️ No hay posiciones Q3 para {buy_year}")
            current_year += 1
            buy_year += 1
            continue
        
        print(f"  🛒 Comprando posiciones Q3 de {buy_year} el {buy_date}")
        
        # Verificar qué acciones están disponibles
        valid_positions = []
        failed_positions = []
        
        print(f"    🔍 Analizando {len(q3_positions)} posiciones Q3 de {buy_year}:")
        for _, position in q3_positions.iterrows():
            ticker = position['accion']
            investor = position['inversionista']
            print(f"      - {investor}: {ticker}", end="")
            
            buy_price = get_stock_price_for_date(ticker, buy_date)
            if buy_price and buy_price > 0:
                valid_positions.append({
                    'ticker': ticker,
                    'investor': investor,
                    'price': buy_price
                })
                print(f" ✅ ${buy_price:.2f}")
            else:
                failed_positions.append({
                    'ticker': ticker,
                    'investor': investor
                })
                print(f" ❌ Sin precio")
        
        print(f"    📊 Total: {len(q3_positions)} posiciones, {len(valid_positions)} válidas, {len(failed_positions)} fallidas")
        
        # Determinar estrategia de capital
        if valid_positions:
            if has_relocation:
                # CON REBALANCEO: Redistribuir todo el capital igualmente entre posiciones válidas
                total_capital_to_invest = current_capital
                capital_per_valid_position = total_capital_to_invest / len(valid_positions)
                
                if failed_positions:
                    print(f"    🔄 REBALANCEO: Redistribuyendo capital de {len(failed_positions)} inversores fallidos entre {len(valid_positions)} válidos")
                
                print(f"    💰 Capital total a invertir: ${total_capital_to_invest:,.2f}")
                print(f"    💰 Capital por posición válida: ${capital_per_valid_position:,.2f}")
                
                for pos in valid_positions:
                    ticker = pos['ticker']
                    investor = pos['investor']
                    buy_price = pos['price']
                    
                    shares_to_buy = capital_per_valid_position / buy_price
                    purchase_value = shares_to_buy * buy_price
                    
                    portfolio[ticker] = {
                        'shares': shares_to_buy,
                        'investor': investor,
                        'buy_price': buy_price
                    }
                    
                    transactions.append({
                        'fecha': buy_date,
                        'accion': 'COMPRA',
                        'ticker': ticker,
                        'precio': buy_price,
                        'shares': shares_to_buy,
                        'valor': purchase_value,
                        'inversionista': investor
                    })
                    
                    print(f"    📥 Comprado {shares_to_buy:.0f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f} ({investor})")
            
            else:
                # SIN REBALANCEO: Cada inversor invierte solo su propio capital
                print(f"    💰 SIN REBALANCEO: Cada inversor invierte su capital individual")
                
                # Mostrar capital disponible por inversor
                for investor in selected_investors:
                    capital = investor_capitals.get(investor, 0)
                    print(f"      - {investor}: ${capital:,.2f} disponible")
                
                total_invested = 0
                investors_with_positions = 0
                
                # Procesar inversores válidos
                for pos in valid_positions:
                    ticker = pos['ticker']
                    investor = pos['investor']
                    buy_price = pos['price']
                    
                    # Usar el capital individual del inversor
                    investor_capital = investor_capitals.get(investor, 0)
                    
                    if investor_capital > 0:
                        shares_to_buy = investor_capital / buy_price
                        purchase_value = shares_to_buy * buy_price
                        
                        # Actualizar capital del inversor (ahora está invertido)
                        investor_capitals[investor] = 0  # Capital ahora en acciones
                        
                        portfolio[ticker] = {
                            'shares': shares_to_buy,
                            'investor': investor,
                            'buy_price': buy_price
                        }
                        
                        transactions.append({
                            'fecha': buy_date,
                            'accion': 'COMPRA',
                            'ticker': ticker,
                            'precio': buy_price,
                            'shares': shares_to_buy,
                            'valor': purchase_value,
                            'inversionista': investor
                        })
                        
                        total_invested += purchase_value
                        investors_with_positions += 1
                        
                        print(f"    📥 {investor}: Comprado {shares_to_buy:.0f} acciones de {ticker} a ${buy_price:.2f} = ${purchase_value:,.2f}")
                    else:
                        print(f"    ⚠️ {investor}: Sin capital disponible para {ticker}")
                
                # Manejar inversores fallidos según configuración
                uninvested_capital = 0
                failed_investors_capital = 0
                
                for investor in selected_investors:
                    if investor not in [pos['investor'] for pos in valid_positions]:
                        failed_investor_capital = investor_capitals.get(investor, 0)
                        failed_investors_capital += failed_investor_capital
                        
                        if REDISTRIBUTE_FAILED and valid_positions:
                            # REDISTRIBUIR: Repartir el dinero del inversor fallido entre los válidos
                            capital_per_valid_investor = failed_investor_capital / len(valid_positions)
                            print(f"    � {investor}: Redistribuyendo ${failed_investor_capital:,.2f} entre {len(valid_positions)} inversores válidos")
                            
                            # Agregar dinero extra a cada posición válida
                            for ticker, position in portfolio.items():
                                if ticker in [p['ticker'] for p in valid_positions]:
                                    extra_shares = capital_per_valid_investor / position['buy_price']
                                    extra_value = extra_shares * position['buy_price']
                                    
                                    # Actualizar posición
                                    portfolio[ticker]['shares'] += extra_shares
                                    
                                    # Agregar transacción extra
                                    transactions.append({
                                        'fecha': buy_date,
                                        'accion': 'COMPRA_REDISTRIBUIDA',
                                        'ticker': ticker,
                                        'precio': position['buy_price'],
                                        'shares': extra_shares,
                                        'valor': extra_value,
                                        'inversionista': f"{position['investor']} (de {investor})"
                                    })
                                    
                                    total_invested += extra_value
                                    print(f"    📥 Extra para {position['investor']}: {extra_shares:.0f} acciones de {ticker} = ${extra_value:,.2f}")
                            
                            # El capital del inversor fallido se redistribuyó
                            investor_capitals[investor] = 0
                        else:
                            # MANTENER EN EFECTIVO: El dinero queda sin invertir
                            uninvested_capital += failed_investor_capital
                            print(f"    💵 {investor}: ${failed_investor_capital:,.2f} queda sin invertir (sin posición Q3)")
                
                print(f"    💰 Total invertido: ${total_invested:,.2f}")
                print(f"    💵 Total sin invertir: ${uninvested_capital:,.2f}")
                print(f"    👥 Inversores con posiciones: {investors_with_positions}/{actual_num_investors}")
                
                # El capital actual es la suma de lo invertido más lo no invertido
                current_capital = total_invested + uninvested_capital
        else:
            print(f"    ❌ No se pudieron comprar acciones para ningún inversor en {buy_date}")
        
        # Avanzar al siguiente año
        current_year += 1
        buy_year += 1
    
    # ===== VENTA FINAL DE LAS POSICIONES 2025 =====
    # Vender las posiciones compradas en enero 2025 usando el precio más reciente
    if portfolio:
        print(f"\n🏁 Venta final de posiciones 2025")
        
        if has_relocation:
            # CON REBALANCEO: Todas las ventas van a un pool común
            total_final_value = 0
            for ticker, position in portfolio.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        final_price = hist['Close'].iloc[-1]
                        final_value = position['shares'] * final_price
                        total_final_value += final_value
                        
                        transactions.append({
                            'fecha': '2025-ACTUAL',
                            'accion': 'VENTA_FINAL',
                            'ticker': ticker,
                            'precio': final_price,
                            'shares': position['shares'],
                            'valor': final_value,
                            'inversionista': position['investor']
                        })
                        
                        print(f"    📤 {ticker}: {position['shares']:.2f} acciones × ${final_price:.2f} = ${final_value:,.2f} ({position['investor']})")
                    else:
                        print(f"    ⚠️ No se pudo obtener precio actual para {ticker}")
                except Exception as e:
                    print(f"    ⚠️ Error obteniendo precio actual para {ticker}: {e}")
            
            current_capital = total_final_value
            
        else:
            # SIN REBALANCEO: Cada venta va al capital individual del inversor
            for ticker, position in portfolio.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        final_price = hist['Close'].iloc[-1]
                        final_value = position['shares'] * final_price
                        investor = position['investor']
                        
                        # Actualizar capital individual del inversor
                        investor_capitals[investor] = final_value
                        
                        transactions.append({
                            'fecha': '2025-ACTUAL',
                            'accion': 'VENTA_FINAL',
                            'ticker': ticker,
                            'precio': final_price,
                            'shares': position['shares'],
                            'valor': final_value,
                            'inversionista': investor
                        })
                        
                        print(f"    📤 {investor}: {ticker} - {position['shares']:.2f} acciones × ${final_price:.2f} = ${final_value:,.2f}")
                    else:
                        print(f"    ⚠️ No se pudo obtener precio actual para {ticker}")
                except Exception as e:
                    print(f"    ⚠️ Error obteniendo precio actual para {ticker}: {e}")
            
            # Sumar todo el capital (invertido + no invertido)
            current_capital = sum(investor_capitals.values())
            
            print(f"\n  💰 Capital final por inversor:")
            for investor, capital in investor_capitals.items():
                print(f"    - {investor}: ${capital:,.2f}")
        
        print(f"  💰 Valor final del portfolio: ${current_capital:,.2f}")
    
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
    
    # Guardar transacciones
    df_transactions = pd.DataFrame(transactions)
    df_transactions.to_csv("backtest_transactions.csv", index=False)
    print(f"💾 Transacciones guardadas en backtest_transactions.csv")
    
    return {
        'capital_inicial': initial_capital,
        'capital_final': final_portfolio_value,
        'ganancia': total_return,
        'retorno_pct': return_percentage,
        'transacciones': transactions
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 CONFIGURACIÓN DEL BACKTEST")
    print("=" * 60)
    print(f"📊 Número de inversores: {NUM_INVESTORS}")
    print(f"📅 Año de inicio: {START_YEAR}")
    print(f"🔄 Rebalanceo: {'Activado' if HAS_RELOCATION else 'Desactivado'}")
    print(f"💸 Redistribuir fallidos: {'Activado' if REDISTRIBUTE_FAILED else 'Desactivado (mantener en efectivo)'}")
    print(f"🎲 Semilla aleatoria: {RANDOM_SEED}")
    print("=" * 60)
    print()
    
    # Ejecutar backtest usando la configuración definida arriba
    resultado = backtest_strategy(
        num_investors=NUM_INVESTORS,
        start_year=START_YEAR,
        has_relocation=HAS_RELOCATION
    )
    
    print(f"\n✅ Backtest completado!")
    print(f"📄 Para cambiar la configuración, edita las variables al inicio del archivo:")
