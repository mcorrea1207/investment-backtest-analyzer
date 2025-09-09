import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/135.0.0.0 Safari/537.36"
}


# ---------------------------
# Obtener lista de inversionistas
# ---------------------------
def get_manager_codes():
    """Obtiene la lista de inversionistas y sus códigos desde la home de dataroma"""
    url = "https://www.dataroma.com/m/home.php"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    codes = []

    for link in soup.find_all("a", href=True):
        if "holdings.php?m=" in link["href"]:
            name = link.text.strip()
            code = link["href"].split("=")[-1]
            codes.append({"inversionista": name, "codigo": code})

    return pd.DataFrame(codes)


# ---------------------------
# Helpers
# ---------------------------
def _sanitize_token(tok: str) -> str:
    """Limpia un token eliminando símbolos y espacios extra"""
    return re.sub(r"[^A-Za-z0-9]", "", tok).strip()


def _looks_like_ticker(tok: str) -> bool:
    """Heurística para decidir si un token es un ticker"""
    if not tok:
        return False
    return tok.isupper() and 1 <= len(tok) <= 5


def _extract_quarter_from_row(cols):
    """Devuelve año y quarter de la fila"""
    # El primer campo contiene "2025 \xa0 Q2" donde \xa0 es un espacio no-rompible
    first_col = cols[0].text.strip()
    
    # Reemplazar espacios no-rompibles por espacios normales
    first_col = first_col.replace('\xa0', ' ')
    
    # Dividir por espacios
    parts = first_col.split()
    if len(parts) >= 2:
        anio = parts[0].strip()
        quarter = parts[1].strip()
    else:
        anio = first_col
        quarter = ""
    
    return anio, quarter


def _extract_first_ticker_from_cell(cell_text: str) -> str:
    """
    Extrae el primer token que parece ticker de la celda de la acción.
    Ejemplo: "FCNCA\nFirst Citizens Bancshares\n6.89% of portfolio" → "FCNCA"
    """
    # Dividir por líneas primero, luego por espacios
    lines = cell_text.strip().split('\n')
    for line in lines:
        tokens = line.split()
        for tok in tokens:
            cleaned_tok = _sanitize_token(tok)
            if _looks_like_ticker(cleaned_tok):
                return cleaned_tok
    return ""


def _extract_pct_from_cell(cell_text: str):
    """Extrae el porcentaje del portafolio desde el texto de la celda.
    Busca patrones como "6.89% of portfolio" y devuelve un float (6.89).
    """
    if not cell_text:
        return None
    # Normalizar espacios
    text = ' '.join(cell_text.split())
    # Regex tolerante a mayúsculas/minúsculas y espacios
    m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*%\s*of\s*portfolio", text, flags=re.IGNORECASE)
    if not m:
        return None
    num = m.group(1).replace(',', '')
    try:
        return float(num)
    except Exception:
        return None


def _extract_company_name_from_cell(cell_text: str):
    """Intenta extraer el nombre de la compañía (2ª línea de la celda)."""
    if not cell_text:
        return None
    lines = cell_text.strip().split('\n')
    if len(lines) >= 2:
        return lines[1].strip()
    return None


def _clean_investor_name(name: str) -> str:
    """
    Limpia el nombre del inversionista removiendo información extra
    Ejemplo: "Bill Nygren - Oakmark Select Fund Updated 29 Aug 2025" → "Bill Nygren"
    """
    # Remover "Updated" y todo lo que viene después
    if "Updated" in name:
        name = name.split("Updated")[0].strip()
    
    # Si hay " - ", tomar solo la parte antes del guión
    if " - " in name:
        name = name.split(" - ")[0].strip()
    
    return name


def _parse_portfolio_value(value_text: str):
    """Convierte texto como "$257.5 B" a dólares numéricos (float) y devuelve (usd, raw).
    Soporta sufijos B, M, K, T.
    """
    if not value_text:
        return None, None
    raw = ' '.join(value_text.split())
    m = re.search(r"\$\s*([0-9]+(?:[\.,][0-9]+)?)\s*([BMKTbmkt])", raw)
    mult = 1.0
    if m:
        num = float(m.group(1).replace(',', ''))
        unit = m.group(2).upper()
        if unit == 'K': mult = 1e3
        elif unit == 'M': mult = 1e6
        elif unit == 'B': mult = 1e9
        elif unit == 'T': mult = 1e12
        return num * mult, raw
    # Fallback: intentar sin sufijo, solo número
    m2 = re.search(r"\$\s*([0-9]+(?:[\.,][0-9]+)?)", raw)
    if m2:
        try:
            return float(m2.group(1).replace(',', '')), raw
        except Exception:
            return None, raw
    return None, raw


# ---------------------------
# Scraping historial top posición (TOP 1)
# ---------------------------

def get_top_position_history(code, investor_name, min_year: int | None = None):
    """
    Extrae la mayor posición por quarter desde la página histórica p_hist.php
    Permite filtrar opcionalmente por año mínimo; si min_year es None, trae todos los años disponibles.
    """
    url = f"https://www.dataroma.com/m/hist/p_hist.php?f={code}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"⚠️ Error {response.status_code} al descargar {investor_name}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"⚠️ No se encontró tabla histórica para {investor_name}")
        return pd.DataFrame()

    rows = table.find_all("tr")
    result = []

    for i, row in enumerate(rows[1:]):  # saltar encabezado
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        anio, quarter = _extract_quarter_from_row(cols)
        
        # Filtrado opcional por año mínimo
        try:
            year_int = int(anio)
        except (ValueError, TypeError):
            continue
        if min_year is not None and year_int < min_year:
            continue
            
        # Columna 3 = top position (incluye ticker y "% of portfolio")
        accion_raw = cols[2].get_text(separator="\n").strip()
        ticker = _extract_first_ticker_from_cell(accion_raw)
        pct = _extract_pct_from_cell(accion_raw)
        
        # Usar solo el ticker como acción
        accion = ticker if ticker else accion_raw.split('\n')[0].strip()
        
        # Limpiar el nombre del inversionista
        clean_name = _clean_investor_name(investor_name)

        result.append({
            "anio": anio,
            "quarter": quarter,
            "accion": accion,
            "porcentaje_portafolio": pct,
            "inversionista": clean_name,
            "codigo_inversionista": code
        })

    return pd.DataFrame(result)


# ---------------------------
# Scraping historial TOP N posiciones (e.g., Top 5/Top 20)
# ---------------------------

def get_top_n_positions_history(code: str, investor_name: str, n: int = 20, min_year: int | None = None) -> pd.DataFrame:
    """Extrae las Top N posiciones por quarter desde p_hist.php.
    Devuelve filas por (año, quarter, rank) con ticker, % y valor de portafolio.
    Si min_year es None, incluye todos los años disponibles.
    """
    url = f"https://www.dataroma.com/m/hist/p_hist.php?f={code}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"⚠️ Error {response.status_code} al descargar {investor_name}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"⚠️ No se encontró tabla histórica para {investor_name}")
        return pd.DataFrame()

    rows = table.find_all("tr")
    result = []

    for i, row in enumerate(rows[1:]):  # saltar encabezado
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        anio, quarter = _extract_quarter_from_row(cols)
        try:
            year_int = int(anio)
        except (ValueError, TypeError):
            continue
        if min_year is not None and year_int < min_year:
            continue

        # Valor de portafolio del periodo (columna 1)
        pv_raw = cols[1].get_text(strip=True)
        pv_usd, _pv_raw_norm = _parse_portfolio_value(pv_raw)

        # Holdings están desde la columna 2 en adelante (0=period,1=value)
        max_idx = min(2 + n, len(cols))
        rank = 1
        for idx in range(2, max_idx):
            cell_text = cols[idx].get_text(separator='\n').strip()
            if not cell_text:
                continue
            ticker = _extract_first_ticker_from_cell(cell_text)
            pct = _extract_pct_from_cell(cell_text)
            accion = ticker if ticker else cell_text.split('\n')[0].strip()
            clean_name = _clean_investor_name(investor_name)

            result.append({
                'anio': anio,
                'quarter': quarter,
                'rank': rank,
                'accion': accion,
                'porcentaje_portafolio': pct,
                'portfolio_value_usd': pv_usd,
                'inversionista': clean_name,
                'codigo_inversionista': code
            })
            rank += 1

    df = pd.DataFrame(result)
    if not df.empty:
        with pd.option_context('mode.chained_assignment', None):
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce')
        df = df.sort_values(['anio', 'quarter', 'rank'], ascending=[False, True, True]).reset_index(drop=True)
    return df


def get_all_managers_top_n(n: int = 20, min_year: int | None = None) -> pd.DataFrame:
    """Descarga Top N por quarter para todos los managers y concatena en un solo DataFrame.
    Si min_year es None, incluye todos los años disponibles.
    """
    managers = get_manager_codes()
    all_rows = []
    for _, row in managers.iterrows():
        name = row['inversionista']
        code = row['codigo']
        print(f"Descargando Top {n}: {name} ({code})")
        try:
            df_n = get_top_n_positions_history(code, name, n=n, min_year=min_year)
            if not df_n.empty:
                all_rows.append(df_n)
        except Exception as e:
            print(f"  ⚠️ Error con {name} ({code}): {e}")
    if not all_rows:
        return pd.DataFrame()
    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # 1) CSV clásico: Top-1 por quarter (sin columna company)
    managers = get_manager_codes()
    top1_data = []
    for _, row in managers.iterrows():
        name = row["inversionista"]
        code = row["codigo"]
        print(f"Descargando Top 1: {name} ({code})")
        df_hist = get_top_position_history(code, name, min_year=None)
        if not df_hist.empty:
            top1_data.append(df_hist)
    if top1_data:
        df_final = pd.concat(top1_data, ignore_index=True)
        df_final.to_csv("top_positions.csv", index=False)
        print("✅ Exportado a top_positions.csv")
    else:
        print("⚠️ No se descargaron filas para Top 1")

    # 2) CSV extendido: Top-20 por quarter para todos los managers (sin company ni portfolio_value_raw)
    print("\nRecolectando Top 20 para todos los managers (todos los años disponibles)...")
    df_all = get_all_managers_top_n(n=20, min_year=None)
    if not df_all.empty:
        keep_cols = ['anio', 'quarter', 'rank', 'accion', 'porcentaje_portafolio', 'portfolio_value_usd', 'inversionista', 'codigo_inversionista']
        df_all = df_all[keep_cols]
        df_all.to_csv("top_positions_all_clean.csv", index=False)
        print(f"✅ Exportado a top_positions_all_clean.csv (filas: {len(df_all)})")
    else:
        print("⚠️ No se descargaron filas para Top 20")
