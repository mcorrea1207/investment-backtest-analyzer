import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/110.0 Safari/537.36"
}

def get_buffett_holdings_mobile():
    url = "https://www.dataroma.com/m/holdings.php?m=BRK"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("⚠️ No se encontró la tabla")
        return pd.DataFrame()

    rows = table.find_all("tr")
    data = []
    for row in rows[1:]:  # salteamos encabezado
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 3:
            continue
        accion = cols[1]
        try:
            peso = float(cols[2])
        except ValueError:
            continue
        data.append({
            "accion": accion,
            "peso": peso,
            "inversionista": "Warren Buffett (BRK)"
        })

    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ No se encontraron datos")
        return df
    return df.sort_values("peso", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df_buffett = get_buffett_holdings_mobile()
    print(df_buffett.head(10))  # Top 10
    df_buffett.to_csv("buffett_holdings.csv", index=False)
    print("✅ Exportado a buffett_holdings.csv")
