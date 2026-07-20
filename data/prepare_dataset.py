"""
Prepara el dataset de tipo de cambio USD/PYG (Guaraní paraguayo).

Fuente: Banco Central del Paraguay (BCP) - Cotización Referencial Histórica
        https://www.bcp.gov.py/webapps/web/cotizacion/monedas-historica
        Planillas oficiales descargadas manualmente (una por año, formato
        HTML/Excel: día del mes en filas, mes en columnas).

Años combinados: 2022, 2023, 2024, 2025 (996 observaciones diarias, días
hábiles del mercado cambiario).

Este script documenta el parseo ya realizado (día-por-mes -> formato largo,
con manejo de "ND" como faltante y formato numérico es-PY "7.037,83").
El resultado ya está guardado en data/datos.csv.
"""
import pandas as pd
from pathlib import Path

MESES = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
         "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12}

# Rutas relativas a la ubicación del propio script (no a la máquina donde
# se escribió originalmente), para que funcione en cualquier clon del repo.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUT_PATH = BASE_DIR / "datos.csv"


def parse_year(path, year):
    tables = pd.read_html(path)
    t = tables[1].rename(columns={tables[1].columns[0]: "dia"})
    rows = []
    for _, r in t.iterrows():
        dia = r["dia"]
        for mes_str, mnum in MESES.items():
            val = r.get(mes_str)
            if pd.isna(val) or val == "ND":
                continue
            try:
                fecha = pd.Timestamp(year=year, month=mnum, day=int(dia))
            except ValueError:
                continue
            valor = float(str(val).replace(".", "").replace(",", "."))
            rows.append((fecha, valor))
    return pd.DataFrame(rows, columns=["fecha", "tipo_cambio"])


if __name__ == "__main__":
    dfs = [parse_year(RAW_DIR / f"cotizacion_anual{y}.xls", y) for y in [2022, 2023, 2024, 2025]]
    full = pd.concat(dfs).sort_values("fecha").drop_duplicates(subset="fecha").reset_index(drop=True)
    full.to_csv(OUT_PATH, index=False)
    print("Filas finales:", full.shape)
    print(f"Rango: {full['fecha'].min().date()} a {full['fecha'].max().date()}")
