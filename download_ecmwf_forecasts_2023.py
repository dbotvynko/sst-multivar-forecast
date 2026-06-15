"""
Téléchargement des forecasts ECMWF journaliers 2023 depuis Copernicus CDS
- Vent : u10 et v10 (TIGGE — ensemble forecasts archivés)
- Vagues : hauteur significative, période, direction (seasonal forecasts)

Prérequis :
    pip install cdsapi xarray netCDF4
    Fichier ~/.cdsapirc :
        url: https://cds.climate.copernicus.eu/api/v2
        key: <UID>:<API-KEY>
"""

import cdsapi
import os

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = cdsapi.Client()

# ---------------------------------------------------------------------------
# 1. VENT DAILY — u10 et v10 via TIGGE (ensemble forecasts archivés ECMWF)
#    Téléchargement mois par mois pour éviter les timeouts
# ---------------------------------------------------------------------------
print("Téléchargement des forecasts journaliers de vent (u10, v10) — TIGGE...")

MONTHS = {
    "01": list(range(1, 32)),
    "02": list(range(1, 29)),
    "03": list(range(1, 32)),
    "04": list(range(1, 31)),
    "05": list(range(1, 32)),
    "06": list(range(1, 31)),
    "07": list(range(1, 32)),
    "08": list(range(1, 32)),
    "09": list(range(1, 31)),
    "10": list(range(1, 32)),
    "11": list(range(1, 31)),
    "12": list(range(1, 32)),
}

for month, days in MONTHS.items():
    out_file = f"{OUTPUT_DIR}/wind_uv10_daily_2023_{month}.nc"
    if os.path.exists(out_file):
        print(f"  -> {out_file} déjà présent, skip.")
        continue

    print(f"  Mois {month}...")
    client.retrieve(
        "tigge",
        {
            "originating_centre": "ecmwf",
            "system": "operational",
            "variable": [
                "10_metre_u_wind_component",
                "10_metre_v_wind_component",
            ],
            "product_type": "ensemble_mean",   # ou "control_forecast" pour 1 membre
            "year": "2023",
            "month": month,
            "day": [f"{d:02d}" for d in days],
            "time": ["00:00", "12:00"],        # 2 runs par jour
            "leadtime_hour": ["24"],           # forecast à +24h (J+1)
            "format": "netcdf",
        },
        out_file,
    )
    print(f"  -> Sauvegardé : {out_file}")

# ---------------------------------------------------------------------------
# 2. VAGUES — moyennes mensuelles (seasonal forecast SEAS5)
#    Note : les vagues daily ne sont pas dans TIGGE.
#    Pour du daily, utiliser CMEMS (copernicusmarine) à la place.
# ---------------------------------------------------------------------------
print("\nTéléchargement des forecasts de vagues (monthly mean — SEAS5)...")

client.retrieve(
    "seasonal-original-single-levels",
    {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": [
            "significant_height_of_combined_wind_waves_and_swell",
            "mean_wave_period",
            "mean_wave_direction",
        ],
        "product_type": "monthly_mean",
        "year": "2023",
        "month": [f"{m:02d}" for m in range(1, 13)],
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "format": "netcdf",
    },
    f"{OUTPUT_DIR}/waves_forecast_2023.nc",
)
print(f"  -> Sauvegardé : {OUTPUT_DIR}/waves_forecast_2023.nc")

# ---------------------------------------------------------------------------
# 3. Vérification rapide
# ---------------------------------------------------------------------------
import xarray as xr

for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(".nc"):
        path = f"{OUTPUT_DIR}/{fname}"
        ds = xr.open_dataset(path)
        print(f"\n{fname}: {list(ds.data_vars)} | time: {ds.time.values[0]} -> {ds.time.values[-1]}")
        ds.close()
