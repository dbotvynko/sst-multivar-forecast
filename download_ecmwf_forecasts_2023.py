"""
Téléchargement des forecasts ECMWF 2023 depuis Copernicus Climate Data Store (CDS)
- Vent : composantes U et V à 10m (u10, v10)
- Vagues : hauteur significative, période et direction

Prérequis :
    pip install cdsapi
    Créer ~/.cdsapirc avec vos identifiants CDS :
        url: https://cds.climate.copernicus.eu/api/v2
        key: <UID>:<API-KEY>
"""

import cdsapi
import os

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = cdsapi.Client()

# ---------------------------------------------------------------------------
# 1. VENT — u10 et v10 depuis les forecasts saisonniers ECMWF (SEAS5)
#    Dataset : seasonal-original-single-levels
#    Résolution : ~1° | Échéances : jusqu'à 6 mois
# ---------------------------------------------------------------------------
print("Téléchargement des forecasts de vent (u10, v10)...")

client.retrieve(
    "seasonal-original-single-levels",
    {
        "originating_centre": "ecmwf",
        "system": "51",                         # SEAS5 (dernier système ECMWF)
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "product_type": "monthly_mean",
        "year": "2023",
        "month": [f"{m:02d}" for m in range(1, 13)],
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "format": "netcdf",
    },
    f"{OUTPUT_DIR}/wind_uv10_forecast_2023.nc",
)

print(f"  -> Sauvegardé dans {OUTPUT_DIR}/wind_uv10_forecast_2023.nc")

# ---------------------------------------------------------------------------
# 2. VAGUES — depuis les forecasts saisonniers ECMWF
#    Variables : hauteur significative, période moyenne, direction moyenne
# ---------------------------------------------------------------------------
print("Téléchargement des forecasts de vagues...")

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

print(f"  -> Sauvegardé dans {OUTPUT_DIR}/waves_forecast_2023.nc")

# ---------------------------------------------------------------------------
# 3. (Optionnel) Vérification rapide des fichiers téléchargés
# ---------------------------------------------------------------------------
import xarray as xr

for fname in ["wind_uv10_forecast_2023.nc", "waves_forecast_2023.nc"]:
    path = f"{OUTPUT_DIR}/{fname}"
    if os.path.exists(path):
        ds = xr.open_dataset(path)
        print(f"\n{fname}:")
        print(ds)
        ds.close()
