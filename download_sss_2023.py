"""
Téléchargement SSS (salinité de surface) journalière 2023 — zone Gulf Stream
Source : CMEMS global physics analysis/forecast (so = sea water salinity, depth=0)

Prérequis : pip install copernicusmarine
Connexion : copernicusmarine login
"""

import os
import copernicusmarine

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSS_FILE = f"{OUTPUT_DIR}/sss_daily_2023.nc"

if os.path.exists(SSS_FILE):
    print(f"Déjà présent : {SSS_FILE}")
else:
    print("Téléchargement SSS (salinité de surface) — Gulf Stream 2023...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["so"],
        minimum_longitude=-79.95, maximum_longitude=-50.35,
        minimum_latitude=24.05, maximum_latitude=39.65,
        minimum_depth=0, maximum_depth=1,
        start_datetime="2023-01-01T00:00:00",
        end_datetime="2023-12-31T00:00:00",
        output_filename=SSS_FILE,
    )
    print(f"Sauvegardé : {SSS_FILE}")
