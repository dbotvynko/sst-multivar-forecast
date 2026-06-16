"""
Téléchargement des données ECMWF journalières 2023

  CMEMS (copernicusmarine) — vagues :
    - SWH  : hauteur significative des vagues         (VHM0)
    - MWP  : période moyenne des vagues               (VTM02)
    - SHWW : hauteur significative mer du vent        (VHM0_WW)
    - MPWW : période moyenne mer du vent              (VTM02_WW)

  CDS ERA5 (cdsapi) — variables atmosphériques :
    - sshf             : flux de chaleur sensible
    - slhf             : flux de chaleur latente
    - MSL              : pression atmosphérique niveau mer
    - neutral wind u/v : composantes du vent neutre à 10m

Prérequis :
    pip install cdsapi copernicusmarine xarray netCDF4

    ~/.cdsapirc :
        url: https://cds.climate.copernicus.eu/api/v2
        key: <UID>:<API-KEY>

    Compte CMEMS : https://marine.copernicus.eu  (login via copernicusmarine login)
"""

import cdsapi
import copernicusmarine
import os
import xarray as xr

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# PARTIE 1 — VAGUES via CMEMS
# Dataset : global-reanalysis-wav-001-032 (réanalyse vagues ECMWF, daily)
# =============================================================================
print("=" * 60)
print("CMEMS — Vagues journalières 2023")
print("=" * 60)

# VHM0_WW et VTM02_WW disponibles uniquement dans le produit analysis/forecast
# On utilise deux datasets séparés :
#   - réanalyse (my) pour VHM0 et VTM02
#   - analysis/forecast (anfc) pour VHM0_WW et VTM02_WW

WAVE_OUTPUT_REAN  = f"{OUTPUT_DIR}/waves_daily_2023_rean.nc"
WAVE_OUTPUT_ANFC  = f"{OUTPUT_DIR}/waves_daily_2023_anfc.nc"

if not os.path.exists(WAVE_OUTPUT_REAN):
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
        variables=["VHM0", "VTM02"],
        start_datetime="2023-01-01T00:00:00",
        end_datetime="2023-12-31T21:00:00",
        output_filename=WAVE_OUTPUT_REAN,
    )
    print(f"  -> Sauvegardé : {WAVE_OUTPUT_REAN}")
else:
    print(f"  -> Déjà présent : {WAVE_OUTPUT_REAN}")

if not os.path.exists(WAVE_OUTPUT_ANFC):
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        variables=["VHM0_WW", "VTM01_WW"],   # VTM02_WW n'existe pas → VTM01_WW (période mer du vent)
        start_datetime="2023-01-01T00:00:00",
        end_datetime="2023-12-31T21:00:00",
        output_filename=WAVE_OUTPUT_ANFC,
    )
    print(f"  -> Sauvegardé : {WAVE_OUTPUT_ANFC}")
else:
    print(f"  -> Déjà présent : {WAVE_OUTPUT_ANFC}")

# =============================================================================
# PARTIE 2 — VARIABLES ATMOSPHÉRIQUES via CDS (ERA5)
# Dataset : reanalysis-era5-single-levels, résolution ~31km, daily
# Téléchargement mois par mois pour éviter les timeouts
# =============================================================================
print("\n" + "=" * 60)
print("CDS ERA5 — Variables atmosphériques journalières 2023")
print("=" * 60)

client = cdsapi.Client()

ATMO_VARS = [
    "surface_sensible_heat_flux",           # sshf
    "surface_latent_heat_flux",             # slhf
    "mean_sea_level_pressure",              # MSL
    "10m_u_component_of_neutral_wind",      # neutral wind u
    "10m_v_component_of_neutral_wind",      # neutral wind v
]

# Tous les mois de 2023
MONTHS = {
    "01": [f"{d:02d}" for d in range(1, 32)],
    "02": [f"{d:02d}" for d in range(1, 29)],
    "03": [f"{d:02d}" for d in range(1, 32)],
    "04": [f"{d:02d}" for d in range(1, 31)],
    "05": [f"{d:02d}" for d in range(1, 32)],
    "06": [f"{d:02d}" for d in range(1, 31)],
    "07": [f"{d:02d}" for d in range(1, 32)],
    "08": [f"{d:02d}" for d in range(1, 32)],
    "09": [f"{d:02d}" for d in range(1, 31)],
    "10": [f"{d:02d}" for d in range(1, 32)],
    "11": [f"{d:02d}" for d in range(1, 31)],
    "12": [f"{d:02d}" for d in range(1, 32)],
}

for month, days in MONTHS.items():
    out_file = f"{OUTPUT_DIR}/atmo_daily_2023_{month}.nc"
    if os.path.exists(out_file):
        print(f"  -> Mois {month} déjà présent, skip.")
        continue

    # Nettoyer un éventuel dossier temporaire laissé par un run précédent
    import shutil
    extract_dir = f"{OUTPUT_DIR}/_tmp_atmo_{month}"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"  Téléchargement mois {month}...")
    raw_file = f"{OUTPUT_DIR}/atmo_raw_2023_{month}.download"
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ATMO_VARS,
            "year": "2023",
            "month": month,
            "day": days,
            "time": [f"{h:02d}:00" for h in range(0, 24)],  # toutes les heures
            "format": "netcdf",
        },
        raw_file,
    )

    # CDS renvoie parfois un zip contenant plusieurs .nc (instant / accum)
    # quand on mélange des variables de types différents (sshf/slhf = accum,
    # msl/u10n/v10n = instant). On détecte et on fusionne dans ce cas.
    import zipfile

    if zipfile.is_zipfile(raw_file):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(raw_file) as zf:
            zf.extractall(extract_dir)
        nc_files = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if f.endswith(".nc")
        ]
        # .load() force la lecture en mémoire et libère les handles de fichier,
        # nécessaire pour pouvoir supprimer le dossier ensuite (NFS notamment)
        sub_datasets = []
        for f in nc_files:
            with xr.open_dataset(f) as ds_tmp:
                sub_datasets.append(ds_tmp.load())
        ds_merged = xr.merge(sub_datasets)
        ds_merged.to_netcdf(out_file)
        ds_merged.close()
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.remove(raw_file)
    else:
        os.rename(raw_file, out_file)

    print(f"  -> Sauvegardé : {out_file}")

# =============================================================================
# PARTIE 3 — Calcul de la vitesse du vent neutre (norme) et résumé daily
# =============================================================================
print("\n" + "=" * 60)
print("Post-traitement — Agrégation en daily mean")
print("=" * 60)

import numpy as np

# Vagues : merger réanalyse + anfc puis resample daily
if os.path.exists(WAVE_OUTPUT_REAN) and os.path.exists(WAVE_OUTPUT_ANFC):
    ds_rean = xr.open_dataset(WAVE_OUTPUT_REAN)
    ds_anfc = xr.open_dataset(WAVE_OUTPUT_ANFC)
    # Interpoler anfc sur la grille réanalyse (0.2°) si nécessaire
    ds_anfc_interp = ds_anfc.interp(longitude=ds_rean.longitude, latitude=ds_rean.latitude)
    ds_wav = xr.merge([ds_rean, ds_anfc_interp])
    ds_wav_daily = ds_wav.resample(time="1D").mean()
    out_wav_daily = f"{OUTPUT_DIR}/waves_daily_mean_2023.nc"
    ds_wav_daily.to_netcdf(out_wav_daily)
    print(f"  Vagues daily mean -> {out_wav_daily}")

# Atmosphérique : concaténer tous les mois et resample daily
atmo_files = sorted([
    f"{OUTPUT_DIR}/{f}" for f in os.listdir(OUTPUT_DIR)
    if f.startswith("atmo_daily_2023_") and f.endswith(".nc")
])

if atmo_files:
    ds_atmo = xr.open_mfdataset(atmo_files, combine="by_coords")

    # Vitesse du vent neutre à 10m (norme)
    ds_atmo["wind_speed_neutral"] = np.sqrt(
        ds_atmo["u10n"] ** 2 + ds_atmo["v10n"] ** 2
    )

    ds_atmo_daily = ds_atmo.resample(time="1D").mean()
    out_atmo = f"{OUTPUT_DIR}/atmo_daily_mean_2023.nc"
    ds_atmo_daily.to_netcdf(out_atmo)
    print(f"  Atmosphérique daily mean -> {out_atmo}")
    print(f"\n  Variables disponibles : {list(ds_atmo_daily.data_vars)}")

print("\nTerminé !")
