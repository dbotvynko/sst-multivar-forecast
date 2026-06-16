"""
Fusion de toutes les variables 2023 (Gulf Stream) dans un seul fichier NetCDF.

Étapes :
  1. Charger les données déjà téléchargées (vagues, atmo, SST, SLA, ugos, vgos)
  2. Télécharger les variables manquantes (SSS, SSH/adt) via CMEMS
  3. Calculer DOS (densité) et les gradients (SST, SSS, SLA)
  4. Régrider tout sur une grille commune 0.25°
  5. Découper sur la zone Gulf Stream
  6. Sauvegarder un seul fichier NetCDF fusionné

Prérequis :
    pip install xarray copernicusmarine gsw netCDF4 scipy
"""

import os
import numpy as np
import xarray as xr
import copernicusmarine
import gsw

# =============================================================================
# CONFIG — À ADAPTER
# =============================================================================

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_OUTPUT = f"{OUTPUT_DIR}/merged_gulfstream_2023_025deg.nc"

# Zone Gulf Stream (ajuste si besoin)
LON_MIN, LON_MAX = -82.0, -40.0
LAT_MIN, LAT_MAX = 20.0, 50.0
TARGET_RES = 0.25  # degrés

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# --- Chemins des fichiers déjà téléchargés (À REMPLIR) -----------------------
EXISTING_FILES = {
    "waves_rean": f"{OUTPUT_DIR}/waves_daily_2023_rean.nc",       # VHM0, VTM02
    "waves_anfc": f"{OUTPUT_DIR}/waves_daily_2023_anfc.nc",       # VHM0_WW, VTM01_WW
    "atmo_glob": f"{OUTPUT_DIR}/atmo_daily_mean_2023.nc",         # sshf, slhf, msl, wind_speed_neutral (si déjà généré)
    "atmo_months": sorted(
        f"{OUTPUT_DIR}/{f}" for f in os.listdir(OUTPUT_DIR)
        if f.startswith("atmo_daily_2023_") and f.endswith(".nc")
    ),
    "sst": "CHEMIN_VERS_TON_FICHIER_SST.nc",       # <-- à remplir
    "sla": "CHEMIN_VERS_TON_FICHIER_SLA.nc",       # <-- à remplir
    "ugos_vgos": "CHEMIN_VERS_TON_FICHIER_UGOS_VGOS.nc",  # <-- à remplir
}

# =============================================================================
# 1. CHARGER LES DONNÉES DÉJÀ TÉLÉCHARGÉES
# =============================================================================
print("=" * 60)
print("1. Chargement des données existantes")
print("=" * 60)

datasets = []

# Vagues
if os.path.exists(EXISTING_FILES["waves_rean"]):
    ds_wav_r = xr.open_dataset(EXISTING_FILES["waves_rean"])
    datasets.append(ds_wav_r)
    print("  Vagues (réanalyse) OK")

if os.path.exists(EXISTING_FILES["waves_anfc"]):
    ds_wav_a = xr.open_dataset(EXISTING_FILES["waves_anfc"])
    datasets.append(ds_wav_a)
    print("  Vagues (anfc) OK")

# Atmo : soit fichier global déjà fait, soit on merge les fichiers mensuels
if os.path.exists(EXISTING_FILES["atmo_glob"]):
    ds_atmo = xr.open_dataset(EXISTING_FILES["atmo_glob"])
    datasets.append(ds_atmo)
    print("  Atmo (global) OK")
elif EXISTING_FILES["atmo_months"]:
    ds_atmo = xr.open_mfdataset(EXISTING_FILES["atmo_months"], combine="by_coords")
    if "u10n" in ds_atmo and "v10n" in ds_atmo:
        ds_atmo["wind_speed_neutral"] = np.sqrt(ds_atmo["u10n"] ** 2 + ds_atmo["v10n"] ** 2)
    datasets.append(ds_atmo)
    print(f"  Atmo (mensuel x{len(EXISTING_FILES['atmo_months'])}) OK")

# SST
if os.path.exists(EXISTING_FILES["sst"]):
    ds_sst = xr.open_dataset(EXISTING_FILES["sst"])
    datasets.append(ds_sst)
    print("  SST OK")
else:
    print(f"  ATTENTION : SST non trouvée -> {EXISTING_FILES['sst']}")

# SLA
if os.path.exists(EXISTING_FILES["sla"]):
    ds_sla = xr.open_dataset(EXISTING_FILES["sla"])
    datasets.append(ds_sla)
    print("  SLA OK")
else:
    print(f"  ATTENTION : SLA non trouvée -> {EXISTING_FILES['sla']}")

# ugos / vgos
if os.path.exists(EXISTING_FILES["ugos_vgos"]):
    ds_uv = xr.open_dataset(EXISTING_FILES["ugos_vgos"])
    datasets.append(ds_uv)
    print("  ugos/vgos OK")
else:
    print(f"  ATTENTION : ugos/vgos non trouvés -> {EXISTING_FILES['ugos_vgos']}")

# =============================================================================
# 2. TÉLÉCHARGER LES VARIABLES MANQUANTES : SSS et SSH (adt)
# =============================================================================
print("\n" + "=" * 60)
print("2. Téléchargement des variables manquantes (SSS, SSH)")
print("=" * 60)

SSS_FILE = f"{OUTPUT_DIR}/sss_daily_2023.nc"
if not os.path.exists(SSS_FILE):
    print("  Téléchargement SSS...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["so"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1,
        start_datetime=f"{START_DATE}T00:00:00",
        end_datetime=f"{END_DATE}T00:00:00",
        output_filename=SSS_FILE,
    )
else:
    print(f"  -> Déjà présent : {SSS_FILE}")

SSH_FILE = f"{OUTPUT_DIR}/ssh_daily_2023.nc"
if not os.path.exists(SSH_FILE):
    print("  Téléchargement SSH (adt)...")
    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        variables=["adt"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{START_DATE}T00:00:00",
        end_datetime=f"{END_DATE}T00:00:00",
        output_filename=SSH_FILE,
    )
else:
    print(f"  -> Déjà présent : {SSH_FILE}")

ds_sss = xr.open_dataset(SSS_FILE).squeeze("depth", drop=True) if os.path.exists(SSS_FILE) else None
ds_ssh = xr.open_dataset(SSH_FILE) if os.path.exists(SSH_FILE) else None

if ds_sss is not None:
    datasets.append(ds_sss)
if ds_ssh is not None:
    datasets.append(ds_ssh)

# =============================================================================
# 3. RÉGRIDDER TOUT SUR UNE GRILLE COMMUNE 0.25° + DÉCOUPER GULF STREAM
# =============================================================================
print("\n" + "=" * 60)
print("3. Régriddage 0.25° + découpe Gulf Stream")
print("=" * 60)

target_lon = np.arange(LON_MIN, LON_MAX + TARGET_RES, TARGET_RES)
target_lat = np.arange(LAT_MIN, LAT_MAX + TARGET_RES, TARGET_RES)

def standardize_coords(ds):
    rename_map = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map) if rename_map else ds

regridded = []
for ds in datasets:
    ds = standardize_coords(ds)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        print(f"  ATTENTION : dataset sans lon/lat standard, ignoré -> {list(ds.data_vars)}")
        continue
    ds_crop = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_regrid = ds_crop.interp(lon=target_lon, lat=target_lat, method="linear")
    regridded.append(ds_regrid)
    print(f"  Régriddé : {list(ds.data_vars)}")

ds_merged = xr.merge(regridded, compat="override", join="outer")

# =============================================================================
# 4. CALCULS DÉRIVÉS — DOS (densité) et gradients
# =============================================================================
print("\n" + "=" * 60)
print("4. Calcul de DOS et des gradients (SST, SSS, SLA)")
print("=" * 60)

sst_var = "analysed_sst" if "analysed_sst" in ds_merged else ("sst" if "sst" in ds_merged else None)
sss_var = "so" if "so" in ds_merged else None
sla_var = "sla" if "sla" in ds_merged else ("zos" if "zos" in ds_merged else None)

if sst_var and sss_var:
    SA = ds_merged[sss_var]
    CT = gsw.CT_from_t(SA, ds_merged[sst_var], p=0)
    ds_merged["DOS"] = (SA.dims, gsw.density.rho(SA.values, CT.values, 0))
    print("  DOS calculée")
else:
    print(f"  DOS non calculée (SST ou SSS manquante : sst={sst_var}, sss={sss_var})")

def add_gradient(ds, varname, outname):
    if varname not in ds:
        print(f"  {outname} non calculé ({varname} absent)")
        return
    field = ds[varname]
    dlat = np.gradient(field, axis=field.dims.index("lat"))
    dlon = np.gradient(field, axis=field.dims.index("lon"))
    grad = np.sqrt(dlat ** 2 + dlon ** 2)
    ds[outname] = (field.dims, grad)
    print(f"  {outname} calculé")

add_gradient(ds_merged, sst_var, "grad_SST")
add_gradient(ds_merged, sss_var, "grad_SSS")
add_gradient(ds_merged, sla_var, "grad_SLA")

# =============================================================================
# 5. VÉRIFICATION DE LA RÉSOLUTION 0.25°
# =============================================================================
print("\n" + "=" * 60)
print("5. Vérification de la résolution")
print("=" * 60)

lon_res = np.unique(np.round(np.diff(ds_merged.lon.values), 4))
lat_res = np.unique(np.round(np.diff(ds_merged.lat.values), 4))

assert np.allclose(lon_res, TARGET_RES, atol=1e-3), f"Résolution lon incorrecte : {lon_res}"
assert np.allclose(lat_res, TARGET_RES, atol=1e-3), f"Résolution lat incorrecte : {lat_res}"
print(f"  OK — résolution lon={lon_res}, lat={lat_res}")

# =============================================================================
# 6. SAUVEGARDE FINALE
# =============================================================================
print("\n" + "=" * 60)
print("6. Sauvegarde du fichier fusionné")
print("=" * 60)

ds_merged.to_netcdf(FINAL_OUTPUT)
print(f"  -> Sauvegardé : {FINAL_OUTPUT}")
print(f"  Variables finales : {list(ds_merged.data_vars)}")
print(f"  Dimensions : {dict(ds_merged.sizes)}")
