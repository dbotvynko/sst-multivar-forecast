"""
Fusion des forecasts GLO12 (GLORYS12) 2023 — zone NATL
Source : CMEMS global ocean physics (1/12° ~ 0.083°)

Variables téléchargées via CMEMS :
  - SST (thetao, depth=0)
  - SSH (zos)
  - Courants (uo, vo, depth=0)

Variables réutilisées depuis fichiers existants :
  - SSS : /Odyssey/public/glorys/reanalysis/...so...2023.nc
  - MDT : GLO12 Mercator MDT
  - BATHY : glorys bathymetry

Tout est régriddé à 0.25° sur la zone NATL.

Prérequis : pip install copernicusmarine xarray scipy netCDF4
            copernicusmarine login
"""

import os
import numpy as np
import xarray as xr
import copernicusmarine

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "data/glo12_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_OUTPUT = f"{OUTPUT_DIR}/merged_natl_glo12_2023_025deg.nc"

LON_MIN, LON_MAX = -100.0, 42.0
LAT_MIN, LAT_MAX = -6.0, 90.0
TARGET_RES = 0.25

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Fichiers existants
SSS_PATH = "/Odyssey/public/glorys/reanalysis/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_180.00W-179.92E_80.00S-90.00N_0.49m_2023-01-01-2023-12-31.nc"
MDT_PATH = "/Odyssey/public/glorys/MDT_Mercator/cmems_mod_glo_phy_my_0.083deg_static_mdt_180.00W-179.92E_80.00S-90.00N.nc"
BATHY_PATH = "/Odyssey/public/glorys/bathymetry/bathymetry.nc"

target_lon = np.arange(LON_MIN, LON_MAX + TARGET_RES, TARGET_RES)
target_lat = np.arange(LAT_MIN, LAT_MAX + TARGET_RES, TARGET_RES)

# =============================================================================
# 1. TÉLÉCHARGEMENT VIA CMEMS (SST, SSH, courants)
# =============================================================================
print("=" * 60)
print("1. Téléchargement GLO12 — zone NATL")
print("=" * 60)

CMEMS_DOWNLOADS = {
    "sst": {
        "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "variables": ["thetao"],
        "depth": True,
        "file": f"{OUTPUT_DIR}/glo12_sst_2023_natl.nc",
    },
    "ssh": {
        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        "variables": ["zos"],
        "depth": False,
        "file": f"{OUTPUT_DIR}/glo12_ssh_2023_natl.nc",
    },
    "currents": {
        "dataset_id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "variables": ["uo", "vo"],
        "depth": True,
        "file": f"{OUTPUT_DIR}/glo12_currents_2023_natl.nc",
    },
}

for name, cfg in CMEMS_DOWNLOADS.items():
    if os.path.exists(cfg["file"]):
        print(f"  {name} -> déjà présent")
        continue

    print(f"  Téléchargement {name}...")
    kwargs = {
        "dataset_id": cfg["dataset_id"],
        "variables": cfg["variables"],
        "minimum_longitude": LON_MIN,
        "maximum_longitude": LON_MAX,
        "minimum_latitude": LAT_MIN,
        "maximum_latitude": LAT_MAX,
        "start_datetime": f"{START_DATE}T00:00:00",
        "end_datetime": f"{END_DATE}T00:00:00",
        "output_filename": cfg["file"],
    }
    if cfg["depth"]:
        kwargs["minimum_depth"] = 0
        kwargs["maximum_depth"] = 1

    copernicusmarine.subset(**kwargs)
    print(f"  -> {cfg['file']}")

# =============================================================================
# 2. CHARGEMENT ET RÉGRIDDAGE
# =============================================================================
print("\n" + "=" * 60)
print("2. Chargement et régriddage 0.25°")
print("=" * 60)


def load_and_regrid(path, name):
    ds = xr.open_dataset(path)
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "depth" in ds.dims:
        ds = ds.squeeze("depth", drop=True)
    ds = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds = ds.interp(lon=target_lon, lat=target_lat, method="linear").load()
    print(f"  {name} -> {list(ds.data_vars)}")
    return ds


datasets = []

# SST (thetao -> sst)
if os.path.exists(CMEMS_DOWNLOADS["sst"]["file"]):
    ds = load_and_regrid(CMEMS_DOWNLOADS["sst"]["file"], "SST")
    ds = ds.rename({"thetao": "sst"})
    datasets.append(ds)

# SSH (zos -> ssh)
if os.path.exists(CMEMS_DOWNLOADS["ssh"]["file"]):
    ds = load_and_regrid(CMEMS_DOWNLOADS["ssh"]["file"], "SSH")
    ds = ds.rename({"zos": "ssh"})
    datasets.append(ds)

# Courants (uo, vo)
if os.path.exists(CMEMS_DOWNLOADS["currents"]["file"]):
    ds = load_and_regrid(CMEMS_DOWNLOADS["currents"]["file"], "Courants")
    datasets.append(ds)

# SSS (fichier existant, global -> crop NATL)
print("  SSS (fichier existant)...")
ds_sss = xr.open_dataset(SSS_PATH)
if "latitude" in ds_sss.dims:
    ds_sss = ds_sss.rename({"latitude": "lat", "longitude": "lon"})
if "depth" in ds_sss.dims:
    ds_sss = ds_sss.squeeze("depth", drop=True)
if "so" in ds_sss:
    ds_sss = ds_sss.rename({"so": "sss"})
ds_sss = ds_sss.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_sss = ds_sss.interp(lon=target_lon, lat=target_lat, method="linear").load()
datasets.append(ds_sss)
print(f"  SSS -> {list(ds_sss.data_vars)}")

# MDT (GLO12 Mercator, statique)
print("  MDT (GLO12 Mercator)...")
ds_mdt = xr.open_dataset(MDT_PATH)
if "latitude" in ds_mdt.dims:
    ds_mdt = ds_mdt.rename({"latitude": "lat", "longitude": "lon"})
if "mdt" in ds_mdt:
    ds_mdt = ds_mdt[["mdt"]]
else:
    mdt_var = [v for v in ds_mdt.data_vars if "mdt" in v.lower()]
    if mdt_var:
        ds_mdt = ds_mdt[[mdt_var[0]]].rename({mdt_var[0]: "mdt"})
ds_mdt = ds_mdt.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_mdt = ds_mdt.interp(lon=target_lon, lat=target_lat, method="linear").load()
datasets.append(ds_mdt)
print(f"  MDT -> {list(ds_mdt.data_vars)}")

# BATHY (statique, réutilisé)
print("  BATHY (statique)...")
ds_bathy = xr.open_dataset(BATHY_PATH)
if "latitude" in ds_bathy.dims:
    ds_bathy = ds_bathy.rename({"latitude": "lat", "longitude": "lon"})
if "deptho" in ds_bathy:
    ds_bathy = ds_bathy.rename({"deptho": "elevation"})
ds_bathy = ds_bathy.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_bathy = ds_bathy.interp(lon=target_lon, lat=target_lat, method="linear").load()
datasets.append(ds_bathy)
print(f"  BATHY -> {list(ds_bathy.data_vars)}")

# =============================================================================
# 3. MERGE + SLA
# =============================================================================
print("\n" + "=" * 60)
print("3. Fusion et calculs dérivés")
print("=" * 60)

ds_merged = xr.merge(datasets, compat="override", join="outer")

# SLA = SSH - MDT
if "ssh" in ds_merged and "mdt" in ds_merged:
    ds_merged["sla"] = ds_merged["ssh"] - ds_merged["mdt"]
    print("  SLA = SSH - MDT")

print(f"  Variables : {list(ds_merged.data_vars)}")
print(f"  Dimensions : {dict(ds_merged.sizes)}")

# =============================================================================
# 4. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 60)
print("4. Sauvegarde")
print("=" * 60)

encoding = {
    v: {"dtype": "float32", "zlib": True, "complevel": 4}
    for v in ds_merged.data_vars
}
ds_merged.to_netcdf(FINAL_OUTPUT, encoding=encoding)
print(f"  -> {FINAL_OUTPUT}")
print("Terminé !")
