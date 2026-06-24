"""
Fusion de toutes les variables 2023 (zone NATL) dans un seul fichier NetCDF.

Zone : lat -6° à 90°, lon -100° à 42°, résolution 0.25°

Sources :
  - Vagues : CMEMS (déjà téléchargé, global)
  - Atmo ERA5 : CDS (déjà téléchargé, global)
  - SSS : CMEMS (à télécharger pour NATL)
  - SLA : reconstruction UNet
  - SST : reconstruction UNet (ABSOLUTE_SST)
  - UGOS/VGOS : reconstruction UNet
  - MDT : CNES_CLS22
  - BATHY : glorys bathymetry

Prérequis : pip install xarray copernicusmarine gsw netCDF4 scipy
"""

import os
import numpy as np
import xarray as xr
import copernicusmarine
import gsw

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_OUTPUT = f"{OUTPUT_DIR}/merged_natl_2023_025deg.nc"

# Zone NATL
LON_MIN, LON_MAX = -100.0, 42.0
LAT_MIN, LAT_MAX = -6.0, 90.0
TARGET_RES = 0.25

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# --- Fichiers déjà téléchargés (vagues, atmo) --------------------------------
WAVE_REAN = f"{OUTPUT_DIR}/waves_daily_2023_rean.nc"
WAVE_ANFC = f"{OUTPUT_DIR}/waves_daily_2023_anfc.nc"
ATMO_MONTHS = sorted(
    f"{OUTPUT_DIR}/{f}" for f in os.listdir(OUTPUT_DIR)
    if f.startswith("atmo_daily_2023_") and f.endswith(".nc")
)

# --- Fichiers sources (SLA, SST, UGOS/VGOS, MDT, BATHY) ----------------------
SLA_PATH = "/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022/nrt_sla/test_data_14.nc"
SST_PATH = "/Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_SLA_grid_SLA_INPUT_src_CORRECTED/nrt_sla/ABSOLUTE_SST/test_data_14.nc"
UGOS_VGOS_PATH = "/Odyssey/public/glorys/rec/evaluation/all_365_days_UNet_sla_real_data_training_2017_2022_GEOS_velocities_leadtime_00.nc"
MDT_PATH = "/Odyssey/public/mean_dynamic_topography/CNES_CLS22.nc"
BATHY_PATH = "/Odyssey/public/glorys/bathymetry/bathymetry.nc"

# =============================================================================
# HELPERS
# =============================================================================

target_lon = np.arange(LON_MIN, LON_MAX + TARGET_RES, TARGET_RES)
target_lat = np.arange(LAT_MIN, LAT_MAX + TARGET_RES, TARGET_RES)


def standardize_coords(ds):
    rename_map = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if rename_map:
        ds = ds.rename(rename_map)
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
    if "lat" in ds.coords and float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sortby("lat")
    return ds


def crop_and_regrid(ds, name=""):
    ds = standardize_coords(ds)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        print(f"  SKIP (pas de lon/lat) : {name} -> {list(ds.data_vars)}")
        return None
    ds_crop = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    if ds_crop.sizes.get("lon", 0) == 0 or ds_crop.sizes.get("lat", 0) == 0:
        print(f"  SKIP (crop vide) : {name}, lon=[{float(ds.lon.min()):.1f}, {float(ds.lon.max()):.1f}]")
        return None
    ds_crop = ds_crop.load()
    ds_regrid = ds_crop.interp(lon=target_lon, lat=target_lat, method="linear")
    if "time" in ds_regrid.dims and ds_regrid.sizes["time"] > 400:
        ds_regrid = ds_regrid.resample(time="1D").mean()
        print(f"  OK (regrid + daily) : {name} -> {list(ds_regrid.data_vars)}")
    else:
        print(f"  OK (regrid) : {name} -> {list(ds_regrid.data_vars)}")
    return ds_regrid


# =============================================================================
# 1. CHARGEMENT + REGRID DE TOUTES LES SOURCES
# =============================================================================
print("=" * 60)
print("1. Chargement et régriddage sur la zone NATL")
print("=" * 60)

regridded = []

# --- Vagues ---
if os.path.exists(WAVE_REAN):
    r = crop_and_regrid(xr.open_dataset(WAVE_REAN), "waves_rean")
    if r is not None:
        regridded.append(r)

if os.path.exists(WAVE_ANFC):
    r = crop_and_regrid(xr.open_dataset(WAVE_ANFC), "waves_anfc")
    if r is not None:
        regridded.append(r)

# --- Atmo ERA5 ---
if ATMO_MONTHS:
    ds_atmo = xr.open_mfdataset(ATMO_MONTHS, combine="by_coords")
    if "u10n" in ds_atmo and "v10n" in ds_atmo:
        ds_atmo["wind_speed_neutral"] = np.sqrt(ds_atmo["u10n"] ** 2 + ds_atmo["v10n"] ** 2)
    r = crop_and_regrid(ds_atmo, "atmo_era5")
    if r is not None:
        regridded.append(r)

# --- SLA (variable "out" -> "sla") ---
if os.path.exists(SLA_PATH):
    ds_sla = xr.open_dataset(SLA_PATH).rename({"out": "sla"})
    r = crop_and_regrid(ds_sla, "sla")
    if r is not None:
        regridded.append(r)

# --- SST (variable "out" -> "sst", K -> °C) ---
if os.path.exists(SST_PATH):
    ds_sst = xr.open_dataset(SST_PATH).rename({"out": "sst"})
    if float(ds_sst["sst"].mean()) > 200:
        ds_sst["sst"] = ds_sst["sst"] - 273.15
    r = crop_and_regrid(ds_sst, "sst")
    if r is not None:
        regridded.append(r)

# --- UGOS / VGOS ---
if os.path.exists(UGOS_VGOS_PATH):
    r = crop_and_regrid(xr.open_dataset(UGOS_VGOS_PATH), "ugos_vgos")
    if r is not None:
        regridded.append(r)

# --- MDT (statique, pas de dim time) ---
if os.path.exists(MDT_PATH):
    ds_mdt = xr.open_dataset(MDT_PATH)
    ds_mdt = standardize_coords(ds_mdt)
    mdt_var = [v for v in ds_mdt.data_vars if "mdt" in v.lower()]
    if mdt_var:
        ds_mdt = ds_mdt[mdt_var]
        if len(mdt_var) == 1 and mdt_var[0] != "mdt":
            ds_mdt = ds_mdt.rename({mdt_var[0]: "mdt"})
    ds_mdt = ds_mdt.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX)).load()
    ds_mdt = ds_mdt.interp(lon=target_lon, lat=target_lat, method="linear")
    regridded.append(ds_mdt)
    print(f"  OK (regrid) : mdt -> {list(ds_mdt.data_vars)}")

# --- Bathymetry (statique) ---
if os.path.exists(BATHY_PATH):
    ds_bathy = xr.open_dataset(BATHY_PATH)
    ds_bathy = standardize_coords(ds_bathy)
    if "deptho" in ds_bathy:
        ds_bathy = ds_bathy.rename({"deptho": "elevation"})
    ds_bathy = ds_bathy.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX)).load()
    ds_bathy = ds_bathy.interp(lon=target_lon, lat=target_lat, method="linear")
    regridded.append(ds_bathy)
    print(f"  OK (regrid) : bathy -> {list(ds_bathy.data_vars)}")

# =============================================================================
# 2. SSS — télécharger pour la zone NATL si nécessaire
# =============================================================================
print("\n" + "=" * 60)
print("2. Téléchargement SSS (zone NATL)")
print("=" * 60)

SSS_FILE = f"{OUTPUT_DIR}/sss_daily_2023_natl.nc"
if not os.path.exists(SSS_FILE):
    print("  Téléchargement SSS...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
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

if os.path.exists(SSS_FILE):
    ds_sss = xr.open_dataset(SSS_FILE).squeeze("depth", drop=True).rename({"so": "sss"})
    r = crop_and_regrid(ds_sss, "sss")
    if r is not None:
        regridded.append(r)

# =============================================================================
# 3. MERGE
# =============================================================================
print("\n" + "=" * 60)
print("3. Fusion de toutes les variables")
print("=" * 60)

ds_merged = xr.merge(regridded, compat="override", join="outer")

# SSH = SLA + MDT
if "sla" in ds_merged and "mdt" in ds_merged:
    ds_merged["ssh"] = ds_merged["sla"] + ds_merged["mdt"]
    print("  SSH = SLA + MDT ✓")

# =============================================================================
# 4. CONVERSIONS D'UNITÉS
# =============================================================================
print("\n" + "=" * 60)
print("4. Conversions d'unités")
print("=" * 60)

# sshf et slhf : J/m² -> W/m²
for flux_var in ["sshf", "slhf"]:
    if flux_var in ds_merged:
        ds_merged[flux_var] = ds_merged[flux_var] / 3600.0
        print(f"  {flux_var} : J/m² -> W/m²")

# =============================================================================
# 5. CALCULS DÉRIVÉS — DOS et gradients
# =============================================================================
print("\n" + "=" * 60)
print("5. Calculs dérivés (DOS, gradients)")
print("=" * 60)

sst_var = "sst" if "sst" in ds_merged else None
sss_var = "sss" if "sss" in ds_merged else None
sla_var = "sla" if "sla" in ds_merged else None

if sst_var and sss_var:
    SA = gsw.SA_from_SP(ds_merged[sss_var], 0, ds_merged.lon, ds_merged.lat)
    CT = gsw.CT_from_t(SA, ds_merged[sst_var], 0)
    ds_merged["DOS"] = (ds_merged[sst_var].dims, gsw.density.rho(SA.values, CT.values, 0))
    print("  DOS ✓")


def add_gradient(ds, varname, outname):
    if varname not in ds or varname is None:
        print(f"  {outname} SKIP ({varname} absent)")
        return
    field = ds[varname]
    dlat = np.gradient(field, axis=field.dims.index("lat"))
    dlon = np.gradient(field, axis=field.dims.index("lon"))
    ds[outname] = (field.dims, np.sqrt(dlat ** 2 + dlon ** 2))
    print(f"  {outname} ✓")


add_gradient(ds_merged, sst_var, "grad_SST")
add_gradient(ds_merged, sss_var, "grad_SSS")
add_gradient(ds_merged, sla_var, "grad_SLA")

# =============================================================================
# 6. VÉRIFICATION + SAUVEGARDE
# =============================================================================
print("\n" + "=" * 60)
print("6. Vérification et sauvegarde")
print("=" * 60)

lon_res = np.unique(np.round(np.diff(ds_merged.lon.values), 4))
lat_res = np.unique(np.round(np.diff(ds_merged.lat.values), 4))
assert np.allclose(lon_res, TARGET_RES, atol=1e-3), f"Résolution lon incorrecte : {lon_res}"
assert np.allclose(lat_res, TARGET_RES, atol=1e-3), f"Résolution lat incorrecte : {lat_res}"
print(f"  Résolution : {lon_res[0]}° x {lat_res[0]}°")
print(f"  Dimensions : {dict(ds_merged.sizes)}")
print(f"  Variables : {list(ds_merged.data_vars)}")

encoding = {
    var: {"dtype": "float32", "zlib": True, "complevel": 4}
    for var in ds_merged.data_vars
}
ds_merged.to_netcdf(FINAL_OUTPUT, encoding=encoding)
print(f"\n  -> Sauvegardé : {FINAL_OUTPUT}")
print("Terminé !")
