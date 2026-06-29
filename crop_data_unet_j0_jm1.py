"""
Crop and merge UNet reconstruction data for j-1 (last obs day) and j0 (first forecast day).

Sources:
  - SLA: test_data_13.nc (j-1), test_data_14.nc (j0)
  - SST: ABSOLUTE_SST/test_data_13.nc (j-1), ABSOLUTE_SST/test_data_14.nc (j0)
  - UGOS/VGOS: computed from SLA via geostrophic velocity
  - SSS: CMEMS GLO12 existing file
  - MDT: CNES_CLS22
  - BATHY: glorys bathymetry

Zone NATL: lat -6/90, lon -100/42, 0.25deg
"""

import os
import numpy as np
import xarray as xr

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "data/ecmwf_forecasts_2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = "/Odyssey/public/glorys/rec/glorys4_global_1patch_SST_SLA_INOUT_UNet_ResBlock_2010_2019_MSEGradLoss_OSE_real_data_pervarnorm_bettereCKPT/nrt_sst_sla_2023_with_inputs"

LEADTIMES = {
    "jm1": {"file_idx": "test_data_13.nc", "label": "j-1"},
    "j0":  {"file_idx": "test_data_14.nc", "label": "j0"},
}

SSS_PATH = "/Odyssey/public/glorys/reanalysis/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_180.00W-179.92E_80.00S-90.00N_0.49m_2023-01-01-2023-12-31.nc"
MDT_PATH = "/Odyssey/public/mean_dynamic_topography/CNES_CLS22.nc"
BATHY_PATH = "/Odyssey/public/glorys/bathymetry/bathymetry.nc"
MDT_UV_PATH = "/Odyssey/public/duacs/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_multi-vars_179.94W-179.94E_89.94S-89.94N_2003-01-01.nc"

LON_MIN, LON_MAX = -100.0, 42.0
LAT_MIN, LAT_MAX = -6.0, 90.0
TARGET_RES = 0.25

target_lon = np.arange(LON_MIN, LON_MAX + TARGET_RES, TARGET_RES)
target_lat = np.arange(LAT_MIN, LAT_MAX + TARGET_RES, TARGET_RES)


# =============================================================================
# GEOSTROPHIC VELOCITY COMPUTATION
# =============================================================================

g_const = 9.81
omega = 7.2921e-5


def compute_geostrophic_velocity(lat, lon, sla, mdt_u, mdt_v):
    f = 2 * omega * np.sin(np.radians(lat))
    dlat = np.gradient(lat) * 111e3
    dlon = np.gradient(lon) * 111e3
    dy = dlat
    dx = np.cos(np.radians(lat[:, None])) * dlon

    dSLA_dy = np.gradient(sla, axis=1) / dy[:, None]
    dSLA_dx = np.gradient(sla, axis=2) / dx

    f_masked = np.where(np.abs(lat) < 2, np.nan, f)

    vg_anomaly = g_const / f_masked[:, None] * dSLA_dx
    ug_anomaly = -g_const / f_masked[:, None] * dSLA_dy

    ug = ug_anomaly + mdt_u
    vg = vg_anomaly + mdt_v

    return ug, vg, ug_anomaly, vg_anomaly


def retrieve_geos_velocities(ds_sla):
    """Compute UGOS/VGOS from SLA using DUACS MDT for u/v.

    Interpolates MDT u/v onto the SLA grid (not the other way around)
    to preserve SLA coverage.
    """
    mdt_UV = xr.open_dataset(MDT_UV_PATH).isel(time=0)
    mdt_UV["longitude"] = (mdt_UV["longitude"] % 360).where(mdt_UV["longitude"] != 360, 0)
    _, index = np.unique(mdt_UV.coords["longitude"], return_index=True)
    mdt_UV = mdt_UV.isel(longitude=index)
    # Convert MDT lon from 0-360 to -180/180 to match SLA
    mdt_UV = mdt_UV.assign_coords(longitude=(mdt_UV.longitude + 180) % 360 - 180).sortby("longitude")

    # Interpolate MDT u/v onto SLA grid (preserves SLA coverage)
    mdt_on_sla = mdt_UV.interp(latitude=ds_sla.lat, longitude=ds_sla.lon)

    lat = ds_sla["lat"].values
    lon = ds_sla["lon"].values

    n_times = ds_sla.time.values.shape[0]
    MDT_u = np.repeat(mdt_on_sla["u"].values[np.newaxis, :, :], n_times, axis=0)
    MDT_v = np.repeat(mdt_on_sla["v"].values[np.newaxis, :, :], n_times, axis=0)

    ugos, vgos, ugosa, vgosa = compute_geostrophic_velocity(lat, lon, ds_sla["sla"].values, MDT_u, MDT_v)

    ds_out = ds_sla.copy()
    ds_out["ugos"] = (("time", "lat", "lon"), ugos)
    ds_out["vgos"] = (("time", "lat", "lon"), vgos)

    return ds_out[["ugos", "vgos"]]


# =============================================================================
# PROCESS EACH LEADTIME
# =============================================================================

for key, cfg in LEADTIMES.items():
    print("=" * 60)
    print(f"Processing {cfg['label']} ({cfg['file_idx']})")
    print("=" * 60)

    output_file = f"{OUTPUT_DIR}/merged_natl_2023_025deg_{key}.nc"

    # --- SLA + SST (same file may contain both) ---
    print("  SLA...")
    sla_path = os.path.join(BASE_DIR, cfg["file_idx"])
    ds_main = xr.open_dataset(sla_path)
    if "out" in ds_main:
        ds_main = ds_main.rename({"out": "sla"})
    ds_sla = ds_main[["sla"]] if "sla" in ds_main else ds_main

    print("  SST...")
    if "sst" in ds_main:
        ds_sst = ds_main[["sst"]]
    else:
        sst_path = os.path.join(BASE_DIR, "ABSOLUTE_SST", cfg["file_idx"])
        ds_sst = xr.open_dataset(sst_path)
        if "out" in ds_sst:
            ds_sst = ds_sst.rename({"out": "sst"})
    if float(ds_sst["sst"].mean()) > 200:
        ds_sst["sst"] = ds_sst["sst"] - 273.15

    # --- UGOS / VGOS (computed from SLA) ---
    print("  UGOS/VGOS (geostrophic from SLA)...")
    ds_vel = retrieve_geos_velocities(ds_sla)

    # --- Crop and regrid SLA/SST ---
    print("  Regridding SLA/SST...")
    ds_sla = ds_sla.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_sla = ds_sla.interp(lon=target_lon, lat=target_lat, method="linear")

    ds_sst = ds_sst.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_sst = ds_sst.interp(lon=target_lon, lat=target_lat, method="linear")

    # --- Crop and regrid UGOS/VGOS ---
    print("  Regridding UGOS/VGOS...")
    ds_vel = ds_vel.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_vel = ds_vel.interp(lon=target_lon, lat=target_lat, method="linear")

    # --- SSS ---
    print("  SSS...")
    ds_sss = xr.open_dataset(SSS_PATH)
    if "latitude" in ds_sss.dims:
        ds_sss = ds_sss.rename({"latitude": "lat", "longitude": "lon"})
    if "depth" in ds_sss.dims:
        ds_sss = ds_sss.squeeze("depth", drop=True)
    if "so" in ds_sss:
        ds_sss = ds_sss.rename({"so": "sss"})
    ds_sss = ds_sss.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_sss = ds_sss.interp(lon=target_lon, lat=target_lat, method="linear")

    # --- MDT ---
    print("  MDT...")
    ds_mdt = xr.open_dataset(MDT_PATH)
    if "latitude" in ds_mdt.dims:
        ds_mdt = ds_mdt.rename({"latitude": "lat", "longitude": "lon"})
    if "mdt" in ds_mdt:
        ds_mdt = ds_mdt[["mdt"]]
    else:
        mdt_var = [v for v in ds_mdt.data_vars if "mdt" in v.lower()]
        if mdt_var:
            ds_mdt = ds_mdt[[mdt_var[0]]].rename({mdt_var[0]: "mdt"})
    if "time" in ds_mdt.dims:
        ds_mdt = ds_mdt.squeeze("time", drop=True)
    ds_mdt = ds_mdt.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_mdt = ds_mdt.interp(lon=target_lon, lat=target_lat, method="linear")

    # --- BATHY ---
    print("  BATHY...")
    ds_bathy = xr.open_dataset(BATHY_PATH)
    if "latitude" in ds_bathy.dims:
        ds_bathy = ds_bathy.rename({"latitude": "lat", "longitude": "lon"})
    if "deptho" in ds_bathy:
        ds_bathy = ds_bathy.rename({"deptho": "elevation"})
    ds_bathy = ds_bathy.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    ds_bathy = ds_bathy.interp(lon=target_lon, lat=target_lat, method="linear")

    # --- MERGE ---
    print("  Merging...")
    ds = xr.merge([ds_sla, ds_sst, ds_vel, ds_mdt, ds_bathy, ds_sss], compat="override", join="outer")

    # SSH = SLA + MDT
    if "sla" in ds and "mdt" in ds:
        ds["ssh"] = ds["sla"] + ds["mdt"]
        print("  SSH = SLA + MDT")

    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")

    # --- SAVE ---
    encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"  -> {output_file}")

print("\nTerminé !")
