import xarray as xr
import numpy as np

LON_MIN, LON_MAX = -100.0, 42.0
LAT_MIN, LAT_MAX = -6.0, 90.0
TARGET_RES = 0.25
OUTPUT = "data/ecmwf_forecasts_2023/merged_natl_2023_025deg.nc"

target_lon = np.arange(LON_MIN, LON_MAX + TARGET_RES, TARGET_RES)
target_lat = np.arange(LAT_MIN, LAT_MAX + TARGET_RES, TARGET_RES)

# --- SLA ---
print("SLA...")
ds_sla = xr.open_dataset("/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022/nrt_sla/test_data_14.nc")
ds_sla = ds_sla.rename({"out": "sla"})
ds_sla = ds_sla.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_sla = ds_sla.interp(lon=target_lon, lat=target_lat, method="linear")

# --- SST ---
print("SST...")
ds_sst = xr.open_dataset("/Odyssey/public/glorys/rec/glorys4_global_1patch_SST_ODYSSEA_UNet_L3_NRT_AnomalyCLIMATO_2010_2019_NormStatsImposed_MSEGradLoss_pyresample_0.25degL3_INPUT_INFRARED_FLAGGED_5_SLA_grid_SLA_INPUT_src_CORRECTED/nrt_sla/ABSOLUTE_SST/test_data_14.nc")
ds_sst = ds_sst.rename({"out": "sst"})
if float(ds_sst["sst"].mean()) > 200:
    ds_sst["sst"] = ds_sst["sst"] - 273.15
ds_sst = ds_sst.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_sst = ds_sst.interp(lon=target_lon, lat=target_lat, method="linear")

# --- UGOS / VGOS ---
print("UGOS/VGOS...")
ds_uv = xr.open_dataset("/Odyssey/private/d21botvy/FORECAST/2023a_SSH_mapping_OSE/ssh_2023_Pierre_GLO12/2023a_SSH_mapping_OSE/nb_diags_global/UNet_sla_real_data_training_Nadir_filt_DUACS_losses_2017-2022_all_days_GEOS_velocities_leadtime_0.nc")
if "latitude" in ds_uv.dims:
    ds_uv = ds_uv.rename({"latitude": "lat", "longitude": "lon"})
ds_uv = ds_uv.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_uv = ds_uv.interp(lon=target_lon, lat=target_lat, method="linear")

# --- MDT (statique) ---
print("MDT...")
ds_mdt = xr.open_dataset("/Odyssey/public/mean_dynamic_topography/CNES_CLS22.nc")
if "latitude" in ds_mdt.dims:
    ds_mdt = ds_mdt.rename({"latitude": "lat", "longitude": "lon"})
if "mdt" in ds_mdt:
    ds_mdt = ds_mdt[["mdt"]]
else:
    mdt_var = [v for v in ds_mdt.data_vars if "mdt" in v.lower()]
    if mdt_var:
        ds_mdt = ds_mdt[[mdt_var[0]]].rename({mdt_var[0]: "mdt"})
ds_mdt = ds_mdt.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_mdt = ds_mdt.interp(lon=target_lon, lat=target_lat, method="linear")

# --- BATHY (statique) ---
print("BATHY...")
ds_bathy = xr.open_dataset("/Odyssey/public/glorys/bathymetry/bathymetry.nc")
if "latitude" in ds_bathy.dims:
    ds_bathy = ds_bathy.rename({"latitude": "lat", "longitude": "lon"})
if "deptho" in ds_bathy:
    ds_bathy = ds_bathy.rename({"deptho": "elevation"})
ds_bathy = ds_bathy.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_bathy = ds_bathy.interp(lon=target_lon, lat=target_lat, method="linear")


# --- SSS ---
print("SSS...")
ds_sss = xr.open_dataset("data/ecmwf_forecasts_2023/sss_daily_2023_natl.nc")
if "depth" in ds_sss.dims:
    ds_sss = ds_sss.squeeze("depth", drop=True)
if "so" in ds_sss:
    ds_sss = ds_sss.rename({"so": "sss"})
if "latitude" in ds_sss.dims:
    ds_sss = ds_sss.rename({"latitude": "lat", "longitude": "lon"})
ds_sss = ds_sss.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
ds_sss = ds_sss.interp(lon=target_lon, lat=target_lat, method="linear")


# --- MERGE ---
print("Merge...")
#ds = xr.merge([ds_sla, ds_sst, ds_uv, ds_mdt, ds_bathy], compat="override", join="outer")
ds = xr.merge([ds_sla, ds_sst, ds_uv, ds_mdt, ds_bathy, ds_sss], compat="override", join="outer")

# SSH = SLA + MDT
if "sla" in ds and "mdt" in ds:
    ds["ssh"] = ds["sla"] + ds["mdt"]
    print("  SSH = SLA + MDT")

print(f"Variables : {list(ds.data_vars)}")
print(f"Dimensions : {dict(ds.sizes)}")

# --- SAUVEGARDE ---
encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(OUTPUT, encoding=encoding)
print(f"Sauvegardé : {OUTPUT}")
