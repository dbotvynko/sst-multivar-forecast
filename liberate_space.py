import xarray as xr
import os

LON_MIN, LON_MAX = -100.0, 42.0
LAT_MIN, LAT_MAX = -6.0, 90.0
DIR = "data/ecmwf_forecasts_2023"

# Waves anfc (96 Go -> ~2 Go)
print("Crop waves_anfc...")
ds = xr.open_dataset(f"{DIR}/waves_daily_2023_anfc.nc", chunks={"time": 100})
ds.sel(longitude=slice(LON_MIN, LON_MAX), latitude=slice(LAT_MIN, LAT_MAX)).to_netcdf(
    f"{DIR}/waves_daily_2023_anfc_natl.nc"
)
ds.close()
print("  -> OK, supprime l'original")
os.remove(f"{DIR}/waves_daily_2023_anfc.nc")

# Waves rean (18 Go -> ~1 Go)
print("Crop waves_rean...")
ds = xr.open_dataset(f"{DIR}/waves_daily_2023_rean.nc", chunks={"time": 100})
ds.sel(longitude=slice(LON_MIN, LON_MAX), latitude=slice(LAT_MIN, LAT_MAX)).to_netcdf(
    f"{DIR}/waves_daily_2023_rean_natl.nc"
)
ds.close()
print("  -> OK, supprime l'original")
os.remove(f"{DIR}/waves_daily_2023_rean.nc")

# Atmo months (75 Go -> ~5 Go)
for f in sorted(os.listdir(DIR)):
    if f.startswith("atmo_daily_2023_") and f.endswith(".nc") and "natl" not in f:
        print(f"Crop {f}...")
        ds = xr.open_dataset(f"{DIR}/{f}", chunks={"valid_time": 100})
        dims = list(ds.dims)
        if "latitude" in dims:
            cropped = ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
        elif "lat" in dims:
            cropped = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        else:
            print(f"  SKIP dims={dims}")
            continue
        cropped.to_netcdf(f"{DIR}/{f.replace('.nc', '_natl.nc')}")
        ds.close()
        os.remove(f"{DIR}/{f}")
        print(f"  -> OK")

print("Terminé !")
