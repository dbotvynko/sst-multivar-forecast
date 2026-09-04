"""
Download ERA5 surface wind stress (eastward/northward turbulent surface
stress), daily mean, 0.25deg, for a range of years.

Wind stress (tau_x, tau_y) is what actually forces ocean surface
dynamics (Ekman transport, upwelling, SSH/SLA response) - a more
physically direct driver than raw 10m wind velocity (u10/v10), which
this replaces as the model's wind input.

CDS variable names requested: mean_eastward_turbulent_surface_stress /
mean_northward_turbulent_surface_stress (N/m^2, already a mean rate, so
daily-averaging it is meaningful). Expected short names in the returned
netCDF are "metss"/"mntss" - double check with
`xr.open_dataset(path).data_vars` after a first download, since CDS
short names occasionally differ from what's documented.

Requires a CDS (Climate Data Store) account and API key - see
download_era5_wind.py's docstring for setup (same ~/.cdsapirc, same
cdsapi package).

Downloads one file per month (resumable - already-downloaded months are
skipped; a whole year in one request hits CDS's per-request cost limit -
"HTTPError: 403 ... cost limits exceeded"), then merges everything into a
single netCDF.

Usage:
    python contrib/data_loading/download_era5_wind_stress.py \
        --output-dir /Odyssey/public/era5_stress/2017_2019 \
        --start-year 2017 --end-year 2019
"""
import argparse
from pathlib import Path

import cdsapi
import xarray as xr


def download_month(client, year, month, out_path):
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return
    print(f"Requesting ERA5 wind stress, daily mean, {year}-{month:02d}...")
    client.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": [
                "mean_eastward_turbulent_surface_stress",
                "mean_northward_turbulent_surface_stress",
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
        },
        str(out_path),
    )
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Only download the per-month files, skip merging into one file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    per_month_paths = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            out_path = output_dir / f"era5_wind_stress_daily_mean_0.25deg_{year}_{month:02d}.nc"
            download_month(client, year, month, out_path)
            per_month_paths.append(out_path)

    if args.skip_merge:
        return

    print("Merging per-month files...")
    ds = xr.open_mfdataset([str(p) for p in per_month_paths], combine="by_coords")
    print("Data variables in the merged file (check these match wind_u_variable/wind_v_variable in the xp config):")
    print(list(ds.data_vars))
    merged_path = output_dir / f"era5_wind_stress_daily_mean_0.25deg_{args.start_year}_{args.end_year}.nc"
    ds.to_netcdf(merged_path)
    print(f"Merged into {merged_path}")


if __name__ == "__main__":
    main()
