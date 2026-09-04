"""
Download ERA5 10m wind (u10, v10), daily mean, 0.25deg, for a range of
years - same product/convention already used for training
(paths.wind_data: era5_10m_wind_daily_mean_0.25deg.nc, 2017-2019), here
extended to 2010-2019 to match GLORYS12 SST/SLA's full coverage.

Requires a CDS (Climate Data Store) account and API key:
    1. Register at https://cds.climate.copernicus.eu
    2. Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-personal-access-token>
    3. pip install cdsapi

Downloads one file per year (resumable - already-downloaded years are
skipped), then merges them into a single netCDF matching the existing
naming convention.

Usage:
    python contrib/data_loading/download_era5_wind.py \
        --output-dir /Odyssey/public/era5/2010_2019 \
        --start-year 2010 --end-year 2019
"""
import argparse
from pathlib import Path

import cdsapi
import xarray as xr


def download_year(client, year, out_path):
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return
    print(f"Requesting ERA5 10m wind, daily mean, {year}...")
    client.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
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
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Only download the per-year files, skip merging into one file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    per_year_paths = []
    for year in range(args.start_year, args.end_year + 1):
        out_path = output_dir / f"era5_10m_wind_daily_mean_0.25deg_{year}.nc"
        download_year(client, year, out_path)
        per_year_paths.append(out_path)

    if args.skip_merge:
        return

    print("Merging per-year files...")
    ds = xr.open_mfdataset([str(p) for p in per_year_paths], combine="by_coords")
    merged_path = output_dir / f"era5_10m_wind_daily_mean_0.25deg_{args.start_year}_{args.end_year}.nc"
    ds.to_netcdf(merged_path)
    print(f"Merged into {merged_path}")


if __name__ == "__main__":
    main()
