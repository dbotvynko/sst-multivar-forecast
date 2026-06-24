"""
Download ERA5 10m wind components (u10, v10) as daily means at 0.25° resolution.

Uses the 'derived-era5-single-levels-daily-statistics' dataset which provides
pre-computed daily means directly — no need to download hourly data.

Requirements:
    pip install cdsapi

Setup:
    1. Register at https://cds.climate.copernicus.eu/
    2. Create ~/.cdsapirc with your API key:
       url: https://cds.climate.copernicus.eu/api
       key: <your-uid>:<your-api-key>

Usage:
    python contrib/data_loading/download_era5_wind.py \
        --output /Odyssey/public/era5/2017_2019/era5_10m_wind_daily_mean_0.25deg.nc \
        --start_year 2017 --end_year 2019
"""
import argparse
import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path


def download_era5_wind_month(client, year, month, output_dir):
    """Download daily-mean ERA5 wind for one month."""
    monthly_path = output_dir / f'era5_10m_wind_daily_{year}_{month:02d}.nc'

    if monthly_path.exists():
        print(f'  {year}-{month:02d}: already exists, skipping')
        return monthly_path

    print(f'  {year}-{month:02d}: downloading daily-mean ERA5 wind...')
    client.retrieve(
        'derived-era5-single-levels-daily-statistics',
        {
            'product_type': 'reanalysis',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'daily_statistic': 'daily_mean',
            'time_zone': 'utc+00:00',
            'frequency': '1_hourly',
            'data_format': 'netcdf',
            'grid': [0.25, 0.25],
        },
        str(monthly_path),
    )
    print(f'  {year}-{month:02d}: downloaded')
    return monthly_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Final merged output path')
    parser.add_argument('--start_year', type=int, default=2017)
    parser.add_argument('--end_year', type=int, default=2019)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    monthly_files = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            path = download_era5_wind_month(client, year, month, output_dir)
            monthly_files.append(path)

    print('Merging all monthly files...')
    ds = xr.open_mfdataset(monthly_files, combine='by_coords')
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.astype(np.float32)
    ds.to_netcdf(output_path)
    print(f'Saved merged file to {output_path}')

    for f in monthly_files:
        f.unlink()
        print(f'  Removed {f}')


if __name__ == '__main__':
    main()
