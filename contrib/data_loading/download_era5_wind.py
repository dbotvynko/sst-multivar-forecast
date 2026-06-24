"""
Download ERA5 10m wind components (u10, v10) as daily means at 0.25° resolution.

Requirements:
    pip install cdsapi

Setup:
    1. Register at https://cds.climate.copernicus.eu/
    2. Create ~/.cdsapirc with your API key:
       url: https://cds.climate.copernicus.eu/api
       key: <your-uid>:<your-api-key>

Usage:
    python contrib/data_loading/download_era5_wind.py \
        --output /Odyssey/public/era5/2010_2019/era5_10m_wind_daily_mean_0.25deg.nc \
        --start_year 2010 --end_year 2019
"""
import argparse
import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path


def download_era5_wind_year(client, year, output_dir):
    """Download hourly ERA5 wind for one year, then compute daily mean."""
    hourly_path = output_dir / f'era5_10m_wind_hourly_{year}.nc'
    daily_path = output_dir / f'era5_10m_wind_daily_{year}.nc'

    if daily_path.exists():
        print(f'  {year}: daily file already exists, skipping')
        return daily_path

    if not hourly_path.exists():
        print(f'  {year}: downloading hourly ERA5 wind...')
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': str(year),
                'month': [f'{m:02d}' for m in range(1, 13)],
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'data_format': 'netcdf',
                'grid': [0.25, 0.25],
            },
            str(hourly_path),
        )

    print(f'  {year}: computing daily means...')
    ds = xr.open_dataset(hourly_path, chunks={'time': 24})
    ds_daily = ds.resample(time='1D').mean()
    ds_daily = ds_daily.astype(np.float32)
    ds_daily.to_netcdf(daily_path)
    ds.close()
    hourly_path.unlink()
    print(f'  {year}: saved daily means to {daily_path}')
    return daily_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Final merged output path')
    parser.add_argument('--start_year', type=int, default=2010)
    parser.add_argument('--end_year', type=int, default=2019)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    yearly_files = []
    for year in range(args.start_year, args.end_year + 1):
        daily_path = download_era5_wind_year(client, year, output_dir)
        yearly_files.append(daily_path)

    print('Merging yearly files...')
    ds = xr.open_mfdataset(yearly_files, combine='by_coords')
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.astype(np.float32)
    ds.to_netcdf(output_path)
    print(f'Saved merged file to {output_path}')

    for f in yearly_files:
        f.unlink()
        print(f'  Removed {f}')


if __name__ == '__main__':
    main()
