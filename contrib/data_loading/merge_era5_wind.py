"""
Merge ERA5 wind monthly ZIP files into a single NetCDF.

The CDS API delivers each month as a ZIP containing two separate files:
  - 10m_u_component_of_wind_stream-oper_daily-mean.nc
  - 10m_v_component_of_wind_0_daily-mean.nc

This script unzips each month to a temp directory, merges u10 and v10,
then concatenates all months into one file.

Usage:
    python contrib/data_loading/merge_era5_wind.py \
        --input_dir /Odyssey/public/era5/2017_2019/ \
        --output /Odyssey/public/era5/2017_2019/era5_10m_wind_daily_mean_0.25deg.nc
"""
import argparse
import zipfile
import tempfile
import xarray as xr
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    zip_files = sorted(input_dir.glob('era5_10m_wind_daily_*.nc'))
    print(f'Found {len(zip_files)} monthly ZIP files')

    monthly_datasets = []
    for zf in zip_files:
        print(f'Processing {zf.name}...')
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(tmpdir)

            nc_files = list(Path(tmpdir).glob('*.nc'))
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            monthly_datasets.append(ds.load())

    print('Concatenating all months...')
    merged = xr.concat(monthly_datasets, dim='time').sortby('time')

    if 'latitude' in merged.dims:
        merged = merged.rename({'latitude': 'lat', 'longitude': 'lon'})

    merged = merged.astype(np.float32)
    merged.to_netcdf(args.output)
    print(f'Saved to {args.output}')
    print(merged)


if __name__ == '__main__':
    main()
