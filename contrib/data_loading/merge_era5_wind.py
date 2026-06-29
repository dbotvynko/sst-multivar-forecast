"""
Merge ERA5 wind monthly ZIP files into a single NetCDF.

The CDS API delivers each month as a ZIP containing two separate files:
  - 10m_u_component_of_wind_stream-oper_daily-mean.nc
  - 10m_v_component_of_wind_0_daily-mean.nc

This script unzips each month to a temp directory, saves a proper
per-month NetCDF, then uses open_mfdataset to lazily merge all months.

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
    tmp_dir = input_dir / '_tmp_merged'
    tmp_dir.mkdir(exist_ok=True)

    zip_files = sorted(input_dir.glob('era5_10m_wind_daily_*.nc'))
    print(f'Found {len(zip_files)} monthly ZIP files')

    merged_monthly = []
    for zf in zip_files:
        out_monthly = tmp_dir / f'merged_{zf.stem}.nc'
        if out_monthly.exists():
            print(f'  {zf.name}: already merged, skipping')
            merged_monthly.append(out_monthly)
            continue

        print(f'Processing {zf.name}...')
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(tmpdir)

            nc_files = list(Path(tmpdir).glob('*.nc'))
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            if 'latitude' in ds.dims:
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds.load().astype(np.float32).to_netcdf(out_monthly)

        merged_monthly.append(out_monthly)

    print('Opening all merged monthly files lazily...')
    ds = xr.open_mfdataset(merged_monthly, combine='by_coords')
    ds = ds.sortby('time')

    print('Writing final merged file...')
    ds.to_netcdf(args.output)
    print(f'Saved to {args.output}')
    print(ds)

    print('Cleaning up temp files...')
    for f in merged_monthly:
        f.unlink()
    tmp_dir.rmdir()
    print('Done!')


if __name__ == '__main__':
    main()
