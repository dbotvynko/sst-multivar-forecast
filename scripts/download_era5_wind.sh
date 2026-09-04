#!/bin/bash
#SBATCH --partition=Odyssey_CPU
#SBATCH --job-name=era5_wind_dl
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_era5_wind_dl.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error.txt
# Selected partition
# Name for the job
# Resources asked
# %j for jobid
#
# No GPU needed - this just downloads data from CDS. Adjust --partition
# if Odyssey_CPU isn't the right one for this cluster.

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12-MULTIVARIATE

conda activate cds_download

python contrib/data_loading/download_era5_wind.py \
    --output-dir /Odyssey/public/era5/2010_2019 \
    --start-year 2010 --end-year 2019
