"""
Saliency maps for the SST+SLA multivariate forecast model.

Computes input gradients to visualize which spatial/temporal regions
of the SST and SLA inputs contribute most to the forecast output.

Usage:
    python contrib/explainability/saliency_maps.py \
        --pipeline_config config/xp/ose_pipeline_proc_2023_global_1patch_SST_SLA_INOUT.yaml \
        --output_dir /Odyssey/public/glorys/explainability/ \
        --time_idx 180 \
        --leadtimes 0 1 3 6
"""
import argparse
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contrib.ose_pipeline.rec_utils import call_cfg_key
from contrib.ose_pipeline.ose_rec_pipeline import setup_model_config_SST_SLA_INOUT
from contrib.ose_pipeline.ose_full_pipeline import setup_config


def compute_saliency(model, batch, target_var='sst', leadtime=0):
    """
    Compute saliency maps via input gradients.

    target_var: 'sst' or 'sla' — which output variable to backprop from
    leadtime: which forecast day (0 = present, 1..6 = future days)
    """
    model.eval()

    input_sst = batch.input.clone().requires_grad_(True)
    input_sla = batch.input_sla.clone().requires_grad_(True)

    modified_batch = batch._replace(input=input_sst, input_sla=input_sla)
    mask_batch = model.mask_batch(modified_batch)

    out = model(batch=mask_batch)
    out = out.view(out.size()[0], 2, 29, *out.size()[-2:])

    var_idx = 0 if target_var == 'sst' else 1
    time_idx = 14 + leadtime

    target = out[:, var_idx, time_idx, :, :].mean()
    target.backward()

    saliency_sst = input_sst.grad.detach().abs().cpu().numpy()
    saliency_sla = input_sla.grad.detach().abs().cpu().numpy()

    return saliency_sst, saliency_sla


def plot_saliency(saliency_sst, saliency_sla, target_var, leadtime, output_path,
                  lat=None, lon=None):
    """Plot temporal-mean and spatial saliency maps."""
    sal_sst = saliency_sst[0]  # remove batch dim
    sal_sla = saliency_sla[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Temporal profile: mean saliency per input timestep
    axes[0, 0].plot(sal_sst.mean(axis=(1, 2)), label='SST input', color='tab:red')
    axes[0, 0].plot(sal_sla.mean(axis=(1, 2)), label='SLA input', color='tab:blue')
    axes[0, 0].axvline(x=14, color='gray', linestyle='--', label='Present (t=0)')
    axes[0, 0].set_xlabel('Input timestep')
    axes[0, 0].set_ylabel('Mean |gradient|')
    axes[0, 0].set_title(f'Temporal sensitivity — target: {target_var} leadtime +{leadtime}d')
    axes[0, 0].legend()

    # Spatial saliency: SST input (mean over time)
    sal_sst_spatial = sal_sst[:14].mean(axis=0)  # only past obs
    im1 = axes[0, 1].imshow(sal_sst_spatial, aspect='auto', cmap='hot', origin='lower')
    axes[0, 1].set_title('SST input saliency (spatial, past 14 days mean)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Spatial saliency: SLA input (mean over time)
    sal_sla_spatial = sal_sla[:14].mean(axis=0)
    im2 = axes[1, 0].imshow(sal_sla_spatial, aspect='auto', cmap='hot', origin='lower')
    axes[1, 0].set_title('SLA input saliency (spatial, past 14 days mean)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Ratio: SLA / (SST + SLA) contribution
    total = sal_sst_spatial + sal_sla_spatial + 1e-10
    ratio = sal_sla_spatial / total
    im3 = axes[1, 1].imshow(ratio, aspect='auto', cmap='RdBu', vmin=0, vmax=1, origin='lower')
    axes[1, 1].set_title('SLA relative importance: SLA / (SST + SLA)')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved saliency map to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_config', required=True,
                        help='Pipeline config (e.g. ose_pipeline_proc_2023_global_1patch_SST_SLA_INOUT.yaml)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--time_idx', type=int, default=180)
    parser.add_argument('--target_var', default='sst', choices=['sst', 'sla'])
    parser.add_argument('--leadtimes', nargs='+', type=int, default=[0, 1, 3, 6])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline config
    pipeline_cfg = OmegaConf.load(args.pipeline_config)

    # Extract paths from pipeline config
    model_config_path = pipeline_cfg.paths.model_config_path
    model_ckpt_path = pipeline_cfg.paths.model_ckpt_path
    sla_input_path = pipeline_cfg.paths.sla_input_path
    tgt_sla_path = pipeline_cfg.paths.tgt_sla_path
    ose_data_path = pipeline_cfg.paths.ose_data_path
    min_time = str(pipeline_cfg.specs.min_time)
    max_time = str(pipeline_cfg.specs.max_time)

    # Setup config the same way the pipeline does
    (
        _, _, _, _, gridded_input_path, _, _, _,
        min_time_offseted, max_time_offseted,
    ) = setup_config(
        min_time, max_time,
        pipeline_cfg.specs.time_day_crop,
        ose_data_path,
        pipeline_cfg.paths.rec_path,
        pipeline_cfg.paths.metrics_path,
        pipeline_cfg.xp_name,
        pipeline_cfg.data_name,
    )

    config = setup_model_config_SST_SLA_INOUT(
        model_config_path=model_config_path,
        gridded_input_path=gridded_input_path,
        sla_input_path=sla_input_path,
        tgt_sla_path=tgt_sla_path,
        rec_paths='',
        min_time=min_time,
        max_time=max_time,
        min_time_offseted=min_time_offseted,
        max_time_offseted=max_time_offseted,
        overwrite=True,
    )

    print('Loading model...')
    model = call_cfg_key(config, 'model')
    ckpt = torch.load(model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()

    print('Loading datamodule...')
    dm = call_cfg_key(config, 'datamodule')
    dm.setup(stage='test')

    print(f'Test dataset size: {len(dm.test_ds)}')
    batch = dm.test_ds[args.time_idx]
    batch = type(batch)(*[torch.from_numpy(np.array(t)).unsqueeze(0).cuda() for t in batch])

    for leadtime in args.leadtimes:
        print(f'Computing saliency for {args.target_var} leadtime +{leadtime}d...')
        saliency_sst, saliency_sla = compute_saliency(
            model, batch, target_var=args.target_var, leadtime=leadtime
        )
        plot_saliency(
            saliency_sst, saliency_sla,
            target_var=args.target_var,
            leadtime=leadtime,
            output_path=output_dir / f'saliency_{args.target_var}_ldt{leadtime}.png',
        )

    # Save raw saliency as NetCDF for further analysis
    print('Saving raw saliency data...')
    saliency_sst, saliency_sla = compute_saliency(
        model, batch, target_var=args.target_var, leadtime=args.leadtimes[0]
    )
    ds = xr.Dataset({
        'saliency_sst': (['time', 'lat', 'lon'], saliency_sst[0]),
        'saliency_sla': (['time', 'lat', 'lon'], saliency_sla[0]),
    })
    ds.to_netcdf(output_dir / f'saliency_raw_{args.target_var}.nc')
    print(f'Saved raw saliency to {output_dir / f"saliency_raw_{args.target_var}.nc"}')


if __name__ == '__main__':
    main()
