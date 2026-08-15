"""
Flow matching forecast model variants for GPU-based patch reconstruction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import xarray as xr

from src.lit_flow_matching import LitFlowMatching_SST_SLA
from src.lit_flow_matching_stochastic import LitFlowMatchingStochastic_SST_SLA
from src.lit_flow_matching_x0pred import LitFlowMatchingX0Pred_SST_SLA
from src.lit_flow_matching_improved import LitFlowMatchingImproved_SST_SLA
from src.lit_flow_matching_glofm import LitFlowMatchingGloFM_SST_SLA
from src.lit_flow_matching_x0cond import LitFlowMatchingX0Cond_SST_SLA


class FlowMatchingForecastPatchGPU_SST_SLA_INOUT(LitFlowMatching_SST_SLA):
    """
    Flow matching model with GPU-accelerated patch reconstruction for SST+SLA.
    Reconstructs SST and SLA separately to avoid OOM (~40 GB each pass).
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        # SST reconstruction pass
        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        # SLA reconstruction pass
        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())


class FlowMatchingStochasticForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingStochastic_SST_SLA):
    """
    Stochastic Flow Matching: noise -> data direction (CIA-Oceanix 4DVarNet-FM style).

    Generates ensemble members by sampling different noise vectors at inference.
    The deterministic forecast x_0 is used as conditioning, not as starting point.
    Each call to forward() / sample() produces a different stochastic ensemble member.
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        # SST reconstruction pass
        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        # SLA reconstruction pass
        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())


class FlowMatchingX0PredForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingX0Pred_SST_SLA):
    """
    Expected-x0 parameterization FM (CIA-Oceanix GradSolver_FM style).

    UNet predicts clean data x_1 directly (not velocity).
    Uses probability-flow ODE at inference.
    Generates ensemble members by sampling fresh noise each call.
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())


class FlowMatchingImprovedForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingImproved_SST_SLA):
    """
    Improved FM: EMA weights + self-attention (via FlowMatchingVelocityUNetAttn) +
    Heun's method + non-uniform timestep schedule (Wetherell 2026).

    Use with velocity_net: src.models_flow_matching_attn.FlowMatchingVelocityUNetAttn
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())


class FlowMatchingGloFMForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingGloFM_SST_SLA):
    """
    GloFM logit-normal timestep + EMA + Heun for SST+SLA global patch reconstruction.

    Conditional FM (past obs + deterministic x_0 as condition), trained with
    GloFM's logit-normal timestep sampling. Fully consistent with end-to-end
    SST+SLA forecasting from sparse observations.
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()
        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())


class FlowMatchingX0CondForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingX0Cond_SST_SLA):
    """
    FM conditioned on pretrained x_0 only — Ronan's suggestion.

    Condition = x_0 (58 ch) only, no sparse obs concatenation.
    velocity_net: n_cond_channels=58 (half of the other variants).
    """
    def __init__(self, *args, rec_weight_fn, output_leadtime_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = False

    def get_dT(self):
        return self.rec_weight.size()[0]

    def clear_gpu_mem(self):
        del self.pretrained.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        test_data_sst = torch.cat(self.test_data_sst).cuda()
        sst_das = {}
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sst, fw)
            if isinstance(rec, list):
                rec = rec[0]
            sst_das[i] = rec
        del test_data_sst
        torch.cuda.empty_cache()

        test_data_sla = torch.cat(self.test_data_sla).cuda()
        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_sla = self.trainer.test_dataloaders.dataset.reconstruct(test_data_sla, fw)
            if isinstance(rec_sla, list):
                rec_sla = rec_sla[0]

            ds_sst = sst_das[i].assign_coords(v0=['sst']).to_dataset(dim='v0')
            ds_sla = rec_sla.assign_coords(v0=['sla']).to_dataset(dim='v0')
            test_data_leadtime = xr.merge([ds_sst, ds_sla])

            if self.logger:
                test_data_leadtime.to_netcdf(
                    Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                )
                print(Path(self.logger.log_dir) / f'test_data_{i + 14}.nc')

            if self.pre_metric_fn is not None:
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

        if metrics:
            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())
