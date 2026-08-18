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
from src.lit_flow_matching_x0past import LitFlowMatchingX0Past_SST_SLA


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


class FlowMatchingGloFMEnsembleForecastPatchGPU_SST_SLA_INOUT(FlowMatchingGloFMForecastPatchGPU_SST_SLA_INOUT):
    """
    GloFM FM with multi-member ensemble inference.

    Generates n_ensemble_members independent samples per 15-day window by drawing
    fresh noise for each member. Saves per lead time:
      - sst_mean / sla_mean : ensemble mean (best single estimate)
      - sst_std  / sla_std  : ensemble std  (uncertainty / spread)

    All members use EMA weights at inference (inherited from LitFlowMatchingImproved).
    """

    def __init__(self, *args, n_ensemble_members=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_ensemble_members = n_ensemble_members

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)
        reference = torch.zeros_like(x_0)

        members = []
        for _ in range(self.n_ensemble_members):
            x_member = self._sample(condition, reference, use_ema=True)
            members.append(x_member)

        # [n_members, B, 58, H, W]
        members = torch.stack(members, dim=0)
        x_mean = members.mean(dim=0)
        x_std  = members.std(dim=0)

        if batch_idx == 0:
            self.test_data_sst_mean = []
            self.test_data_sla_mean = []
            self.test_data_sst_std  = []
            self.test_data_sla_std  = []

        dm = self.trainer.datamodule
        if hasattr(dm, 'norm_stats_per_var') and dm.normalize_per_var:
            m_sst, s_sst = dm.norm_stats_per_var()['tgt']
            m_sla, s_sla = dm.norm_stats_per_var()['tgt_sla']
        elif self.norm_stats is not None:
            m_sst, s_sst = self.norm_stats
            m_sla, s_sla = m_sst, s_sst
        else:
            m_sst, s_sst = 0, 1
            m_sla, s_sla = 0, 1

        def _split(x):
            v = x.view(x.size(0), 2, 29, x.size(-2), x.size(-1)).detach().cpu()
            sst = v[:, 0:1] * s_sst + m_sst
            sla = v[:, 1:2] * s_sla + m_sla
            return sst, sla

        sst_mean, sla_mean = _split(x_mean)
        sst_std,  sla_std  = _split(x_std)

        self.test_data_sst_mean.append(sst_mean)
        self.test_data_sla_mean.append(sla_mean)
        self.test_data_sst_std.append(sst_std)
        self.test_data_sla_std.append(sla_std)

    def on_test_epoch_end(self):
        self.clear_gpu_mem()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        output_start = 0 if self.output_only_forecast else -14
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start

        sst_mean_t = torch.cat(self.test_data_sst_mean).cuda()
        sla_mean_t = torch.cat(self.test_data_sla_mean).cuda()
        sst_std_t  = torch.cat(self.test_data_sst_std).cuda()
        sla_std_t  = torch.cat(self.test_data_sla_std).cuda()

        metrics = []
        for i in range(output_start, 7):
            fw = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())

            def _rec(t):
                r = self.trainer.test_dataloaders.dataset.reconstruct(t, fw)
                return r[0] if isinstance(r, list) else r

            rec_sst_mean = _rec(sst_mean_t)
            rec_sla_mean = _rec(sla_mean_t)
            rec_sst_std  = _rec(sst_std_t)
            rec_sla_std  = _rec(sla_std_t)

            ds = xr.merge([
                rec_sst_mean.assign_coords(v0=['sst_mean']).to_dataset(dim='v0'),
                rec_sla_mean.assign_coords(v0=['sla_mean']).to_dataset(dim='v0'),
                rec_sst_std.assign_coords(v0=['sst_std']).to_dataset(dim='v0'),
                rec_sla_std.assign_coords(v0=['sla_std']).to_dataset(dim='v0'),
            ])

            if self.logger:
                out_path = Path(self.logger.log_dir) / f'test_data_{i + 14}.nc'
                ds.to_netcdf(out_path)
                print(out_path)

            if self.pre_metric_fn is not None:
                ds_mean = xr.merge([
                    rec_sst_mean.assign_coords(v0=['sst']).to_dataset(dim='v0'),
                    rec_sla_mean.assign_coords(v0=['sla']).to_dataset(dim='v0'),
                ])
                metric_data = ds_mean.pipe(self.pre_metric_fn)
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


class FlowMatchingX0PastForecastPatchGPU_SST_SLA_INOUT(LitFlowMatchingX0Past_SST_SLA):
    """
    FM conditioned on past x_0 only (future half zeroed).

    Forces the FM to learn genuine forecasting from the distribution
    rather than correcting the deterministic forecast for future steps.
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
