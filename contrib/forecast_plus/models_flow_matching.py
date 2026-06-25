"""
Flow matching forecast model variants for GPU-based patch reconstruction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import xarray as xr

from src.lit_flow_matching import LitFlowMatching_SST_SLA


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
