"""
PyTorch Lightning module for flow matching refinement of SST+SLA forecasts.

The flow matching model learns to refine deterministic forecasts from a
pre-trained SST+SLA model by learning a velocity field in output space.

Training:
  1. Frozen pre-trained model produces x_0 (deterministic forecast)
  2. x_1 = ground truth (SST + SLA targets)
  3. Sample t ~ U(0,1), interpolate x_t = (1-t)*x_0 + t*x_1
  4. Velocity net predicts v(x_t, t, condition) with target u = x_1 - x_0
  5. Loss = MSE(v_pred, u)

Inference:
  Euler integration of the ODE from x_0 to x_1 in n_steps.
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import xarray as xr


class LitFlowMatching_SST_SLA(pl.LightningModule):
    def __init__(self, velocity_net, pretrained_model, pretrained_ckpt_path,
                 opt_fn, rec_weight, n_inference_steps=10, val_n_inference_steps=None,
                 pre_metric_fn=None, test_metrics=None, norm_stats=None,
                 persist_rw=True):
        super().__init__()
        self.velocity_net = velocity_net
        self.n_inference_steps = n_inference_steps
        self.val_n_inference_steps = val_n_inference_steps if val_n_inference_steps is not None else n_inference_steps
        self._opt_fn = opt_fn
        self.pre_metric_fn = pre_metric_fn
        self.norm_stats = norm_stats

        if persist_rw:
            self.register_buffer('rec_weight', torch.from_numpy(rec_weight).float())
        else:
            self.rec_weight = torch.from_numpy(rec_weight).float()

        self.test_quantities = ['out']
        self.metrics = test_metrics or {}

        # Load and freeze pre-trained deterministic model
        self.pretrained = pretrained_model
        ckpt = torch.load(pretrained_ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        self.pretrained.load_state_dict(state_dict)
        self.pretrained.eval()
        for p in self.pretrained.parameters():
            p.requires_grad_(False)

    def _mask_future(self, batch):
        new_input = batch.input.clone()
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = np.nan
        mask_batch = batch._replace(input=new_input)
        new_input = batch.input_sla.clone()
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = np.nan
        mask_batch = mask_batch._replace(input_sla=new_input)
        return mask_batch

    def _get_deterministic_forecast(self, batch):
        """Get x_0 from frozen pre-trained model."""
        with torch.no_grad():
            out = self.pretrained(batch=batch)
        return out.view(out.size(0), 2, 29, out.size(-2), out.size(-1))

    def _get_condition(self, batch, x_0=None):
        """Past observations + deterministic forecast as condition."""
        parts = [
            torch.nan_to_num(batch.input),
            torch.nan_to_num(batch.input_sla),
        ]
        if x_0 is not None:
            parts.append(torch.nan_to_num(x_0))
        return torch.cat(parts, dim=1)

    def _get_target(self, batch):
        """Ground truth output: SST + SLA targets stacked as [B, 2, 29, H, W] then flattened."""
        return torch.stack([batch.tgt, batch.tgt_sla], dim=1)

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        # x_0: deterministic forecast [B, 2, 29, H, W]
        x_0 = self._get_deterministic_forecast(batch)
        # x_1: ground truth [B, 2, 29, H, W]
        x_1 = self._get_target(batch)
        # flatten var dim: [B, 58, H, W]
        x_0 = x_0.view(x_0.size(0), -1, x_0.size(-2), x_0.size(-1))
        x_1 = x_1.view(x_1.size(0), -1, x_1.size(-2), x_1.size(-1))

        # mask for valid (non-NaN) pixels in both x_0 and x_1
        valid = torch.isfinite(x_0) & torch.isfinite(x_1)
        x_0 = torch.nan_to_num(x_0)
        x_1 = torch.nan_to_num(x_1)

        # condition: past observations + deterministic forecast [B, 116, H, W]
        condition = self._get_condition(batch, x_0=x_0)

        # sample t ~ U(0,1)
        t = torch.rand(x_0.size(0), device=x_0.device)

        # interpolate
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # target velocity
        u = x_1 - x_0

        # predict velocity
        v_pred = self.velocity_net(x_t, t, condition)

        # masked MSE loss: only on valid pixels
        loss = (valid * (v_pred - u) ** 2).sum() / valid.sum().clamp(min=1)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.norm_stats is not None:
            m, s = self.norm_stats
            self.log('train_mse', loss * s**2, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)
        x_0 = x_0.view(x_0.size(0), -1, x_0.size(-2), x_0.size(-1))
        x_1 = x_1.view(x_1.size(0), -1, x_1.size(-2), x_1.size(-1))

        valid = torch.isfinite(x_0) & torch.isfinite(x_1)
        x_0 = torch.nan_to_num(x_0)
        x_1 = torch.nan_to_num(x_1)

        condition = self._get_condition(batch, x_0=x_0)

        # evaluate with Euler integration (more steps than training for better val signal)
        x_refined = self._euler_integrate(x_0, condition, n_steps=self.val_n_inference_steps)

        loss = (valid * (x_refined - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.norm_stats is not None:
            m, s = self.norm_stats
            self.log('val_mse', loss * s**2, prog_bar=True, on_step=False, on_epoch=True)

    def _euler_integrate(self, x_0, condition, n_steps=None):
        """Euler ODE integration from x_0 (deterministic forecast) to refined forecast."""
        n_steps = n_steps if n_steps is not None else self.n_inference_steps
        dt = 1.0 / n_steps
        x_t = x_0
        for i in range(n_steps):
            t = torch.full((x_t.size(0),), i * dt, device=x_t.device)
            v = self.velocity_net(x_t, t, condition)
            x_t = x_t + v * dt
        return x_t

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_0 = x_0.view(x_0.size(0), -1, x_0.size(-2), x_0.size(-1))
        condition = self._get_condition(batch, x_0=x_0)

        x_refined = self._euler_integrate(x_0, condition)

        if batch_idx == 0:
            self.test_data_sst = []
            self.test_data_sla = []

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

        out_2var = x_refined.view(x_refined.size(0), 2, 29, x_refined.size(-2), x_refined.size(-1)).detach().cpu()
        out_sst = out_2var[:, 0:1] * s_sst + m_sst
        out_sla = out_2var[:, 1:2] * s_sla + m_sla
        self.test_data_sst.append(out_sst)
        self.test_data_sla.append(out_sla)

    def configure_optimizers(self):
        return self._opt_fn(self)

    def forward(self, batch):
        batch = self._mask_future(batch)
        x_0 = self._get_deterministic_forecast(batch)
        x_0 = x_0.view(x_0.size(0), -1, x_0.size(-2), x_0.size(-1))
        condition = self._get_condition(batch)
        return self._euler_integrate(x_0, condition)
