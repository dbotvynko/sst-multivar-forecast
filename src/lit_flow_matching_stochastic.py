"""
Stochastic Flow Matching Lightning module for SST+SLA forecast ensemble generation.

Follows the CIA-Oceanix 4DVarNet-FM approach (Fablet et al.):
  - Direction: noise -> data (opposite of the deterministic-refinement version)
  - Training: interpolate between Gaussian noise and ground truth, predict velocity
  - Inference: sample fresh noise -> Euler integration -> one ensemble member
  - Ensemble: call sample() N times with different noise draws

Training:
  1. Frozen pre-trained model produces x_0 (deterministic forecast) — used as CONDITION
  2. x_1 = ground truth (SST + SLA targets)
  3. epsilon ~ N(0, sigma_prior)
  4. Sample tau ~ U(0,1), interpolate x_tau = (1-tau)*epsilon + tau*x_1
  5. Target velocity: v_target = x_1 - epsilon  (constant for linear interpolant)
  6. velocity_net predicts v(x_tau, tau, condition) where condition = [obs_sst, obs_sla, x_0]
  7. Loss = masked MSE(v_pred, v_target)

Inference (one ensemble member):
  epsilon ~ N(0, sigma_prior)
  x = epsilon
  for step in range(n_inference_steps):
      tau = step / n_inference_steps
      v = velocity_net(x, tau, condition)
      x = x + v * dt
  return x   # one stochastic ensemble member
"""
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import xarray as xr


class LitFlowMatchingStochastic_SST_SLA(pl.LightningModule):
    def __init__(self, velocity_net, pretrained_model, pretrained_ckpt_path,
                 opt_fn, rec_weight, n_inference_steps=10, val_n_inference_steps=None,
                 sigma_prior=1.0, pre_metric_fn=None, test_metrics=None,
                 norm_stats=None, persist_rw=True):
        super().__init__()
        self.velocity_net = velocity_net
        self.n_inference_steps = n_inference_steps
        self.val_n_inference_steps = val_n_inference_steps if val_n_inference_steps is not None else n_inference_steps
        self.sigma_prior = sigma_prior
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
        """Get x_0 from frozen pre-trained model — used as condition, not starting point."""
        with torch.no_grad():
            out = self.pretrained(batch=batch)
        return out.view(out.size(0), -1, out.size(-2), out.size(-1))

    def _get_condition(self, batch, x_0):
        """Condition: past SST obs + past SLA obs + deterministic forecast x_0."""
        return torch.cat([
            torch.nan_to_num(batch.input),
            torch.nan_to_num(batch.input_sla),
            torch.nan_to_num(x_0),
        ], dim=1)

    def _get_target(self, batch):
        """Ground truth x_1: SST + SLA targets flattened to [B, 58, H, W]."""
        return torch.stack([batch.tgt, batch.tgt_sla], dim=1).view(
            batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1)
        )

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)   # [B, 58, H, W] — condition only
        x_1 = self._get_target(batch)                   # [B, 58, H, W] — ground truth

        # valid pixel mask (ocean only)
        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)
        x_0 = torch.nan_to_num(x_0)

        # sample noise from prior
        epsilon = torch.randn_like(x_1) * self.sigma_prior

        # sample tau ~ U(0,1) and interpolate noise -> data
        tau = torch.rand(x_1.size(0), device=x_1.device)
        tau_e = tau[:, None, None, None]
        x_tau = (1 - tau_e) * epsilon + tau_e * x_1

        # target velocity (constant for linear interpolant)
        v_target = x_1 - epsilon

        # condition: past obs + deterministic forecast
        condition = self._get_condition(batch, x_0)

        # predict velocity
        v_pred = self.velocity_net(x_tau, tau, condition)

        # masked MSE loss
        loss = (valid * (v_pred - v_target) ** 2).sum() / valid.sum().clamp(min=1)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)

        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)
        x_0 = torch.nan_to_num(x_0)

        condition = self._get_condition(batch, x_0)

        # generate one sample and measure MSE against ground truth
        x_sample = self._sample(condition, x_1, n_steps=self.val_n_inference_steps)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def _sample(self, condition, reference, n_steps=None):
        """Generate one ensemble member via Euler integration from noise to data."""
        n_steps = n_steps if n_steps is not None else self.n_inference_steps
        dt = 1.0 / n_steps
        x = torch.randn_like(reference) * self.sigma_prior
        for i in range(n_steps):
            tau = torch.full((x.size(0),), i * dt, device=x.device)
            v = self.velocity_net(x, tau, condition)
            x = x + v * dt
        return x

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)

        # reference shape for noise sampling
        reference = torch.zeros_like(x_0)
        x_refined = self._sample(condition, reference)

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
        """Generate one ensemble member."""
        batch = self._mask_future(batch)
        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)
        reference = torch.zeros_like(x_0)
        return self._sample(condition, reference)
