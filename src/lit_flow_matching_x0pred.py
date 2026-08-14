"""
Stochastic Flow Matching — expected-x0 parameterization (CIA-Oceanix style).

Follows GradSolver_FM from 4dvarnet-global-mapping/paul_dev branch:
  - Direction: noise -> data (linear interpolant)
  - UNet predicts x_1 (clean data) directly, not the velocity
  - Loss: MSE(unet(x_tau, tau, condition) - x_1)
  - Inference: probability-flow ODE using predicted x_1

Training:
  1. Frozen pre-trained model produces x_0 (deterministic forecast) — used as CONDITION
  2. x_1 = ground truth (SST + SLA targets)
  3. epsilon ~ N(0, sigma_prior)
  4. tau ~ U(0,1), x_tau = (1-tau)*epsilon + tau*x_1
  5. UNet predicts x_1_pred = unet(x_tau, tau, condition)
  6. Loss = masked MSE(x_1_pred, x_1)

Inference (one ensemble member):
  epsilon ~ N(0, sigma_prior)
  x = epsilon
  for step in 1..n_inference_steps:
      t     = step / n_inference_steps
      t_prev= (step-1) / n_inference_steps
      alpha, dalpha = (1-t), -1
      beta,  dbeta  =  t,    +1
      x_1_pred = unet(x, t_prev, condition)
      x_0_back = (x - beta_prev * x_1_pred) / alpha_prev   # back-infer noise
      x = x + (1/n_steps) * (dalpha * x_0_back + dbeta * x_1_pred)
"""
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import xarray as xr


class LitFlowMatchingX0Pred_SST_SLA(pl.LightningModule):
    def __init__(self, velocity_net, pretrained_model, pretrained_ckpt_path,
                 opt_fn, rec_weight, n_inference_steps=10, val_n_inference_steps=None,
                 sigma_prior=1.0, pre_metric_fn=None, test_metrics=None,
                 norm_stats=None, persist_rw=True):
        super().__init__()
        self.velocity_net = velocity_net  # predicts x_1, not velocity
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
        """Get x_0 from frozen pre-trained model — used as condition."""
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

        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)

        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)
        x_0 = torch.nan_to_num(x_0)

        epsilon = torch.randn_like(x_1) * self.sigma_prior

        # same tau for all items in batch (following CIA-Oceanix)
        tau = torch.rand(1, device=x_1.device).expand(x_1.size(0))
        tau_e = tau[:, None, None, None]
        x_tau = (1 - tau_e) * epsilon + tau_e * x_1

        condition = self._get_condition(batch, x_0)

        # predict x_1 directly (expected-x0 parameterization)
        x_1_pred = self.velocity_net(x_tau, tau, condition)

        # loss: MSE between predicted and true clean data
        loss = (valid * (x_1_pred - x_1) ** 2).sum() / valid.sum().clamp(min=1)

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
        x_sample = self._sample(condition, x_1, n_steps=self.val_n_inference_steps)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def _sample(self, condition, reference, n_steps=None):
        """
        Generate one ensemble member via probability-flow ODE.
        Follows GradSolver_FM update rule from CIA-Oceanix.
        """
        n_steps = n_steps if n_steps is not None else self.n_inference_steps
        x = torch.randn_like(reference) * self.sigma_prior

        for step in range(1, n_steps + 1):
            t_prev = (step - 1) / n_steps
            t_curr = step / n_steps

            alpha_prev = 1 - t_prev
            beta_prev  = t_prev
            dalpha = -1.0
            dbeta  = +1.0

            tau = torch.full((x.size(0),), t_prev, device=x.device)
            x_1_pred = self.velocity_net(x, tau, condition)

            # back-infer noise from current state and predicted x_1
            if alpha_prev > 1e-6:
                x_0_back = (x - beta_prev * x_1_pred) / alpha_prev
            else:
                x_0_back = torch.zeros_like(x)

            x = x + (1.0 / n_steps) * (dalpha * x_0_back + dbeta * x_1_pred)

        return x

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)
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
