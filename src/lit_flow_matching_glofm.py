"""
GloFM: GLORYS Flow-Matching emulator (Garcia et al., VISAPP 2026).

Adapted to the SST+SLA end-to-end setup from the original paper's approach:

  Training (unconditional):
    - Learn p(SST, SLA) from clean simulation data only
    - NO conditioning on observations or pre-trained model during training
    - Logit-normal timestep: z ~ N(0,1), s = sigmoid(z)
      → denser sampling around s=0.5, downweights extremes
    - OT path: x_s = s*x_1 + (1-s)*eps,  v_target = x_1 - eps
    - EMA (decay=0.999) for stable inference

  Inference (MMPS — Moment-Matching Posterior Sampling, Rozet et al. 2024):
    - Start from x_0 ~ N(0, I)
    - At each ODE step, add gradient of log p(y|x_s) to the FM velocity
    - Posterior score (linearized Tweedie, Jacobian ≈ I):
        E[x_1|x_s] = x_s + (1-s) * v_θ(x_s, s)
        ∇_xs log p(y|xs) ≈ obs_mask * (y - E[x_1|xs]) / (σ_y² + (1-s)²)
    - Integrator: Heun's 2nd-order method + non-uniform schedule

  Observations used for conditioning (MMPS):
    - Sparse SST L3 (past half of temporal window, NaN elsewhere)
    - Sparse SLA L3 (past half of temporal window, NaN elsewhere)
    - No pre-trained deterministic model needed!

Key difference from previous FM variants:
  - Previous: conditional training (obs concatenated) → deterministic at inference
  - GloFM:    unconditional training → stochastic + MMPS conditioning at inference
              → more flexible (any observation type can be assimilated)
"""
import copy
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import xarray as xr


def _logit_normal_s(n, device):
    """Logit-normal timestep sampling (GloFM paper Section 3.2)."""
    z = torch.randn(n, device=device)
    return torch.sigmoid(z)


def _nonuniform_schedule(n_steps, device):
    """Non-uniform schedule: t_i = 1-(1-i/n)^2 (Wetherell 2026)."""
    i = torch.arange(n_steps + 1, device=device, dtype=torch.float32)
    return 1.0 - (1.0 - i / n_steps) ** 2


class LitFlowMatchingGloFM_SST_SLA(pl.LightningModule):
    """
    GloFM-style unconditional FM with MMPS posterior sampling at inference.

    Args:
        velocity_net:         unconditioned velocity UNet (FlowMatchingVelocityUNetUnconditional)
        opt_fn:               optimizer factory
        rec_weight:           spatial/temporal reconstruction weight (numpy array)
        n_inference_steps:    ODE steps at test time
        val_n_inference_steps: ODE steps at validation time
        ema_decay:            EMA weight decay (default 0.999, from GloFM/Wetherell)
        sigma_y_sst:          SST observation noise in normalized space
                              (default 0.5 ≈ 0.23°C at std=0.47)
        sigma_y_sla:          SLA observation noise in normalized space
                              (default 0.3 ≈ 0.015m at std=0.051)
        obs_lambda:           MMPS correction strength multiplier
        pre_metric_fn:        xarray selection for metrics
        test_metrics:         dict of metric functions
        norm_stats:           (mean, std) for denormalization
        persist_rw:           register rec_weight as buffer (True) or plain tensor
    """
    def __init__(self, velocity_net, opt_fn, rec_weight, n_inference_steps=20,
                 val_n_inference_steps=None, ema_decay=0.999,
                 sigma_y_sst=0.5, sigma_y_sla=0.3, obs_lambda=1.0,
                 pre_metric_fn=None, test_metrics=None, norm_stats=None,
                 persist_rw=True):
        super().__init__()
        self.velocity_net = velocity_net
        self._opt_fn = opt_fn
        self.n_inference_steps = n_inference_steps
        self.val_n_inference_steps = val_n_inference_steps or n_inference_steps
        self.ema_decay = ema_decay
        self.sigma_y_sst = sigma_y_sst
        self.sigma_y_sla = sigma_y_sla
        self.obs_lambda = obs_lambda
        self.pre_metric_fn = pre_metric_fn
        self.norm_stats = norm_stats
        self.metrics = test_metrics or {}
        self.test_quantities = ['out']

        if persist_rw:
            self.register_buffer('rec_weight', torch.from_numpy(rec_weight).float())
        else:
            self.rec_weight = torch.from_numpy(rec_weight).float()

        # EMA shadow copy (on CPU, moved to device on first update)
        self._ema_shadow = copy.deepcopy(self.velocity_net)
        self._ema_shadow.requires_grad_(False)
        self._ema_initialised = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_target(self, batch):
        """Ground truth x_1: SST + SLA stacked → [B, 58, H, W]."""
        return torch.stack([batch.tgt, batch.tgt_sla], dim=1).view(
            batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1)
        )

    def _mask_future(self, batch):
        """Mask second half of temporal dimension (future) with NaN."""
        new_input = batch.input.clone()
        T = new_input.size(1)
        new_input[:, T // 2:] = np.nan
        batch = batch._replace(input=new_input)
        new_sla = batch.input_sla.clone()
        T = new_sla.size(1)
        new_sla[:, T // 2:] = np.nan
        return batch._replace(input_sla=new_sla)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema()

    def _update_ema(self):
        if not self._ema_initialised:
            for ep, p in zip(self._ema_shadow.parameters(), self.velocity_net.parameters()):
                ep.data.copy_(p.data)
            self._ema_initialised = True
        else:
            decay = self.ema_decay
            for ep, p in zip(self._ema_shadow.parameters(), self.velocity_net.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def _ema_net(self):
        device = next(self.velocity_net.parameters()).device
        return self._ema_shadow.to(device)

    # ------------------------------------------------------------------
    # Training (unconditional, logit-normal timestep)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        # Unconditional: only use ground truth — no observation conditioning
        x_1 = self._get_target(batch)
        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)

        eps = torch.randn_like(x_1)

        # Logit-normal timestep (GloFM paper Section 3.2)
        s = _logit_normal_s(x_1.size(0), x_1.device)
        s_e = s[:, None, None, None]

        # OT path: x_s = s*x1 + (1-s)*eps
        x_s = s_e * x_1 + (1.0 - s_e) * eps
        v_target = x_1 - eps  # OT velocity: constant for linear path

        # Unconditional velocity prediction (NO condition argument)
        v_pred = self.velocity_net(x_s, s)

        loss = (valid * (v_pred - v_target) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # MMPS sampling (GloFM Section 3.3.1 + Heun's method)
    # ------------------------------------------------------------------

    def _mmps_correction(self, x_s, s_val, obs_sst, obs_sla, sst_mask, sla_mask, net):
        """
        Linearized MMPS posterior score (Jacobian ≈ I approximation).

        Equation (10) from GloFM paper, linearized:
            ∇_xs log p(y|xs) ≈ A^T (Σ_y + A V[x1|xs] A^T)^{-1} (y - A E[x1|xs])
        With Jacobian ≈ I and diagonal A (observation mask):
            = obs_mask * (y - x1_pred) / (σ_y² + (1-s)²)

        Returns correction term to add to ODE velocity.
        """
        v_pred = net(x_s, torch.full((x_s.size(0),), s_val, device=x_s.device))
        x_1_pred = x_s + (1.0 - s_val) * v_pred  # Tweedie estimate

        # Prior variance at this timestep: V[x1|xs] = (1-s)^2 for OT path
        prior_var = max((1.0 - s_val) ** 2, 1e-6)

        x_1_sst = x_1_pred[:, :29]
        x_1_sla = x_1_pred[:, 29:]

        # Correction weight per variable
        w_sst = prior_var / (self.sigma_y_sst ** 2 + prior_var)
        w_sla = prior_var / (self.sigma_y_sla ** 2 + prior_var)

        corr_sst = sst_mask * w_sst * (obs_sst - x_1_sst)
        corr_sla = sla_mask * w_sla * (obs_sla - x_1_sla)

        correction = torch.cat([corr_sst, corr_sla], dim=1)
        return v_pred, correction

    def _sample_mmps(self, obs_sst, obs_sla, n_steps=None, use_ema=False):
        """
        Generate one ensemble member via MMPS + Heun's method.

        Args:
            obs_sst:  [B, 29, H, W] SST observations (NaN where unobserved)
            obs_sla:  [B, 29, H, W] SLA observations (NaN where unobserved)
            n_steps:  number of ODE steps
            use_ema:  use EMA weights for the network call

        Returns:
            x_1: [B, 58, H, W] reconstructed SST+SLA
        """
        n_steps = n_steps or self.n_inference_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        B, _, H, W = obs_sst.shape
        sst_mask = torch.isfinite(obs_sst).float()
        sla_mask = torch.isfinite(obs_sla).float()
        y_sst = torch.nan_to_num(obs_sst)
        y_sla = torch.nan_to_num(obs_sla)

        # Start from pure noise
        x = torch.randn(B, 58, H, W, device=obs_sst.device)

        ts = _nonuniform_schedule(n_steps, obs_sst.device)

        with torch.no_grad():
            for i in range(n_steps):
                s_curr = float(ts[i])
                s_next = float(ts[i + 1])
                dt = s_next - s_curr

                # Predictor: FM velocity + MMPS correction at s_curr
                k1_v, corr1 = self._mmps_correction(
                    x, s_curr, y_sst, y_sla, sst_mask, sla_mask, net
                )
                k1 = k1_v + self.obs_lambda * corr1

                # Heun corrector: step forward, evaluate at s_next
                x_pred = x + dt * k1
                k2_v, corr2 = self._mmps_correction(
                    x_pred, s_next, y_sst, y_sla, sst_mask, sla_mask, net
                )
                k2 = k2_v + self.obs_lambda * corr2

                x = x + dt * (k1 + k2) / 2.0

        net.train()
        return x

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        batch = self._mask_future(batch)
        x_1 = self._get_target(batch)
        valid = torch.isfinite(x_1)
        x_1_clean = torch.nan_to_num(x_1)

        x_sample = self._sample_mmps(
            batch.input, batch.input_sla,
            n_steps=self.val_n_inference_steps,
            use_ema=True,
        )
        loss = (valid * (x_sample - x_1_clean) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_refined = self._sample_mmps(
            batch.input, batch.input_sla,
            n_steps=self.n_inference_steps,
            use_ema=True,
        )

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

        B, _, H, W = x_refined.shape
        out_2var = x_refined.view(B, 2, 29, H, W).detach().cpu()
        out_sst = out_2var[:, 0:1] * s_sst + m_sst
        out_sla = out_2var[:, 1:2] * s_sla + m_sla
        self.test_data_sst.append(out_sst)
        self.test_data_sla.append(out_sla)

    # ------------------------------------------------------------------
    # Optimiser + forward
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self._opt_fn(self)

    def forward(self, batch):
        """Generate one ensemble member."""
        batch = self._mask_future(batch)
        return self._sample_mmps(
            batch.input, batch.input_sla,
            use_ema=True,
        )
