"""
Flow Matching for OSSE SLA reconstruction and forecasting.

OSSE = Observing System Simulation Experiment.
Fully controlled setting: both sparse and complete GLORYS12 fields are known.

Framework:
    x_source = masked GLORYS12 SLA (sparse obs + noise on obs pixels, 0 elsewhere)
    x_1      = complete GLORYS12 SLA (ground truth)
    FM learns the transport: sparse obs → complete field

Source distribution:
    Observed pixels:  obs + σ*ε  (noisy obs → ensemble diversity)
    Missing pixels:   0           (climatological mean = 0 in anomaly space)
    Future pixels:    0           (masked by _mask_future, causal forecasting)

Condition (static, throughout all ODE steps):
    nan_to_num(masked_sla, 0)  — tells FM where observations are

No pretrained model needed — GLORYS provides complete x_1 directly.

Training:
    - Logit-normal timestep (GloFM trick): z~N(0,1), tau=sigmoid(z)
    - EMA (decay=0.999) for stable inference
    - Heun's 2nd-order ODE + non-uniform schedule at inference

Batch structure (TrainingItem namedtuple):
    batch.input : masked SLA [B, 29, H, W]  (sparse, NaN where no satellite)
    batch.tgt   : complete SLA [B, 29, H, W] (full GLORYS field)
"""
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


def _nonuniform_schedule(n_steps, device):
    """t_i = 1 - (1 - i/n)^2 — denser steps near t=1."""
    i = torch.arange(n_steps + 1, device=device, dtype=torch.float32)
    return 1.0 - (1.0 - i / n_steps) ** 2


class LitFlowMatchingOSSE_SLA(pl.LightningModule):
    """
    Standalone FM for OSSE SLA — no pretrained model dependency.

    Args:
        velocity_net:      time-conditioned UNet (n_output_channels=29, n_cond_channels=29)
        opt_fn:            optimizer factory (partial)
        rec_weight:        spatial reconstruction weight [T, H, W]
        n_inference_steps: ODE steps at test time (default 50)
        val_n_inference_steps: ODE steps at val time (default 50)
        sigma_prior:       noise std on observed pixels (default 1.0)
        ema_decay:         EMA decay for inference weights (default 0.999)
        pre_metric_fn:     optional xarray selector for metrics
        test_metrics:      dict of metric functions
        persist_rw:        register rec_weight as buffer (default False)
    """

    def __init__(self, velocity_net, opt_fn, rec_weight,
                 n_inference_steps=50, val_n_inference_steps=None,
                 sigma_prior=1.0, ema_decay=0.999,
                 pre_metric_fn=None, test_metrics=None,
                 norm_stats=None, persist_rw=False):
        super().__init__()
        self.velocity_net = velocity_net
        self.n_inference_steps = n_inference_steps
        self.val_n_inference_steps = val_n_inference_steps or n_inference_steps
        self.sigma_prior = sigma_prior
        self._opt_fn = opt_fn
        self.pre_metric_fn = pre_metric_fn
        self.norm_stats = norm_stats
        self.metrics = test_metrics or {}

        if persist_rw:
            self.register_buffer('rec_weight', torch.from_numpy(rec_weight).float())
        else:
            self.rec_weight = torch.from_numpy(rec_weight).float()

        # EMA shadow copy
        self._ema_shadow = copy.deepcopy(self.velocity_net)
        self._ema_shadow.requires_grad_(False)
        self._ema_decay = ema_decay
        self._ema_initialised = False

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
            for ep, p in zip(self._ema_shadow.parameters(), self.velocity_net.parameters()):
                ep.data.mul_(self._ema_decay).add_(p.data, alpha=1.0 - self._ema_decay)

    def _ema_net(self):
        device = next(self.velocity_net.parameters()).device
        return self._ema_shadow.to(device)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _mask_future(self, batch):
        """Zero out the future half of the input (causal forecasting)."""
        new_input = batch.input.clone()
        T = new_input.size(1)
        new_input[:, T // 2:, :, :] = np.nan
        return batch._replace(input=new_input)

    def _get_source(self, batch):
        """
        Build source distribution from masked SLA observations.
        Observed pixels: obs + σ*ε  (stochastic → ensemble diversity)
        Missing/future:  0           (climatological mean)
        """
        obs = batch.input                              # [B, 29, H, W], NaN where missing
        obs_mask = torch.isfinite(obs).float()
        noise = torch.randn_like(obs) * self.sigma_prior * obs_mask
        return torch.nan_to_num(obs, nan=0.0) + noise  # [B, 29, H, W]

    def _get_condition(self, batch):
        """Static condition: masked SLA with NaN→0. Tells FM where obs are."""
        return torch.nan_to_num(batch.input, nan=0.0)  # [B, 29, H, W]

    def _get_target(self, batch):
        """Complete GLORYS12 SLA field — ground truth x_1."""
        return batch.tgt                               # [B, 29, H, W]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_1 = self._get_target(batch)
        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)

        x_source = self._get_source(batch)
        condition = self._get_condition(batch)

        # Logit-normal timestep (GloFM): z~N(0,1), tau=sigmoid(z)
        tau = torch.sigmoid(torch.randn(x_1.size(0), device=x_1.device))
        tau_e = tau[:, None, None, None]

        # OT path: x_source → x_1
        x_tau = (1.0 - tau_e) * x_source + tau_e * x_1

        # Velocity target: direction from obs-source to complete field
        v_target = x_1 - x_source
        v_pred = self.velocity_net(x_tau, tau, condition)

        # Train only on missing/future pixels — observed pixels have trivial
        # near-zero velocity (obs ≈ x_1 in OSSE) and dominate the loss otherwise
        obs_mask = torch.isfinite(batch.input).float()   # 1 where obs available
        missing_mask = valid.float() * (1.0 - obs_mask)  # missing but valid in tgt
        loss = (missing_mask * (v_pred - v_target) ** 2).sum() / missing_mask.sum().clamp(min=1)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_1 = self._get_target(batch)
        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)

        x_source = self._get_source(batch)
        condition = self._get_condition(batch)

        x_sample = self._sample(condition, x_source,
                                n_steps=self.val_n_inference_steps,
                                use_ema=True)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_mse', loss, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Sampling (Heun + non-uniform schedule)
    # ------------------------------------------------------------------

    def _sample(self, condition, x_source, n_steps=None, use_ema=False):
        """
        Integrate ODE from x_source to generated complete SLA field.

        Args:
            condition: [B, 29, H, W] — static masked SLA (NaN→0)
            x_source:  [B, 29, H, W] — noisy obs as starting point
            n_steps:   number of Heun steps
            use_ema:   use EMA weights if available
        """
        n_steps = n_steps or self.n_inference_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        ts = _nonuniform_schedule(n_steps, device=condition.device)
        x = x_source.clone()

        with torch.no_grad():
            for i in range(n_steps):
                t_curr, t_next = ts[i], ts[i + 1]
                dt = t_next - t_curr

                tau_curr = t_curr.expand(x.size(0))
                k1 = net(x, tau_curr, condition)

                x_pred = x + dt * k1
                tau_next = t_next.expand(x.size(0))
                k2 = net(x_pred, tau_next, condition)

                x = x + dt * (k1 + k2) / 2.0

        net.train()
        return x

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_source = self._get_source(batch)
        condition = self._get_condition(batch)

        x_refined = self._sample(condition, x_source, use_ema=True)

        if batch_idx == 0:
            self.test_data = []

        dm = self.trainer.datamodule
        if self.norm_stats is not None:
            m, s = self.norm_stats
        else:
            m, s = 0, 1

        out = x_refined.detach().cpu().unsqueeze(1) * s + m
        self.test_data.append(out)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self._opt_fn(self)
