"""
Flow Matching for OSSE SLA — mirroring the ocean-FM-forecast implementation.

Source: https://github.com/dbotvynko/ocean-FM-forecast

Key design choices (matching ocean-FM-forecast):
    - Source x_0: pure uniform noise U[0,1] (not observation-based)
    - Condition y: nan_to_num(masked_input, 0) — concatenated to x_t channel-wise
    - Velocity target: v* = x_1 - x_0 (constant, linear interpolation)
    - Loss: weighted MSE on finite pixels with rec_weight > 0
    - ODE sampling: Euler integration (simple, T steps)
    - SDE sampling: ODE + time-varying noise epsilon(t) for ensemble diversity

Differences from LitFlowMatchingOSSE_SLA:
    - Source is pure noise (not obs-based) → more standard FM
    - Condition concatenated to x_t (not passed separately) → simpler UNet interface
    - Euler instead of Heun → faster inference, less accurate per step
    - SDE option for stochastic ensemble generation
    - Uniform timestep sampling (not logit-normal)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class LitFlowMatchingOSSE_SLA_OceanFM(pl.LightningModule):
    """
    OSSE SLA FM matching the ocean-FM-forecast design.

    velocity_net: UNet that takes [B, 2*n_channels, H, W] input (x_t || condition)
                  and outputs [B, n_channels, H, W] velocity field.
                  Must accept (x, tau) — tau is [B] int or float timestep.
    opt_fn:       optimizer factory (partial)
    rec_weight:   spatial weighting for loss [T, H, W]
    n_steps:      number of ODE/SDE integration steps (default 50)
    ema_decay:    EMA decay for inference weights (default 0.999)
    norm_stats:   optional (mean, std) for denormalization at test time
    persist_rw:   register rec_weight as buffer (default False)
    """

    def __init__(self, velocity_net, opt_fn, rec_weight,
                 n_steps=50, ema_decay=0.999,
                 t_power=1, n_ensemble=1,
                 pre_metric_fn=None, test_metrics=None,
                 norm_stats=None, persist_rw=False):
        super().__init__()
        self.velocity_net = velocity_net
        self.n_steps = n_steps
        self.t_power = t_power   # 1 = uniform, >1 = concentrated near tau=0
        self.n_ensemble = n_ensemble  # >1 = ensemble: saves mean + std
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

    def _get_condition(self, batch):
        """Masked SLA with NaN→0. Tells FM where observations are."""
        return torch.nan_to_num(batch.input, nan=0.0)  # [B, T, H, W]

    def _get_target(self, batch):
        """Complete GLORYS12 SLA — ground truth x_1."""
        return batch.tgt

    def _get_source(self, batch):
        """Pure uniform noise — same as ocean-FM-forecast."""
        return torch.rand_like(batch.tgt)

    # ------------------------------------------------------------------
    # Weighted MSE loss (matching ocean-FM-forecast)
    # ------------------------------------------------------------------

    def _weighted_mse(self, err, weight):
        """
        MSE weighted by rec_weight, masked to finite values and nonzero weight.
        Matches ocean-FM-forecast weighted_mse().
        """
        weight = weight.to(err.device)
        err_w = err * weight[None, ...]
        nonzero_weight = weight[None, ...] != 0.0
        finite_mask = err.isfinite() & nonzero_weight
        if finite_mask.sum() == 0:
            return err_w.sum() * 0.0
        return torch.nn.functional.mse_loss(
            err_w[finite_mask], torch.zeros_like(err_w[finite_mask])
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_1 = self._get_target(batch)
        x_0 = self._get_source(batch)
        condition = self._get_condition(batch)

        x_1 = torch.nan_to_num(x_1)

        # Timestep sampling: uniform (t_power=1) or power-law (t_power>1)
        # pow-law: tau = u^(1/p) concentrates training near tau=0 (noisy end)
        B = x_1.size(0)
        u = torch.rand(B, device=x_1.device)
        tau = u ** self.t_power                             # [B] in [0, 1); t_power>1 concentrates near tau=0
        tau_e = tau[:, None, None, None]

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_tau = (1.0 - tau_e) * x_0 + tau_e * x_1

        # Velocity target: v* = x_1 - x_0  (constant along path)
        v_target = x_1 - x_0

        # velocity_net concatenates x_tau and condition internally
        v_pred = self.velocity_net(x_tau, tau, condition)

        loss = self._weighted_mse(v_pred - v_target, self.rec_weight)
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

        condition = self._get_condition(batch)
        x_0 = self._get_source(batch)

        x_sample = self._sample_ode(condition, x_0, use_ema=True)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_mse', loss, prog_bar=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # ODE sampling (Euler — matching ocean-FM-forecast sample())
    # ------------------------------------------------------------------

    def _sample_ode(self, condition, x_0, n_steps=None, use_ema=False):
        """
        Euler integration of the learned velocity field.
        x_{t+dt} = x_t + v(x_t, t, cond) * dt
        """
        n_steps = n_steps or self.n_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        x = x_0.clone()
        dt = 1.0 / n_steps

        with torch.no_grad():
            for i in range(n_steps):
                tau = torch.full((x.size(0),), i / n_steps,
                                 device=x.device, dtype=torch.float32)
                v = net(x, tau, condition)
                x = x + v * dt

        net.train()
        return x

    # ------------------------------------------------------------------
    # SDE sampling (matching ocean-FM-forecast sample_sde())
    # ------------------------------------------------------------------

    def _sample_sde(self, condition, x_0, n_steps=None, use_ema=False,
                    epsilon_fn=None):
        """
        SDE integration for ensemble diversity.
        Adds time-varying noise epsilon(t) on top of the ODE drift.

        epsilon_fn: callable(t_int) -> float, noise schedule.
                    Default: small constant eps=0.01.
        """
        n_steps = n_steps or self.n_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        if epsilon_fn is None:
            epsilon_fn = lambda t: 0.01  # small constant noise

        x = x_0.clone()

        with torch.no_grad():
            for i in range(n_steps):
                t_norm = i / n_steps                        # t in [0, 1)
                tau = torch.full((x.size(0),), t_norm,
                                 device=x.device, dtype=torch.float32)
                v = net(x, tau, condition)

                eps = epsilon_fn(i)
                Wt = torch.rand_like(x) / n_steps           # uniform Wiener increment

                denom = t_norm ** 2 - t_norm - 1
                if abs(denom) < 1e-6 or eps == 0:
                    # Degenerate case: fall back to pure Euler step
                    x = x + v / n_steps
                else:
                    x = (x + x * t_norm / n_steps
                         + (v - t_norm * x) * (denom + eps) / denom / n_steps
                         + np.sqrt(2 * eps) * Wt)

        net.train()
        return x

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)
        condition = self._get_condition(batch)

        if batch_idx == 0:
            self.test_data = []
            if self.n_ensemble > 1:
                self.test_data_std = []

        if self.norm_stats is not None:
            m, s = self.norm_stats
        else:
            m, s = 0, 1

        if self.n_ensemble > 1:
            members = []
            for _ in range(self.n_ensemble):
                x_0 = self._get_source(batch)
                x_member = self._sample_ode(condition, x_0, use_ema=True)
                members.append(x_member)
            members = torch.stack(members, dim=0)          # [N, B, T, H, W]
            x_mean = members.mean(dim=0)
            x_std  = members.std(dim=0)
            self.test_data.append(x_mean.detach().cpu().unsqueeze(1) * s + m)
            self.test_data_std.append(x_std.detach().cpu().unsqueeze(1) * s)
        else:
            x_0 = self._get_source(batch)
            x_refined = self._sample_ode(condition, x_0, use_ema=True)
            self.test_data.append(x_refined.detach().cpu().unsqueeze(1) * s + m)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self._opt_fn(self)
