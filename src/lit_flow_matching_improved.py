"""
Improved stochastic flow matching with three enhancements from Wetherell (2026):

1. EMA (Exponential Moving Average, decay=0.999): shadow copy of velocity_net
   weights maintained during training; used exclusively at inference/validation.
   Stabilises generation quality without changing training dynamics.

2. Self-attention at coarsest UNet resolution: handled by FlowMatchingVelocityUNetAttn
   in src/models_flow_matching_attn.py — this module just uses that network.

3. Heun's method + non-uniform timestep schedule: replaces simple Euler.
   Schedule: t_i = 1 - (1 - i/n)^2  (more steps near t=1 where signal appears).
   Heun = predictor-corrector (2nd-order), halves discretisation error at same step count.

Class: LitFlowMatchingImproved_SST_SLA
  - Inherits data plumbing from LitFlowMatchingStochastic_SST_SLA
  - Overrides _sample with Heun's method and non-uniform schedule
  - Adds EMA bookkeeping
"""
import copy
import torch
import pytorch_lightning as pl
from src.lit_flow_matching_stochastic import LitFlowMatchingStochastic_SST_SLA


def _nonuniform_schedule(n_steps, device):
    """
    Non-uniform timestep schedule: t_i = 1 - (1 - i/n)^2.
    Produces denser steps near t=1 (data end) where gradients are sharper.
    Returns tensor of shape (n_steps+1,) with t[0]=0, t[n]=1.
    """
    i = torch.arange(n_steps + 1, device=device, dtype=torch.float32)
    return 1.0 - (1.0 - i / n_steps) ** 2


class LitFlowMatchingImproved_SST_SLA(LitFlowMatchingStochastic_SST_SLA):
    """
    Stochastic FM with EMA weights + Heun's method + non-uniform timestep schedule.

    Additional constructor args (beyond LitFlowMatchingStochastic_SST_SLA):
        ema_decay: EMA weight, default 0.999 (Wetherell 2026)
    """
    def __init__(self, *args, ema_decay=0.999, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        # EMA shadow copy — lives on CPU, moved to device on first update
        self._ema_shadow = copy.deepcopy(self.velocity_net)
        self._ema_shadow.requires_grad_(False)
        self._ema_initialised = False

    # ------------------------------------------------------------------
    # EMA maintenance
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._update_ema()

    def _update_ema(self):
        decay = self.ema_decay
        if not self._ema_initialised:
            # First update: copy current weights exactly
            for ema_p, p in zip(self._ema_shadow.parameters(),
                                 self.velocity_net.parameters()):
                ema_p.data.copy_(p.data)
            self._ema_initialised = True
        else:
            for ema_p, p in zip(self._ema_shadow.parameters(),
                                 self.velocity_net.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def _ema_net(self):
        """Return EMA network on the correct device."""
        device = next(self.velocity_net.parameters()).device
        return self._ema_shadow.to(device)

    # ------------------------------------------------------------------
    # Override validation and sampling to use EMA weights
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        batch = self._mask_future(batch)
        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)

        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)
        x_0 = torch.nan_to_num(x_0)

        condition = self._get_condition(batch, x_0)
        x_sample = self._sample(condition, x_1,
                                n_steps=self.val_n_inference_steps,
                                use_ema=True)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def forward(self, batch):
        batch = self._mask_future(batch)
        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)
        reference = torch.zeros_like(x_0)
        return self._sample(condition, reference, use_ema=True)

    # ------------------------------------------------------------------
    # Heun's method + non-uniform schedule
    # ------------------------------------------------------------------

    def _sample(self, condition, reference, n_steps=None, use_ema=False):
        """
        Generate one ensemble member via Heun's 2nd-order ODE + non-uniform schedule.

        Direction: noise -> data (same as LitFlowMatchingStochastic_SST_SLA).
        Velocity: v(x, t, cond) = dx/dt ≈ x_1 - epsilon.

        Schedule: t_i = 1 - (1 - i/n)^2  → finer steps near t=1.
        Heun update (predictor-corrector):
            k1 = v(x_t,   t,      cond)
            k2 = v(x_t + dt*k1, t+dt, cond)
            x_{t+dt} = x_t + dt * (k1 + k2) / 2
        """
        n_steps = n_steps if n_steps is not None else self.n_inference_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        ts = _nonuniform_schedule(n_steps, device=condition.device)
        x = torch.randn_like(reference) * self.sigma_prior

        with torch.no_grad():
            for i in range(n_steps):
                t_curr = ts[i]
                t_next = ts[i + 1]
                dt = t_next - t_curr

                tau_curr = t_curr.expand(x.size(0))
                k1 = net(x, tau_curr, condition)

                x_pred = x + dt * k1
                tau_next = t_next.expand(x.size(0))
                k2 = net(x_pred, tau_next, condition)

                x = x + dt * (k1 + k2) / 2.0

        net.train()
        return x
