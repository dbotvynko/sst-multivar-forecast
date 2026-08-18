"""
FM with sparse observations as the source distribution.

Instead of starting from pure Gaussian noise, the ODE starts from the
sparse observations (NaN → 0) and learns to transport them to the
complete ground truth fields.

Key idea:
    Standard FM:   noise ──────────────────────→ GT
    This variant:  sparse_obs + zeros ──────────→ GT

Source layout [B, 58, H, W]:
    channels  0-28: SST obs (NaN→0, future masked to 0 by _mask_future)
    channels 29-57: SLA obs (NaN→0, future masked to 0 by _mask_future)

Pixel-level interpretation:
    Observed pixels:  obs value  → close to GT, short path, easy correction
    Missing pixels:   0          → climatological mean, medium path
    Future pixels:    0          → no obs available, full generation

Why this is better than noise-source for ocean data:
    - Observed pixels already carry real signal → FM only needs small corrections
    - Missing/future pixels start from climatological mean (0 in anomaly space)
      which is a physically meaningful prior, unlike random noise
    - Velocity target v = x_1 - x_source is smaller where obs are dense
      → easier training signal in well-observed regions

Condition: x_0 from frozen pretrained UNet (58ch, x0cond variant)
    The deterministic forecast provides global context especially for
    future time steps where x_source = 0.

Inherits from LitFlowMatchingGloFM_SST_SLA:
    - Logit-normal timestep (GloFM trick)
    - EMA (decay=0.999)
    - Heun ODE + non-uniform schedule
    - Frozen pretrained UNet
"""
import torch
from src.lit_flow_matching_x0cond import LitFlowMatchingX0Cond_SST_SLA
from src.lit_flow_matching_improved import _nonuniform_schedule


class LitFlowMatchingObsSource_SST_SLA(LitFlowMatchingX0Cond_SST_SLA):
    """
    FM conditioned on x_0, with sparse observations as the source distribution.

    Single overrides vs LitFlowMatchingX0Cond_SST_SLA (which inherits GloFM):
        - _get_source: builds [B, 58, H, W] source from sparse obs (NaN→0)
        - training_step: uses obs-source instead of noise in OT path
        - validation_step: starts ODE from obs-source instead of noise
        - _sample: accepts x_source as starting point instead of drawing noise

    Everything else unchanged:
        - Condition: x_0 only (58ch, from frozen pretrained UNet)
        - Logit-normal timestep
        - EMA (decay=0.999)
        - Heun ODE + non-uniform schedule
    """

    def _get_source(self, batch):
        """
        Build obs-source with noise on observed pixels only.

        Observed pixels:  obs + σ*ε  (noisy obs → ensemble diversity)
        Missing pixels:   0           (climatological mean, no noise)
        Future pixels:    0           (already masked by _mask_future, no noise)
        """
        obs_sst = batch.input        # [B, 29, H, W], NaN where missing/future
        obs_sla = batch.input_sla    # [B, 29, H, W], NaN where missing/future

        mask_sst = torch.isfinite(obs_sst).float()
        mask_sla = torch.isfinite(obs_sla).float()

        noise_sst = torch.randn_like(obs_sst) * self.sigma_prior * mask_sst
        noise_sla = torch.randn_like(obs_sla) * self.sigma_prior * mask_sla

        sst = torch.nan_to_num(obs_sst, nan=0.0) + noise_sst
        sla = torch.nan_to_num(obs_sla, nan=0.0) + noise_sla
        return torch.cat([sst, sla], dim=1)                 # [B, 58, H, W]

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)

        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)

        # Source: sparse obs + zeros — no noise
        x_source = self._get_source(batch)

        # Logit-normal timestep (GloFM)
        tau = torch.sigmoid(torch.randn(x_1.size(0), device=x_1.device))
        tau_e = tau[:, None, None, None]

        # OT path: x_source → x_1
        x_tau = (1.0 - tau_e) * x_source + tau_e * x_1

        condition = self._get_condition(batch, x_0)   # x_0 only (58ch)

        # Velocity target: direction from obs-source to GT
        v_target = x_1 - x_source
        v_pred = self.velocity_net(x_tau, tau, condition)

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

        x_source = self._get_source(batch)
        condition = self._get_condition(batch, x_0)

        x_sample = self._sample(condition, x_source,
                                n_steps=self.val_n_inference_steps,
                                use_ema=True)

        loss = (valid * (x_sample - x_1) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def _sample(self, condition, x_source, n_steps=None, use_ema=False):
        """
        Integrate ODE from obs-source to generated field (Heun + non-uniform schedule).

        Args:
            condition: [B, 58, H, W] — x_0 from frozen pretrained UNet
            x_source:  [B, 58, H, W] — sparse obs with NaN→0 (starting point)
            n_steps:   number of ODE steps (default: self.n_inference_steps)
            use_ema:   use EMA weights if available
        """
        n_steps = n_steps if n_steps is not None else self.n_inference_steps
        net = self._ema_net() if use_ema and self._ema_initialised else self.velocity_net
        net.eval()

        ts = _nonuniform_schedule(n_steps, device=condition.device)
        x = x_source.clone()   # start from obs-source, not noise

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

    def test_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_0 = torch.nan_to_num(x_0)
        condition = self._get_condition(batch, x_0)
        x_source = self._get_source(batch)

        x_refined = self._sample(condition, x_source, use_ema=True)

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

        out_2var = x_refined.view(
            x_refined.size(0), 2, 29, x_refined.size(-2), x_refined.size(-1)
        ).detach().cpu()
        out_sst = out_2var[:, 0:1] * s_sst + m_sst
        out_sla = out_2var[:, 1:2] * s_sla + m_sla
        self.test_data_sst.append(out_sst)
        self.test_data_sla.append(out_sla)
