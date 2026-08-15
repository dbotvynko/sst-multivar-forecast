"""
GloFM-inspired conditional FM for end-to-end SST+SLA forecasting.

Adapts the key training trick from Garcia et al. (VISAPP 2026) to the
conditional forecasting setup: logit-normal timestep sampling.

What we keep from GloFM (paper Section 3.2):
    Logit-normal timestep: z ~ N(0,1), s = sigmoid(z)
    → training signal is denser around s=0.5 (hard interpolation regime)
    → downweights s≈0 (pure noise, trivial) and s≈1 (clean data, trivial)

What we do NOT adopt from GloFM (incompatible with forecasting):
    - Unconditional training: GloFM learns p(state) without conditioning.
      For forecasting we NEED conditioning on past observations + x_0 so
      the model can extrapolate from past to future.
    - MMPS at inference: useful for assimilation (estimating the present).
      For forecasting the deterministic x_0 already encodes the temporal
      dynamics; MMPS over past obs cannot produce future fields.

Everything else — EMA, Heun's ODE + non-uniform schedule, self-attention,
frozen pre-trained model conditioning — is inherited unchanged from
LitFlowMatchingImproved_SST_SLA.
"""
import torch
from src.lit_flow_matching_improved import LitFlowMatchingImproved_SST_SLA


class LitFlowMatchingGloFM_SST_SLA(LitFlowMatchingImproved_SST_SLA):
    """
    Conditional FM with GloFM's logit-normal timestep (forecasting edition).

    Identical to LitFlowMatchingImproved_SST_SLA except the training timestep
    distribution is changed from Uniform[0,1] to Logit-Normal(0,1):
        tau = sigmoid(z),  z ~ N(0,1)

    Inherited from LitFlowMatchingImproved_SST_SLA:
        - EMA (decay=0.999) — shadow weights used at inference
        - Heun's 2nd-order ODE + non-uniform schedule at inference
        - Self-attention at coarsest resolution (via FlowMatchingVelocityUNetAttn)

    Inherited from LitFlowMatchingStochastic_SST_SLA:
        - Frozen pre-trained deterministic model (x_0 as condition)
        - Condition = [past SST obs | past SLA obs | x_0]  (116 channels)
        - Stochastic direction: noise → data
        - OT velocity target: v = x_1 - epsilon

    Args: same as LitFlowMatchingImproved_SST_SLA — no new parameters.
    """

    def training_step(self, batch, batch_idx):
        batch = self._mask_future(batch)

        x_0 = self._get_deterministic_forecast(batch)
        x_1 = self._get_target(batch)

        valid = torch.isfinite(x_1)
        x_1 = torch.nan_to_num(x_1)
        x_0 = torch.nan_to_num(x_0)

        epsilon = torch.randn_like(x_1) * self.sigma_prior

        # ── GloFM logit-normal timestep ──────────────────────────────────
        # z ~ N(0,1),  tau = sigmoid(z)
        # vs. uniform: training signal concentrates around tau=0.5,
        # where the interpolant is most ambiguous and learning is hardest.
        tau = torch.sigmoid(torch.randn(x_1.size(0), device=x_1.device))
        tau_e = tau[:, None, None, None]

        # OT-path interpolation: x_tau = (1-tau)*epsilon + tau*x_1
        x_tau = (1.0 - tau_e) * epsilon + tau_e * x_1

        condition = self._get_condition(batch, x_0)

        # OT velocity target (constant for linear interpolant)
        v_target = x_1 - epsilon
        v_pred = self.velocity_net(x_tau, tau, condition)

        loss = (valid * (v_pred - v_target) ** 2).sum() / valid.sum().clamp(min=1)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
