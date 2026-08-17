"""
FM conditioned on past x_0 only (future half zeroed out).

Observation: during training, tgt (real observations) is available for all
29 time steps. Giving the FM x_0 for the future steps means conditioning on
a worse signal (deterministic forecast) when the real target is already known.

Fix: pass x_0 only for the past half of the temporal window (reconstruction),
zero out the future half. The FM must learn to forecast the future from the
distribution it learned, guided only by the past reconstruction.

condition: x_0[:, :14] for SST + x_0[:, :14] for SLA (past only, 58 ch total)
           future channels set to 0

Everything else inherited from LitFlowMatchingX0Cond_SST_SLA:
    - Logit-normal timestep
    - EMA (decay=0.999)
    - Heun ODE + non-uniform schedule
    - Frozen pretrained model
"""
import torch
from src.lit_flow_matching_x0cond import LitFlowMatchingX0Cond_SST_SLA


class LitFlowMatchingX0Past_SST_SLA(LitFlowMatchingX0Cond_SST_SLA):
    """
    FM conditioned on past-only x_0 (future steps zeroed out).

    Single override of _get_condition: zero out the future half of x_0
    so the FM cannot rely on the deterministic forecast for future time steps.
    Forces the FM to genuinely learn the forecast distribution.

    x_0 layout: [B, 58, H, W]
        channels  0-28: SST (29 time steps)
        channels 29-57: SLA (29 time steps)

    Past half:   channels  0-13 (SST) +  29-42 (SLA)  → kept
    Future half: channels 14-28 (SST) +  43-57 (SLA)  → zeroed
    """

    def _get_condition(self, batch, x_0):
        x_0_masked = torch.nan_to_num(x_0).clone()
        T = 29
        half = T // 2  # 14
        x_0_masked[:, half:T] = 0       # zero future SST channels (14-28)
        x_0_masked[:, T + half:] = 0    # zero future SLA channels (43-57)
        return x_0_masked
