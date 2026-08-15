"""
FM conditioned on pretrained model output only (x_0-only condition).

Ronan's suggestion: since the pretrained UNet already maps sparse obs →
complete SST+SLA fields, feeding the sparse obs again to the FM is redundant.
x_0 already encodes everything the sparse obs contain.

Condition: x_0 only (58 ch = 29 SST + 29 SLA from frozen pretrained UNet)
vs. previous: [past SST obs | past SLA obs | x_0] (116 ch)

The FM learns: given the deterministic forecast x_0, what stochastic
corrections produce a more realistic ocean state?

Everything else inherited from LitFlowMatchingGloFM_SST_SLA:
    - Logit-normal timestep sampling
    - EMA (decay=0.999)
    - Heun's ODE + non-uniform schedule
    - Frozen pretrained deterministic model
"""
import torch
from src.lit_flow_matching_glofm import LitFlowMatchingGloFM_SST_SLA


class LitFlowMatchingX0Cond_SST_SLA(LitFlowMatchingGloFM_SST_SLA):
    """
    FM with x_0-only condition (pretrained forecast as sole conditioning signal).

    Single override: _get_condition returns x_0 directly instead of
    concatenating sparse observations alongside it.

    velocity_net input: x_t (58 ch) + x_0 (58 ch) = 116 ch total
    → set n_cond_channels: 58 in the config
    """

    def _get_condition(self, batch, x_0):
        return torch.nan_to_num(x_0)
