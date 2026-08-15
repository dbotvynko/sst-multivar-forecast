"""
Unconditional UNet variant for GloFM training.

Identical architecture to FlowMatchingVelocityUNetAttn but takes x_t only —
no condition concatenation. Used when conditioning happens at inference
via MMPS (Moment-Matching Posterior Sampling) rather than input concat.
"""
from src.models_flow_matching_attn import FlowMatchingVelocityUNetAttn


class FlowMatchingVelocityUNetUnconditional(FlowMatchingVelocityUNetAttn):
    """
    Unconditional velocity UNet (GloFM style).

    Inherits full architecture from FlowMatchingVelocityUNetAttn (UNet + self-attention
    at coarsest resolution) but removes the condition concatenation.
    At inference, conditioning is handled externally via MMPS gradient correction.

    Args:
        n_output_channels: number of output/state channels (58 = 29 SST + 29 SLA)
        bilinear, time_dim, dropout, base_channels, attn_heads: same as parent
    """
    def __init__(self, n_output_channels=58, bilinear=True, time_dim=256,
                 dropout=0.0, base_channels=64, attn_heads=4):
        # Force n_cond_channels=0 → inc takes n_output_channels channels only
        super().__init__(
            n_output_channels=n_output_channels,
            n_cond_channels=0,
            bilinear=bilinear,
            time_dim=time_dim,
            dropout=dropout,
            base_channels=base_channels,
            attn_heads=attn_heads,
        )

    def forward(self, x_t, t, condition=None):
        """
        Args:
            x_t:       [B, n_output_channels, H, W] noisy state
            t:         [B] values in [0, 1]
            condition: ignored (accepted for API compatibility)
        Returns:
            velocity: [B, n_output_channels, H, W]
        """
        t_emb = self.time_embed(t)
        # No concatenation — just x_t directly into the UNet
        x1 = self.inc(x_t, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)
        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        return self.outc(x)
