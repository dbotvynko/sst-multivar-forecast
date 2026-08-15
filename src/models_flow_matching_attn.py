"""
Flow Matching velocity UNet with self-attention at the coarsest resolution.

Adds attention blocks in the bottleneck (after down4 / before up1) to capture
long-range spatial dependencies — following Dhariwal & Nichol (ADM, 2021) and
Wetherell (2026) which applies self-attention at the 16×16 level.

New class: FlowMatchingVelocityUNetAttn
Everything else is unchanged from models_flow_matching.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models_flow_matching import (
    SinusoidalTimeEmbedding,
    TimeConditionedBlock,
    DownTime,
    UpTime,
    _norm,
)
from src.parts import OutConv


class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention over spatial feature maps.
    Reshapes (B, C, H, W) -> (B, H*W, C), runs attention, reshapes back.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = _norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class DownTimeAttn(nn.Module):
    """MaxPool + TimeConditionedBlock + optional self-attention."""
    def __init__(self, in_channels, out_channels, time_dim, sf=1, dropout=0.0,
                 with_attn=False, num_heads=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = TimeConditionedBlock(in_channels, out_channels, time_dim,
                                          sf=sf, dropout=dropout)
        self.attn = SelfAttention2d(out_channels, num_heads) if with_attn else None

    def forward(self, x, t_emb):
        x = self.pool(x)
        x = self.block(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class FlowMatchingVelocityUNetAttn(nn.Module):
    """
    Time-conditioned UNet with self-attention at the coarsest resolution.

    Identical to FlowMatchingVelocityUNet except down4 / bottleneck has a
    SelfAttention2d block appended, following Wetherell (2026) and ADM (2021).

    Args:
        n_output_channels: channels in x_t and output velocity (58 = 29 SST + 29 SLA)
        n_cond_channels:   channels in the condition (116 = 29+29 obs + 58 x_0)
        bilinear:          use bilinear upsampling
        time_dim:          dimension of time embedding
        dropout:           spatial dropout rate in conv blocks
        base_channels:     width multiplier (default 64)
        attn_heads:        heads for multi-head self-attention
    """
    def __init__(self, n_output_channels=58, n_cond_channels=116, bilinear=True,
                 time_dim=256, dropout=0.0, base_channels=64, attn_heads=4):
        super().__init__()
        self.n_output_channels = n_output_channels
        self.n_cond_channels = n_cond_channels
        factor = 2 if bilinear else 1
        b = base_channels

        sfs = 1 / torch.arange(1, 10).sqrt()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = n_output_channels + n_cond_channels
        self.inc   = TimeConditionedBlock(in_ch, b,          time_dim, sf=1,      dropout=dropout)
        self.down1 = DownTime(b,     b*2,          time_dim, sf=sfs[1], dropout=dropout)
        self.down2 = DownTime(b*2,   b*4,          time_dim, sf=sfs[2], dropout=dropout)
        self.down3 = DownTime(b*4,   b*8,          time_dim, sf=sfs[3], dropout=dropout)
        # Coarsest resolution — add self-attention here
        self.down4 = DownTimeAttn(b*8, b*16 // factor, time_dim, sf=sfs[4],
                                  dropout=dropout, with_attn=True, num_heads=attn_heads)

        self.up1 = UpTime(b*16,      b*8  // factor, time_dim, bilinear, sf=sfs[5], dropout=dropout)
        self.up2 = UpTime(b*8,       b*4  // factor, time_dim, bilinear, sf=sfs[6], dropout=dropout)
        self.up3 = UpTime(b*4,       b*2  // factor, time_dim, bilinear, sf=sfs[7], dropout=dropout)
        self.up4 = UpTime(b*2,       b,              time_dim, bilinear, sf=sfs[8], dropout=dropout)
        self.outc = OutConv(b, n_output_channels)

    def forward(self, x_t, t, condition):
        """
        Args:
            x_t:       [B, n_output_channels, H, W]
            t:         [B] values in [0, 1]
            condition: [B, n_cond_channels, H, W]
        Returns:
            velocity or x_1 prediction: [B, n_output_channels, H, W]
        """
        t_emb = self.time_embed(t)
        x_input = torch.cat([x_t, condition], dim=1)

        x1 = self.inc(x_input, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)

        return self.outc(x)
