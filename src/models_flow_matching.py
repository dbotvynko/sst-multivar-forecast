"""
Flow Matching velocity UNet for SST+SLA forecast refinement.

Time-conditioned UNet that predicts a velocity field v(x_t, t, condition)
to transform the deterministic forecast into the ground truth distribution.

Input:  x_t (58 ch) + condition (58 ch: past SST + past SLA) = 116 channels
Output: velocity (58 ch: 29 SST + 29 SLA)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.parts import Down, Up, OutConv, ResBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeConditionedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, mid_channels=None,
                 kernel_size=3, sf=1):
        super().__init__()
        self._scaling_factor = sf
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.projection_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        # inject time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        if hasattr(self, 'projection_conv'):
            x = self.projection_conv(x)
        return F.relu(h * self._scaling_factor + x)


class DownTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, sf=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = TimeConditionedBlock(in_channels, out_channels, time_dim, sf=sf)

    def forward(self, x, t_emb):
        x = self.pool(x)
        return self.block(x, t_emb)


class UpTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, bilinear=True, sf=1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.block = TimeConditionedBlock(in_channels, out_channels, time_dim,
                                              mid_channels=in_channels // 2, sf=sf)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.block = TimeConditionedBlock(in_channels, out_channels, time_dim, sf=sf)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x, t_emb)


class FlowMatchingVelocityUNet(nn.Module):
    """
    Time-conditioned UNet for flow matching velocity prediction.

    Args:
        n_output_channels: channels in x_t and output velocity (58 = 29 SST + 29 SLA)
        n_cond_channels: channels in the condition (58 = 29 SST input + 29 SLA input)
        bilinear: use bilinear upsampling
        time_dim: dimension of time embedding
    """
    def __init__(self, n_output_channels=58, n_cond_channels=58, bilinear=True, time_dim=256):
        super().__init__()
        self.n_output_channels = n_output_channels
        self.n_cond_channels = n_cond_channels
        factor = 2 if bilinear else 1

        sfs = 1 / torch.arange(1, 10).sqrt()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = n_output_channels + n_cond_channels
        self.inc = TimeConditionedBlock(in_ch, 64, time_dim, sf=1)
        self.down1 = DownTime(64, 128, time_dim, sf=sfs[1])
        self.down2 = DownTime(128, 256, time_dim, sf=sfs[2])
        self.down3 = DownTime(256, 512, time_dim, sf=sfs[3])
        self.down4 = DownTime(512, 1024 // factor, time_dim, sf=sfs[4])

        self.up1 = UpTime(1024, 512 // factor, time_dim, bilinear, sf=sfs[5])
        self.up2 = UpTime(512, 256 // factor, time_dim, bilinear, sf=sfs[6])
        self.up3 = UpTime(256, 128, time_dim, bilinear, sf=sfs[7])
        self.up4 = UpTime(128, 64, time_dim, bilinear, sf=sfs[8])
        self.outc = OutConv(64, n_output_channels)

    def forward(self, x_t, t, condition):
        """
        Args:
            x_t: [B, n_output_channels, H, W] noisy/interpolated state
            t: [B] time values in [0, 1]
            condition: [B, n_cond_channels, H, W] past observations
        Returns:
            velocity: [B, n_output_channels, H, W]
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
