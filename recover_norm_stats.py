"""
One-off script to recover the exact SST/SLA/Wind norm_stats computed by the
SST+SLA+Wind OpenAI UNet OSSE training run (fixed_sst_norm), for hardcoding
into the real-data OSE inference config. Instantiates the actual training
datamodule (same files, same domain/time windows), so the values are
bit-for-bit identical to what training used - deterministic, no need to dig
through old SLURM logs.

Usage (same resources as training, not on a login node):
    srun python recover_norm_stats.py
"""
import hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load(
    'config/xp/sst_training_anomalyClimato_MSE_2016_2019_SLA_WIND_INOUT_SLA_and_SST_OUTPUTS_OpenAIUNet_OSSE.yaml'
)
dm = hydra.utils.instantiate(cfg.datamodule)

print('SST norm stats', dm.norm_stats())
print('SLA norm stats', dm.sla_norm_stats())
print('Wind norm stats', dm.wind_norm_stats())
