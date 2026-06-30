"""
Observation Perturbation Ensemble (OPE) for 4D-Var uncertainty quantification.

Physically mirrors Ensemble Data Assimilation: draw N perturbed observation
vectors from the observation error distribution, run the 4D-Var solver for
each member, and estimate posterior mean/std from the ensemble spread.

No retraining required — inference only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import NamedTuple


class EnsembleOutput(NamedTuple):
    mean: torch.Tensor          # (B, T, H, W)
    std: torch.Tensor           # (B, T, H, W)
    members: torch.Tensor       # (N, B, T, H, W)  — kept for downstream analysis


class ObsEnsembleUQ(nn.Module):
    """
    Wraps a 4D-Var solver to produce uncertainty estimates via observation
    perturbation.

    Args:
        solver: a GradSolver (or subclass) with a callable forward(batch).
        n_members: number of ensemble members.
        obs_noise_std: standard deviation of additive Gaussian noise applied
            to observed pixels (normalised space, same as model inputs).
        perturb_missing: if True, also add noise to NaN-filled positions
            (useful when the fill value carries meaningful information).
        seed: optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        solver: nn.Module,
        n_members: int = 20,
        obs_noise_std: float = 0.05,
        perturb_missing: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.solver = solver
        self.n_members = n_members
        self.obs_noise_std = obs_noise_std
        self.perturb_missing = perturb_missing
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    @torch.no_grad()
    def forward(self, batch) -> EnsembleOutput:
        """
        Run ensemble forward pass and return mean, std, and all members.

        ``batch`` is the standard namedtuple produced by the datamodule
        (fields: input, tgt, …).  Only ``batch.input`` is perturbed; all
        other fields are shared across members.
        """
        members = []
        obs_mask = batch.input.isfinite()   # True where satellite obs exist

        for _ in range(self.n_members):
            noise = torch.zeros_like(batch.input).normal_(
                mean=0.0, std=self.obs_noise_std, generator=self._rng
            )
            if not self.perturb_missing:
                noise = noise * obs_mask.float()

            perturbed_input = batch.input.nan_to_num() + noise
            # Restore NaN mask so the obs cost sees the same structure
            perturbed_input = perturbed_input.masked_fill(~obs_mask, float("nan"))

            perturbed_batch = batch._replace(input=perturbed_input)
            out = self.solver(perturbed_batch)          # (B, T, H, W)
            members.append(out.detach())

        members_t = torch.stack(members, dim=0)        # (N, B, T, H, W)
        mean = members_t.mean(dim=0)
        std = members_t.std(dim=0, unbiased=True)

        return EnsembleOutput(mean=mean, std=std, members=members_t)
