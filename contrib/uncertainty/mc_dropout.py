"""
Monte Carlo Dropout uncertainty quantification.

The ConvLstmGradModel already contains a Dropout layer (dropout=0.1).
Enabling dropout at inference time and running N stochastic forward passes
gives a cheap approximation to a Bayesian posterior (Gal & Ghahramani 2016).

No retraining required — dropout is already part of the trained model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import NamedTuple


class MCDropoutOutput(NamedTuple):
    mean: torch.Tensor      # (B, T, H, W)
    std: torch.Tensor       # (B, T, H, W)
    members: torch.Tensor   # (N, B, T, H, W)


@contextmanager
def _dropout_eval_mode(model: nn.Module):
    """
    Keep the whole model in eval mode (no BN statistics update) but
    re-enable all Dropout layers so stochastic inference is active.
    """
    model.eval()
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for d in dropout_layers:
        d.train()
    try:
        yield
    finally:
        # restore: dropout layers back to eval
        for d in dropout_layers:
            d.eval()


class MCDropoutUQ(nn.Module):
    """
    Wraps a Lit4dVarNet solver for MC Dropout inference.

    Args:
        solver: GradSolver or any nn.Module containing Dropout layers.
        n_samples: number of stochastic forward passes.
    """

    def __init__(self, solver: nn.Module, n_samples: int = 30):
        super().__init__()
        self.solver = solver
        self.n_samples = n_samples

    @torch.no_grad()
    def forward(self, batch) -> MCDropoutOutput:
        members = []
        with _dropout_eval_mode(self.solver):
            for _ in range(self.n_samples):
                out = self.solver(batch)
                members.append(out.detach())

        members_t = torch.stack(members, dim=0)   # (N, B, T, H, W)
        return MCDropoutOutput(
            mean=members_t.mean(dim=0),
            std=members_t.std(dim=0, unbiased=True),
            members=members_t,
        )
