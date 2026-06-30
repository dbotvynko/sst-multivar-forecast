"""
Diagonal Hessian-based posterior uncertainty for 4D-Var.

In variational data assimilation, the posterior covariance is:
    P_a = (H^T R^{-1} H + B^{-1})^{-1}

where J(x) = obs_cost + prior_cost is the 4D-Var cost function.
The full Hessian is intractable at field resolution, so we estimate
the diagonal using a Hutchinson stochastic estimator:

    diag(H_J) ≈ (1/N) sum_i  v_i ⊙ (H_J v_i)

where v_i ~ Rademacher(±1) and H_J v_i is a Hessian-vector product
computed cheaply via two backpropagation passes (the "pearlmutter trick").

Posterior variance is then approximated as  σ²(x) ≈ 1 / diag(H_J(x*)),
where x* is the 4D-Var solution.

No retraining required — uses autograd on the frozen model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import NamedTuple


class HessianUQOutput(NamedTuple):
    state: torch.Tensor         # MAP solution  (B, T, H, W)
    variance: torch.Tensor      # posterior variance  (B, T, H, W)
    std: torch.Tensor           # posterior std  (B, T, H, W)


def _var_cost(state: torch.Tensor, batch, prior_cost: nn.Module, obs_cost: nn.Module) -> torch.Tensor:
    return prior_cost(state) + obs_cost(state, batch)


def _hutchinson_diag_hessian(
    state: torch.Tensor,
    batch,
    prior_cost: nn.Module,
    obs_cost: nn.Module,
    n_probes: int,
) -> torch.Tensor:
    """
    Estimate diag(H_J) at ``state`` using Hutchinson's estimator.
    Returns a tensor with the same shape as ``state``.
    """
    diag_est = torch.zeros_like(state)

    for _ in range(n_probes):
        v = torch.randint_like(state, low=0, high=2).float() * 2 - 1  # Rademacher

        # First-order gradient (with graph retained for second pass)
        state_req = state.detach().requires_grad_(True)
        J = _var_cost(state_req, batch, prior_cost, obs_cost)
        grad = torch.autograd.grad(J, state_req, create_graph=True)[0]

        # Hessian-vector product: ∇(g · v)
        gv = (grad * v).sum()
        hvp = torch.autograd.grad(gv, state_req, retain_graph=False)[0]

        diag_est = diag_est + v * hvp.detach()

    return diag_est / n_probes


class DiagHessianUQ(nn.Module):
    """
    Posterior uncertainty from the diagonal of the 4D-Var cost Hessian.

    Args:
        solver: GradSolver (or subclass) — used to obtain x* first.
        prior_cost: BilinAEPriorCost (or compatible module).
        obs_cost: BaseObsCost (or compatible module).
        n_probes: number of Hutchinson probe vectors.
        eps: small constant added to diagonal before inversion (numerical stability).
    """

    def __init__(
        self,
        solver: nn.Module,
        prior_cost: nn.Module,
        obs_cost: nn.Module,
        n_probes: int = 10,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.solver = solver
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.n_probes = n_probes
        self.eps = eps

    def forward(self, batch) -> HessianUQOutput:
        # Step 1 — obtain the MAP solution x* from the 4D-Var solver
        with torch.set_grad_enabled(True):
            state_map = self.solver(batch).detach()

        # Step 2 — estimate diag(H_J) at x*
        diag_h = _hutchinson_diag_hessian(
            state_map, batch, self.prior_cost, self.obs_cost, self.n_probes
        )

        # Posterior variance ≈ 1 / max(diag_h, eps)  (element-wise)
        variance = 1.0 / (diag_h.abs().clamp(min=self.eps))

        return HessianUQOutput(
            state=state_map,
            variance=variance,
            std=variance.sqrt(),
        )
