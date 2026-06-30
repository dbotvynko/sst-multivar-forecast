"""
Split Conformal Prediction for ocean forecasts.

Conformal prediction provides distribution-free, finite-sample coverage
guarantees:  P(y ∈ Ĉ(x)) ≥ 1 − α  for any user-chosen miscoverage α.

Protocol (split / inductive CP):
  1. Calibration pass  — collect nonconformity scores on a held-out set.
  2. Quantile extraction — find the (1−α)(1+1/n) quantile of scores.
  3. Prediction          — inflate each test interval by that quantile.

Two nonconformity score flavours are supported:
  • "absolute":    s_i = |ŷ_i − y_i|               (symmetric intervals)
  • "normalized":  s_i = |ŷ_i − y_i| / σ̂_i        (heteroscedastic, requires
                                                      a companion uncertainty σ̂)

References:
  Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification", 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Literal, NamedTuple


ScoreType = Literal["absolute", "normalized"]


class ConformalInterval(NamedTuple):
    lower: torch.Tensor     # (B, T, H, W)
    upper: torch.Tensor     # (B, T, H, W)
    q_hat: float            # scalar quantile threshold


class ConformalCalibrator:
    """
    Offline calibration + inference-time interval construction.

    Usage::

        cal = ConformalCalibrator(alpha=0.1, score_type="absolute")

        # --- calibration (once, on a held-out split) ---
        for batch in cal_dataloader:
            pred = model(batch)
            cal.update(pred, batch.tgt)
        cal.fit()

        # --- inference ---
        pred = model(test_batch)
        interval = cal.predict_interval(pred)   # guaranteed 90% coverage

    For normalized CP, pass ``sigma`` (same shape as ``pred``) to both
    ``update`` and ``predict_interval``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_type: ScoreType = "absolute",
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.score_type = score_type
        self._scores: list[torch.Tensor] = []
        self.q_hat: float | None = None

    def _nonconformity(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = (pred - target).abs()
        valid = target.isfinite()
        if self.score_type == "normalized":
            if sigma is None:
                raise ValueError("sigma required for normalized conformal scores")
            residual = residual / sigma.clamp(min=1e-8)
        return residual[valid].flatten()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> None:
        """Accumulate nonconformity scores from one calibration batch."""
        with torch.no_grad():
            scores = self._nonconformity(pred.detach(), target.detach(), sigma)
        self._scores.append(scores.cpu())

    def fit(self) -> float:
        """
        Compute the conformal quantile from accumulated calibration scores.
        Returns q_hat.
        """
        if not self._scores:
            raise RuntimeError("No calibration scores — call update() first.")
        all_scores = torch.cat(self._scores)
        n = len(all_scores)
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self.q_hat = float(torch.quantile(all_scores, level).item())
        return self.q_hat

    def predict_interval(
        self,
        pred: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> ConformalInterval:
        """
        Build conformal prediction intervals around ``pred``.

        For absolute CP:    [pred − q̂, pred + q̂]
        For normalized CP:  [pred − q̂·σ, pred + q̂·σ]
        """
        if self.q_hat is None:
            raise RuntimeError("Call fit() before predict_interval().")

        if self.score_type == "normalized":
            if sigma is None:
                raise ValueError("sigma required for normalized conformal intervals")
            half_width = self.q_hat * sigma.clamp(min=1e-8)
        else:
            half_width = torch.full_like(pred, self.q_hat)

        return ConformalInterval(
            lower=pred - half_width,
            upper=pred + half_width,
            q_hat=self.q_hat,
        )

    def reset(self) -> None:
        self._scores = []
        self.q_hat = None
