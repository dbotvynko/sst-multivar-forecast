"""
Lightning module wrapper that adds inference-time uncertainty quantification
to any existing Lit4dVarNet checkpoint.

Three UQ modes are supported (can be combined):
  • "ope"       — Observation Perturbation Ensemble
  • "mcd"       — Monte Carlo Dropout
  • "hessian"   — Diagonal Hessian posterior variance
  • "conformal" — Conformal prediction intervals (requires prior calibration)

The module is read-only with respect to the checkpoint: no training, no
weight modification.  It replaces test_step / on_test_epoch_end to output
additional uncertainty fields alongside the standard reconstruction.

Output netCDF variables:
  • out_mean    — point estimate (same as standard 'out')
  • out_std_ope — OPE ensemble spread
  • out_std_mcd — MC Dropout spread
  • out_std_hes — Hessian-based posterior std
  • out_lo_cp   — Conformal lower bound
  • out_hi_cp   — Conformal upper bound
"""

from __future__ import annotations

import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Literal

from .obs_ensemble import ObsEnsembleUQ
from .mc_dropout import MCDropoutUQ
from .hessian_uq import DiagHessianUQ
from .conformal import ConformalCalibrator

UQMode = Literal["ope", "mcd", "hessian", "conformal"]


class Lit4dVarNetUQ(pl.LightningModule):
    """
    Wraps a pre-trained Lit4dVarNet (or subclass) for uncertainty-aware
    inference.  The wrapped model is frozen; only test-time paths are
    active.

    Args:
        lit_model:        A loaded Lit4dVarNet instance (from checkpoint).
        uq_modes:         List of active UQ methods.
        ope_members:      Ensemble size for OPE.
        ope_noise_std:    Obs noise std (normalised units) for OPE.
        mcd_samples:      Number of MC Dropout samples.
        hessian_probes:   Hutchinson probe count for diagonal Hessian.
        conformal_alpha:  Miscoverage level for conformal intervals.
        conformal_score:  "absolute" or "normalized".
        save_members:     If True, store all ensemble members in test output.
    """

    def __init__(
        self,
        lit_model: pl.LightningModule,
        uq_modes: list[UQMode] | None = None,
        ope_members: int = 20,
        ope_noise_std: float = 0.05,
        mcd_samples: int = 30,
        hessian_probes: int = 10,
        conformal_alpha: float = 0.1,
        conformal_score: str = "absolute",
        save_members: bool = False,
    ):
        super().__init__()
        self.lit_model = lit_model
        self.uq_modes = set(uq_modes or ["ope", "mcd"])
        self.save_members = save_members

        # Freeze the wrapped model
        for p in self.lit_model.parameters():
            p.requires_grad_(False)
        self.lit_model.eval()

        solver = self.lit_model.solver

        if "ope" in self.uq_modes:
            self.ope = ObsEnsembleUQ(
                solver=solver,
                n_members=ope_members,
                obs_noise_std=ope_noise_std,
            )

        if "mcd" in self.uq_modes:
            self.mcd = MCDropoutUQ(solver=solver, n_samples=mcd_samples)

        if "hessian" in self.uq_modes:
            self.hessian_uq = DiagHessianUQ(
                solver=solver,
                prior_cost=solver.prior_cost,
                obs_cost=solver.obs_cost,
                n_probes=hessian_probes,
            )

        if "conformal" in self.uq_modes:
            self.conformal = ConformalCalibrator(
                alpha=conformal_alpha,
                score_type=conformal_score,
            )

        self.test_data: list = []

    @property
    def norm_stats(self):
        return self.lit_model.norm_stats

    @property
    def rec_weight(self):
        return self.lit_model.rec_weight

    # ------------------------------------------------------------------
    # Calibration pass (conformal only)
    # ------------------------------------------------------------------

    def calibrate_conformal(self, cal_dataloader) -> float:
        """
        Run a calibration forward pass to fit conformal quantiles.
        Call this once on a validation / held-out dataloader *before* testing.

        Returns:
            q_hat: the fitted conformal quantile.
        """
        if "conformal" not in self.uq_modes:
            raise RuntimeError("conformal not in uq_modes")

        self.lit_model.eval()
        device = next(self.lit_model.parameters()).device

        for batch in cal_dataloader:
            batch = batch._replace(**{
                f: getattr(batch, f).to(device)
                for f in batch._fields
                if torch.is_tensor(getattr(batch, f))
            })
            with torch.no_grad():
                pred = self.lit_model.solver(batch)
            sigma = None
            if "ope" in self.uq_modes:
                ope_out = self.ope(batch)
                sigma = ope_out.std
            self.conformal.update(pred, batch.tgt, sigma=sigma)

        q_hat = self.conformal.fit()
        return q_hat

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        m, s = self.norm_stats

        # Deterministic MAP solution (baseline)
        with torch.no_grad():
            out_map = self.lit_model.solver(batch)

        result = {
            "inp": batch.input.cpu() * s + m,
            "tgt": batch.tgt.cpu() * s + m,
            "out_mean": out_map.detach().cpu() * s + m,
        }

        # OPE
        if "ope" in self.uq_modes:
            ope_out = self.ope(batch)
            result["out_std_ope"] = ope_out.std.cpu() * s
            if self.save_members:
                result["out_members_ope"] = ope_out.members.cpu() * s + m

        # MC Dropout
        if "mcd" in self.uq_modes:
            mcd_out = self.mcd(batch)
            result["out_std_mcd"] = mcd_out.std.cpu() * s
            if self.save_members:
                result["out_members_mcd"] = mcd_out.members.cpu() * s + m

        # Hessian
        if "hessian" in self.uq_modes:
            hes_out = self.hessian_uq(batch)
            result["out_std_hes"] = hes_out.std.cpu() * s

        # Conformal
        if "conformal" in self.uq_modes:
            if self.conformal.q_hat is None:
                raise RuntimeError(
                    "Run calibrate_conformal() before calling test_step()."
                )
            sigma = result.get("out_std_ope")
            if sigma is not None:
                sigma = sigma.to(out_map.device)
            interval = self.conformal.predict_interval(out_map, sigma=sigma)
            result["out_lo_cp"] = interval.lower.cpu() * s + m
            result["out_hi_cp"] = interval.upper.cpu() * s + m

        # Stack into (B, n_fields, T, H, W) tensor for reconstruct()
        field_keys = sorted(k for k in result if k not in ("out_members_ope", "out_members_mcd"))
        self._field_keys = field_keys
        stacked = torch.stack([result[k].squeeze(dim=-1) for k in field_keys], dim=1)
        self.test_data.append(stacked)

    @property
    def test_quantities(self):
        return getattr(self, "_field_keys", ["inp", "tgt", "out_mean"])

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data,
            self.rec_weight.cpu().numpy(),
        )
        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim="v0")

        if self.logger:
            out_path = Path(self.logger.log_dir) / "test_data_uq.nc"
            self.test_data.to_netcdf(out_path)
            print(f"UQ output saved to {out_path}")

    # Training stubs — UQ module is inference-only
    def training_step(self, *args, **kwargs):
        raise NotImplementedError("Lit4dVarNetUQ is inference-only.")

    def configure_optimizers(self):
        raise NotImplementedError("Lit4dVarNetUQ is inference-only.")
