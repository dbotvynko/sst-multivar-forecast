"""
Entrypoint for uncertainty-quantification inference.

Called by the Hydra config via:
  _target_: contrib.uncertainty.run_uq_inference
"""

from __future__ import annotations

import torch
import pytorch_lightning as pl


def run_uq_inference(
    trainer: pl.Trainer,
    lit_uq,
    dm: pl.LightningDataModule,
    ckpt: str | None = None,
    run_conformal_calibration: bool = False,
) -> None:
    """
    Load checkpoint weights into the wrapped base model, optionally run
    conformal calibration on the validation set, then run the test loop.

    Args:
        trainer:                    PyTorch Lightning Trainer.
        lit_uq:                     Lit4dVarNetUQ instance.
        dm:                         DataModule.
        ckpt:                       Path to a .ckpt file.  Weights are loaded
                                    into ``lit_uq.lit_model`` only.
        run_conformal_calibration:  If True and "conformal" is in uq_modes,
                                    run a forward pass on the val set to fit
                                    the conformal quantile before testing.
    """
    # Load checkpoint weights into the base model (strict=False allows
    # missing keys for rec_weight buffers marked persist_rw=False)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        state_dict = state.get("state_dict", state)
        # Strip the "lit_model." prefix if needed (depends on how the ckpt was saved)
        base_sd = {
            k.removeprefix("solver."): v
            for k, v in state_dict.items()
            if not k.startswith("rec_weight")
        }
        missing, unexpected = lit_uq.lit_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[UQ] Missing keys when loading ckpt: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[UQ] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    lit_uq.lit_model.eval()
    for p in lit_uq.lit_model.parameters():
        p.requires_grad_(False)

    # Setup datamodule
    dm.setup(stage="test")

    # Optional conformal calibration on the validation set
    if run_conformal_calibration and "conformal" in lit_uq.uq_modes:
        dm.setup(stage="fit")
        print("[UQ] Running conformal calibration on validation set …")
        q_hat = lit_uq.calibrate_conformal(dm.val_dataloader())
        print(f"[UQ] Conformal quantile q̂ = {q_hat:.4f}")

    # Test
    trainer.test(lit_uq, datamodule=dm)
