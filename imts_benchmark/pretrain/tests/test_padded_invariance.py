"""Test: padded-variate slots must not affect predictions on valid slots.

Builds a real batch via the smoke datamodule, then perturbs every value/
timestamp/deltat at slots where ``valid_variate_mask is False`` and re-runs
the model. Predictions on positions where ``pred_mask is True`` must be
bit-equal to the original.

If this test fails the model is leaking padded-variate information into
real predictions, which would make pretraining with ``max_dim=20`` and
variable per-sample V_active produce position-conditioned bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imts_benchmark.mamba_mv.multivariate_forecaster import (  # noqa: E402
    MultivariateMambaForecaster,
)
from imts_benchmark.pretrain.datamodule import (  # noqa: E402
    PretrainDataModule,
    PretrainDataModuleArgs,
)


def _get_one_batch_with_padding() -> dict:
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    args = PretrainDataModuleArgs(
        stage_cfg_path=str(cfg_dir / "stage_a_smoke.yaml"),
        sources_cfg_path=str(cfg_dir / "sources_smoke.yaml"),
        batch_size=8,
        num_workers=0,
        max_dim=20,
        seed=11,
        persistent_workers=False,
    )
    dm = PretrainDataModule(args)
    dm.setup()
    loader = dm.train_dataloader()
    for batch in loader:
        if (~batch["valid_variate_mask"]).any():
            return batch
    raise RuntimeError("could not find a batch with at least one padded variate")


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultivariateMambaForecaster(
        d_model=64, d_hidden=64, max_dim=20,
        n_perv_layer=2, n_fusion_blocks=2, n_heads_varattn=2,
        d_state=8, grid_K=32, t_max=1.0, loss_type="huber",
    ).to(device).eval()

    batch = _get_one_batch_with_padding()
    batch_d = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }

    valid_var = batch_d["valid_variate_mask"]
    pred_mask = batch_d["pred_mask"]

    print(f"valid_variate_mask coverage: {valid_var.float().mean().item():.3f}")
    print(f"pred_mask True positions:    {int(pred_mask.sum().item())}")
    print(f"padded slots overall:        {int((~valid_var).sum().item())}")

    with torch.no_grad():
        preds_clean = model(batch_d).detach().clone()

    pad_v = ~valid_var
    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_d.items()}
    perturbed["values"][pad_v] = torch.randn_like(perturbed["values"][pad_v]) * 100.0
    perturbed["timestamps"][pad_v] = torch.rand_like(perturbed["timestamps"][pad_v])
    perturbed["deltat"][pad_v] = torch.rand_like(perturbed["deltat"][pad_v])
    perturbed["valid_mask"][pad_v] = False

    with torch.no_grad():
        preds_perturbed = model(perturbed).detach().clone()

    diff = (preds_clean - preds_perturbed).abs()
    diff_at_targets = diff[pred_mask]
    diff_at_padded_v = diff[~valid_var.unsqueeze(-1).expand_as(diff)]

    print(f"max |Δpred| at pred_mask=True : {diff_at_targets.max().item():.3e}")
    print(f"max |Δpred| at padded-V slots : {diff_at_padded_v.max().item():.3e}")

    tol = 1e-3 if preds_clean.dtype == torch.bfloat16 else 1e-5
    if diff_at_targets.max().item() > tol:
        print(
            f"FAIL: padded-variate values leak into real predictions "
            f"({diff_at_targets.max().item():.3e} > {tol:.0e})"
        )
        sys.exit(1)
    print("PASS: predictions at pred_mask=True are invariant to padded-variate content")


if __name__ == "__main__":
    main()
