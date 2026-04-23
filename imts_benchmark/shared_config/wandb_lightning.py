"""Optional Weights & Biases logging for PyTorch Lightning benchmark trainers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping


def _args_to_config(args: Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def build_wandb_logger(
    args: Namespace,
    *,
    extra_config: Mapping[str, Any] | None = None,
) -> Any:
    """Return a ``WandbLogger`` when ``args.use_wandb`` is set, else ``False``.

    Passing ``False`` to ``pl.Trainer(logger=...)`` disables the default
    TensorBoard logger. Install wandb in the environment when using this.
    """
    if not getattr(args, "use_wandb", False):
        return False
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as e:
        raise RuntimeError(
            "imts_benchmark: --use_wandb requires pytorch_lightning with "
            "wandb support. Install with: pip install wandb"
        ) from e

    config = _args_to_config(args)
    if extra_config:
        config.update(dict(extra_config))

    kwargs: dict[str, Any] = {"log_model": False, "config": config}
    if getattr(args, "wandb_project", None):
        kwargs["project"] = args.wandb_project
    if getattr(args, "wandb_entity", None):
        kwargs["entity"] = args.wandb_entity
    if getattr(args, "wandb_run_name", None):
        kwargs["name"] = args.wandb_run_name

    try:
        return WandbLogger(**kwargs)
    except ImportError as e:
        raise RuntimeError(
            "imts_benchmark: --use_wandb requires the wandb package. "
            "Install with: pip install wandb"
        ) from e


def log_wandb_after_fit(
    logger: Any,
    wall_fit_sec: float,
    best_val_mse: float | None,
) -> None:
    """Log fit wall time and best validation MSE after ``trainer.fit``."""
    if logger is False:
        return
    from pytorch_lightning.loggers import WandbLogger

    if not isinstance(logger, WandbLogger):
        return
    payload: dict[str, Any] = {
        "fit/wall_sec": float(wall_fit_sec),
        "fit/wall_min": float(wall_fit_sec) / 60.0,
    }
    if best_val_mse is not None:
        payload["best/val_mse"] = float(best_val_mse)
    exp = logger.experiment
    if exp is not None:
        exp.log(payload)


def log_wandb_run_summary(
    logger: Any,
    metrics_row: Mapping[str, Any],
) -> None:
    """Log final CSV-style metrics under ``run/`` for sweep dashboards."""
    if logger is False:
        return
    from pytorch_lightning.loggers import WandbLogger

    if not isinstance(logger, WandbLogger):
        return
    exp = logger.experiment
    if exp is None:
        return
    row: dict[str, Any] = {}
    for k, v in metrics_row.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[f"run/{k}"] = v
    if row:
        exp.log(row)
