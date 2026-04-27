"""Task sampler.

Two layers:
  1. ``StageSpec`` — frozen, deserialized from a stage YAML. Defines
     the global per-stage knobs: per-variate window range, context
     fraction, V distribution, target-subset distribution, regime
     distribution.
  2. ``SourceTaskOverride`` — per-source priors that *override* StageSpec
     keys, so synthetic and LOTSA can have different irregularity
     priors (per the plan §10).

The factory ``make_transform(stage, source_name, overrides)`` returns a
configured :class:`~imts_benchmark.synthetic_data.degradation.LOTSAToIrregular`
ready to be called on a raw source entry.

Per-example randomness (V, W, history, regime, target subset) lives
inside ``LOTSAToIrregular.__call__``; the sampler here just picks the
*distributions* it draws from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Optional

from ..synthetic_data.degradation import LOTSAToIrregular


@dataclass
class StageSpec:
    """Global per-stage sampling config."""

    # Per-variate window length (in source steps)
    per_var_min_length: int = 64
    per_var_max_length: int = 512

    # Variate count cap (model embeds up to max_dim=20)
    max_variates: int = 20
    # P[V=1], P[V in 2..8], P[V in 9..16], P[V in 17..max_variates]
    variate_count_dist: tuple = (0.10, 0.60, 0.25, 0.05)

    # History split
    context_fraction: tuple = (0.5, 0.95)
    tail_prediction_prob: float = 0.05
    tail_prediction_range: tuple = (0.92, 0.98)

    # Task type distribution
    # (univariate target / all-target / partial-target with aux)
    task_type_probs: tuple = (0.20, 0.50, 0.30)

    # Irregularity regime distribution (regular / sync / mixed / async)
    regime_dist: tuple = (0.15, 0.15, 0.45, 0.25)

    # Min observations per variate (target vs aux)
    min_ctx_obs_per_target: int = 3
    min_pred_obs_per_target: int = 1
    min_ctx_obs_per_aux: int = 1

    # Whether to keep the clean future as supervision (degraded LOTSA
    # should always be True; synthetic typically also True since it has
    # natural noise but no extra "missingness" already baked in).
    clean_target_mode: bool = True

    # Whether to allow timestamp jitter at all (per-source overridable).
    # See pretrain/README.md §3.3: jitter on real-data observation
    # times distorts the empirical distribution downstream models will
    # see, so we keep it on for synthetic and off for LOTSA.
    apply_jitter: bool = True


@dataclass
class SourceTaskOverride:
    """Per-source overrides on a StageSpec.

    Any field set here replaces the corresponding StageSpec field at
    transform-construction time. Designed to express
    "synthetic = harder async; LOTSA = milder, preserve regular".
    """

    per_var_min_length: Optional[int] = None
    per_var_max_length: Optional[int] = None
    max_variates: Optional[int] = None
    variate_count_dist: Optional[tuple] = None
    context_fraction: Optional[tuple] = None
    tail_prediction_prob: Optional[float] = None
    tail_prediction_range: Optional[tuple] = None
    task_type_probs: Optional[tuple] = None
    regime_dist: Optional[tuple] = None
    min_ctx_obs_per_target: Optional[int] = None
    min_pred_obs_per_target: Optional[int] = None
    min_ctx_obs_per_aux: Optional[int] = None
    clean_target_mode: Optional[bool] = None
    apply_jitter: Optional[bool] = None


def _stage_to_kwargs(stage: StageSpec) -> dict[str, Any]:
    return {
        "min_length": stage.per_var_min_length,
        "max_length": stage.per_var_max_length,
        "max_variates": stage.max_variates,
        "variate_count_dist": stage.variate_count_dist,
        "context_fraction": stage.context_fraction,
        "tail_prediction_prob": stage.tail_prediction_prob,
        "tail_prediction_range": stage.tail_prediction_range,
        "task_type_probs": stage.task_type_probs,
        "regime_dist": stage.regime_dist,
        "min_ctx_obs_per_target": stage.min_ctx_obs_per_target,
        "min_pred_obs_per_target": stage.min_pred_obs_per_target,
        "min_ctx_obs_per_aux": stage.min_ctx_obs_per_aux,
        "clean_target_mode": stage.clean_target_mode,
        "apply_jitter": stage.apply_jitter,
    }


def merge_override(stage: StageSpec, override: Optional[SourceTaskOverride]) -> StageSpec:
    if override is None:
        return stage
    diff = {k: v for k, v in asdict(override).items() if v is not None}
    return replace(stage, **diff)


def make_transform(
    stage: StageSpec,
    source_name: str,
    override: Optional[SourceTaskOverride] = None,
) -> LOTSAToIrregular:
    """Construct an LOTSAToIrregular tailored to (stage, source)."""
    eff = merge_override(stage, override)
    return LOTSAToIrregular(
        source_tag_prefix=source_name,
        **_stage_to_kwargs(eff),
    )


# ---------------------------------------------------------------------------
# Built-in source overrides reflecting plan §10
# ---------------------------------------------------------------------------

def default_overrides() -> dict[str, SourceTaskOverride]:
    """Per-source priors per the plan §10.

    Synthetic sources push harder async + more all-target; LOTSA stays
    milder so the model also sees regular + sync structure on real data.
    """
    return {
        "chronos2_synth": SourceTaskOverride(
            regime_dist=(0.05, 0.10, 0.45, 0.40),
            task_type_probs=(0.10, 0.65, 0.25),
            tail_prediction_prob=0.10,
        ),
        "kernelsynth": SourceTaskOverride(
            regime_dist=(0.05, 0.10, 0.40, 0.45),
            task_type_probs=(0.20, 0.55, 0.25),
            tail_prediction_prob=0.10,
        ),
        "lotsa_regular": SourceTaskOverride(
            regime_dist=(1.0, 0.0, 0.0, 0.0),
            task_type_probs=(0.20, 0.55, 0.25),
            clean_target_mode=False,
            apply_jitter=False,
        ),
        "lotsa_degraded": SourceTaskOverride(
            regime_dist=(0.0, 0.20, 0.55, 0.25),
            task_type_probs=(0.20, 0.50, 0.30),
            clean_target_mode=True,
            apply_jitter=False,
        ),
    }


def stage_from_dict(d: dict[str, Any]) -> StageSpec:
    """Tolerant constructor: ignore keys not in StageSpec."""
    valid = {f.name for f in fields(StageSpec)}
    return StageSpec(**{k: tuple(v) if isinstance(v, list) else v
                        for k, v in d.items() if k in valid})


def overrides_from_dict(d: dict[str, dict[str, Any]]) -> dict[str, SourceTaskOverride]:
    valid = {f.name for f in fields(SourceTaskOverride)}
    out = {}
    for src, vals in d.items():
        out[src] = SourceTaskOverride(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in vals.items() if k in valid
        })
    return out
