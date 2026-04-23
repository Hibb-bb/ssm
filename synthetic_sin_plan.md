# Synthetic Sin Dataset Plan

## Purpose

This folder is the working area for the new synthetic sinusoidal benchmark used to test whether our Mamba-based architecture is better than transformer-based baselines on irregular multivariate time series forecasting, with special focus on long-range missing gaps.

## Main Goal

Build a new synthetic dataset pipeline without overwriting previous code in:

`/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm`

The new data should support controlled evaluation of:

- sparse long-gap forecasting
- multivariate forecasting
- dependent vs independent cross-variate structure
- later evaluation with patch-based forecasting pipelines

## Current Decisions

### Dataset regimes

We will generate two main multivariate regimes:

1. `sparse_independent`
2. `sparse_dependent`

These are the minimum clean benchmark cases.

### Why these two cases

- `sparse_independent` is a negative control:
  other variables should provide little or no useful predictive information.
- `sparse_dependent` is a positive control:
  other variables should help recover information across long missing gaps.

This makes the experimental interpretation clearer:

- better only on dependent data suggests the model benefits from cross-variate reasoning
- better on both suggests the gain may come from handling sparse irregular dynamics more generally
- worse on independent data suggests the model may overuse irrelevant variables

### Core difficulty to preserve

The main benchmark property must be:

- long-range missing gaps

This should remain central in both regimes.

### Multivariate timing design

We want asynchronous per-variate observations, not just shared timestamps repeated across variables.

This is important because the current multivariate sinusoid code in the old folder still repeats the same timestamps for all variables within a sample, which is not the setting we want.

### Dependency claim

It is reasonable to say the architecture handles both dependent and independent situations only if both regimes are explicitly present in the benchmark.

That is why the dataset must include both:

- independent variables
- dependent variables

## Findings From Existing `ssm` Folder

Inspected context folder:

`/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm`

### Existing sinusoid generation

Relevant files found:

- `mamba_experiments/dataset_generation/generate_sinusoidal_data.py`
- `mamba_experiments/dataset_generation/generate_sinusoidal_raw.py`
- `mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py`
- `mamba_experiments/dataset_generation/convert_to_moirai.py`
- `mamba_experiments/dataset_generation/convert_tpatchgnn_data.py`

### How existing sin data has been built

#### Old NCDSSM sine dataset

File:

- `src/ncdssm/datasets/synthetic.py`

Characteristics:

- regular sampling
- univariate
- random point masking only
- no explicit long missing block
- no async multivariate structure

#### New univariate Mamba sinusoid generator

File:

- `mamba_experiments/dataset_generation/generate_sinusoidal_data.py`

Characteristics:

- univariate irregular timestamps
- irregularity controlled by fraction of identical gaps
- stores `target`, `timestamp`, `past_feat_dynamic_real`, `n_obs_per_var`, `history`
- useful storage format reference
- still not designed around explicit long-range gaps

#### Existing multivariate sinusoid generator

File:

- `mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py`

Characteristics:

- three variables
- two sinusoidal sources plus one convex mixture
- multivariate sparse flat storage
- timestamps are shared across variates within each sample
- not asynchronous per variate
- not a true long-gap benchmark

### Sparse multivariate format already used

Important file:

- `mamba_experiments/dataset_generation/convert_tpatchgnn_data.py`

This file is the main format reference.

It stores each sample as:

- `target`
- `timestamp`
- `past_feat_dynamic_real`
- `n_obs_per_var`
- `history`

with all observations concatenated across variables.

This format should be preserved so later evaluation stays compatible with existing irregular multivariate pipelines.

## Patching-Related Finding

Patching does exist in the broader experiment flow, but mainly on the MOIRAI / T-PatchGNN side, not in the standalone Mamba sinusoid forecaster.

Relevant file:

- `mamba_experiments/evaluation/eval_tpatchgnn.py`

Observed transform components include:

- `GetPatchSize`
- `Patchify`
- `AddTimeIndex`
- `AddVariateIndex`
- `FixedHorizonPrediction`

Interpretation:

- patching is relevant for forecasting evaluation
- context window and prediction window handling likely belong after dataset generation
- dataset generation should first produce correct sparse irregular multivariate samples
- patch-based slicing can be layered on top later

## Draft Paper Model Status

The full draft-paper model does not appear to already exist in the inspected `ssm` folder.

What exists:

- univariate irregular Mamba forecaster
- Mamba block with `delta_t` injection
- patch-aware evaluation / preprocessing for MOIRAI and T-PatchGNN

What does not appear implemented yet:

- per-variate irregular encoding followed by alignment to a shared internal grid
- null-state handling for unavailable variables
- staleness features
- variable-axis attention over aligned variates
- temporal Mamba after cross-variate fusion
- general multivariate query-time decoding from the draft

Conclusion:

- the dataset pipeline can be built now
- the draft model itself will be a new implementation later

## Proposed New Dataset Design

### Shared requirements for both regimes

Each sample should include:

- multiple variables
- asynchronous timestamps across variables
- long missing block or blocks
- irregular sampling before and after the long gap
- forecasting setup with clear history and prediction horizon

### `sparse_independent`

Each variable is generated from its own latent sinusoidal process, for example:

- different amplitude
- different frequency
- different phase
- independent noise
- no shared latent source

Goal:

- test whether the model avoids being hurt by irrelevant channels

### `sparse_dependent`

Variables should share useful latent structure, for example:

- shared latent sinusoidal source
- lagged / phase-shifted variants
- linear mixtures of shared sources
- small private noise terms

Goal:

- test whether the model can recover target behavior by using other variables across long gaps

### Long-gap mechanism

The missingness process should explicitly create long unobserved intervals, not only pointwise irregularity.

Possible mechanism:

- sample a full dense latent trajectory
- choose one or more long forbidden intervals
- drop observations inside those intervals for selected variables
- keep asynchronous sparse observations outside the forbidden intervals

This is more aligned with the paper goal than only controlling local `delta_t` randomness.

## Next Implementation Steps

### Step 1

Create a new isolated code path inside this `ssm_dk` folder for synthetic long-gap multivariate sin generation.

### Step 2

Implement a generator that outputs the same sparse multivariate flat format already used by the existing pipeline:

- `target`
- `timestamp`
- `past_feat_dynamic_real`
- `n_obs_per_var`
- `history`

### Step 3

Support two benchmark regimes:

- `sparse_independent`
- `sparse_dependent`

### Step 4

Make timestamps asynchronous per variate.

### Step 5

Add explicit long-gap controls, such as:

- gap start range
- gap length range
- number of gaps
- which variables receive the gap

### Step 6

Save train / val / test splits separately.

### Step 7

After dataset generation is stable, connect it to evaluation settings involving:

- context window
- prediction horizon
- patching

### Step 8

After the dataset is working, compare models:

- current Mamba baselines
- transformer / MOIRAI baselines
- possibly RopeMAE later if bandwidth allows

## Practical Working Rule

Do not overwrite old sinusoid code. Keep all new work in this `ssm_dk` folder unless there is a clear need to integrate with the existing training and evaluation pipeline later.
