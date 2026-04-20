# Setup — Mamba-MV vs RoMAE

Reproducing this branch requires:

1. **The Mamba conda env** (Python 3.12, PyTorch 2.5 + CUDA 12.4,
   `mamba-ssm` built from source for the CUDA selective-scan kernel).
   Use the existing Quest build script:

   ```bash
   # From the ssm repo root:
   sbatch mamba_experiments/environment/build_mamba_env.sh
   ```

   This produces `/projects/b1094/StarEmbed/pythonenvs/mamba` on Quest.
   On a different machine, replicate the env with `python=3.12` +
   `pytorch pytorch-cuda=12.4 -c pytorch -c nvidia` +
   `pip install causal-conv1d mamba-ssm --no-build-isolation --no-cache-dir`
   + `pip install pytorch-lightning datasets scipy einops`.

2. **RoMAE** — pinned to commit `480cfaf` of
   [github.com/Chromeilion/RoMAE](https://github.com/Chromeilion/RoMAE).
   Install into the same env:

   ```bash
   cd ssm_dk/mv_vs_romae/romae_forecaster
   git clone https://github.com/Chromeilion/RoMAE.git _romae_repo
   cd _romae_repo && git checkout 480cfaf && cd -
   /path/to/pythonenvs/mamba/bin/pip install -e ./_romae_repo --no-deps
   /path/to/pythonenvs/mamba/bin/pip install \
       pydantic-settings accelerate safetensors nvidia-ml-py wandb
   ```

   RoMAE is excluded from this repo via `.gitignore` because it's a
   pip-installable third-party package; committing it would inflate the
   repo and desynchronize from upstream.

3. **Synthetic data** — regenerate with fixed seeds:

   ```bash
   cd ssm_dk
   python generate_longgap_multisin.py \
       --regime sparse_independent --n_train 1000 --n_val 200 --n_test 200
   python generate_longgap_multisin.py \
       --regime sparse_dependent   --n_train 1000 --n_val 200 --n_test 200
   ```

   This produces `ssm_dk/data/{sparse_independent,sparse_dependent}/{train,val,test}/`
   as HuggingFace arrow datasets, byte-identical across machines given the
   default seeds.

4. **Smoke-test before SLURM submission**:

   ```bash
   cd ssm_dk
   /path/to/pythonenvs/mamba/bin/python -m mv_vs_romae.mamba_mv.train_mv \
       --regime sparse_dependent --dt_mode replace --seed 1 \
       --output_dir /tmp/smoke_mamba --max_epochs 2
   /path/to/pythonenvs/mamba/bin/python -m mv_vs_romae.romae_forecaster.train_romae \
       --regime sparse_dependent --seed 1 \
       --output_dir /tmp/smoke_romae --max_epochs 2
   ```

## Full experiment

From Quest:

```bash
cd ssm_dk/mv_vs_romae/scripts
MAMBA_JID=$(sbatch --parsable run_mamba_mv.sbatch)    # 30 runs
ROMAE_JID=$(sbatch --parsable run_romae.sbatch)        # 10 runs
sbatch --dependency=afterany:${MAMBA_JID}:${ROMAE_JID} run_aggregate.sbatch
```

Outputs land in
`/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/`.
Aggregate stats at `summary.csv`, per-sample raw metrics at
`per_sample.jsonl` inside each seed folder.

See `BUILD_REPORT.md` for the full write-up.
