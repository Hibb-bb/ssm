#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=mamba_plot
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed/%x_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba

${MAMBA_ENV}/bin/python << 'PYEOF'
import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datasets

sys.path.insert(0, "/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm/mamba_experiments")
from forecaster.mamba_forecaster import MambaForecaster

DATA_ROOT = "/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data_nobs160"
LOG_BASE = "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed"
OUTPUT_DIR = os.path.join(LOG_BASE, "comparison_high_irreg")
HISTORY = 7.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load test data
base = os.path.join(DATA_ROOT, "sinusoidal_high_irreg")
hf = datasets.load_from_disk(os.path.join(base, "test"))
samples = []
for row in hf:
    values = np.array(row["target"], dtype=np.float32)
    timestamps = np.array(row["timestamp"], dtype=np.float32)
    delta_t = np.zeros_like(timestamps)
    delta_t[1:] = np.diff(timestamps)
    samples.append(dict(values=values, timestamps=timestamps, delta_t=delta_t))
print(f"Loaded {len(samples)} test samples")


def run_inference(ckpt_path, samples, history, device):
    model = MambaForecaster.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    results = []
    batch_size = 32
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        values = torch.stack([torch.from_numpy(s["values"]) for s in batch_samples]).to(device)
        timestamps = torch.stack([torch.from_numpy(s["timestamps"]) for s in batch_samples]).to(device)
        delta_t = torch.stack([torch.from_numpy(s["delta_t"]) for s in batch_samples]).to(device)

        pred_mask = timestamps >= history
        masked_values = values.clone()
        masked_values[pred_mask] = 0.0

        with torch.no_grad():
            preds = model(masked_values, delta_t=delta_t)

        B = values.shape[0]
        for i in range(B):
            m = pred_mask[i]
            pred_hat = preds[i][m].cpu().numpy()
            pred_true = values[i][m].cpu().numpy()
            pred_times = timestamps[i][m].cpu().numpy()

            ctx_mask = ~m.cpu().numpy()
            ctx_times = timestamps[i].cpu().numpy()[ctx_mask]
            ctx_vals = values[i].cpu().numpy()[ctx_mask]

            ss_res = np.sum((pred_true - pred_hat) ** 2)
            ss_tot = np.sum((pred_true - np.mean(pred_true)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            mse = float(np.mean((pred_true - pred_hat) ** 2))

            results.append(dict(
                ctx_times=ctx_times, ctx_vals=ctx_vals,
                pred_times=pred_times, pred_vals=pred_true,
                pred_hat=pred_hat, r2=r2, mse=mse,
            ))
    return results


def estimate_frequency(times, values):
    n_interp = 256
    t_uniform = np.linspace(times.min(), times.max(), n_interp)
    v_uniform = np.interp(t_uniform, times, values)
    v_uniform -= np.mean(v_uniform)
    dt = (times.max() - times.min()) / (n_interp - 1)
    fft_vals = np.abs(np.fft.rfft(v_uniform))
    freqs = np.fft.rfftfreq(n_interp, d=dt)
    fft_vals[0] = 0
    return freqs[np.argmax(fft_vals)]


# Run inference for all 3 variants (seed 1)
variants = [
    ("Vanilla Mamba", "vanilla_mamba"),
    ("True $\\Delta t$", "mamba_true_dt"),
    ("Hybrid $\\Delta t$", "mamba_hybrid_dt"),
]

all_results = []
labels = []
for label, vname in variants:
    ckpt = os.path.join(LOG_BASE, vname, "high_irreg", "seed1", "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(LOG_BASE, vname, "high_irreg", "seed1", "checkpoints", "best-v1.ckpt")
    print(f"Running {label} inference from {ckpt}...")
    results = run_inference(ckpt, samples, HISTORY, device)
    mean_r2 = np.mean([r["r2"] for r in results])
    print(f"  {len(results)} samples, mean R²={mean_r2:.4f}")
    all_results.append(results)
    labels.append(label)

# Estimate frequencies from first variant's results
ref = all_results[0]
sample_freqs = []
for res in ref:
    all_t = np.concatenate([res["ctx_times"], res["pred_times"]])
    all_v = np.concatenate([res["ctx_vals"], res["pred_vals"]])
    order = np.argsort(all_t)
    sample_freqs.append(estimate_frequency(all_t[order], all_v[order]))
sample_freqs = np.array(sample_freqs)
sort_idx = np.argsort(sample_freqs)

# Plot 1: frequency-sorted comparison
os.makedirs(OUTPUT_DIR, exist_ok=True)
n_models = len(all_results)
percentiles = (0.25, 0.5, 0.75, 1.0)
n_total = len(ref)
selected = [int(round(p * (n_total - 1))) for p in percentiles]

fig, axes = plt.subplots(len(selected), n_models, figsize=(7 * n_models, 4 * len(selected)))
if len(selected) == 1:
    axes = axes[np.newaxis, :]

for row, sel in enumerate(selected):
    for col, (label, results) in enumerate(zip(labels, all_results)):
        ax = axes[row, col]
        res = results[sort_idx[sel]]
        ax.plot(res["ctx_times"], res["ctx_vals"], "o-", color="gray",
                markersize=3, alpha=0.5, label="Context", linewidth=0.8)
        ax.plot(res["pred_times"], res["pred_vals"], "o", color="black",
                markersize=3, label="True", zorder=5)
        ax.plot(res["pred_times"], res["pred_hat"], "s-", color="red",
                markersize=3, label="Predicted", zorder=5, linewidth=0.8, alpha=0.8)
        ax.fill_between(res["pred_times"], res["pred_vals"], res["pred_hat"],
                        alpha=0.12, color="cornflowerblue", label="|Error|")
        ax.set_title(
            f"{label} | R²={res['r2']:.3f}, MSE={res['mse']:.4f}  "
            f"(f={sample_freqs[sort_idx[sel]]:.2f} Hz)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time")

fig.suptitle("Mamba variants (leakage-fixed) — sorted by frequency (low -> high)",
             fontsize=14, y=1.01)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "mamba_fixed_sortby_frequency.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# Plot 2: distributions
colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 3)))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for i in range(n_models):
    for j in range(i + 1, n_models):
        r2_i = [r["r2"] for r in all_results[i]]
        r2_j = [r["r2"] for r in all_results[j]]
        ax.scatter(r2_i, r2_j, alpha=0.3, s=12, label=f"{labels[i]} vs {labels[j]}")
lims = [-1.5, 1.1]
ax.plot(lims, lims, "k--", alpha=0.5)
ax.set_xlabel("R² (model on x)")
ax.set_ylabel("R² (model on y)")
ax.set_title("Per-sample R² scatter", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[1]
for i, (results, label) in enumerate(zip(all_results, labels)):
    r2 = [r["r2"] for r in results]
    ax.hist(r2, bins=40, alpha=0.5,
            label=f"{label} (mean={np.mean(r2):.3f})", color=colors[i])
ax.set_xlabel("R²")
ax.set_ylabel("Count")
ax.set_title("R² Distribution", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
for i, (results, label) in enumerate(zip(all_results, labels)):
    mse = [r["mse"] for r in results]
    ax.hist(mse, bins=40, alpha=0.5,
            label=f"{label} (mean={np.mean(mse):.4f})", color=colors[i])
ax.set_xlabel("MSE")
ax.set_ylabel("Count")
ax.set_title("MSE Distribution", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "mamba_fixed_distributions.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

print("\nDone!")
PYEOF
