"""
Plot comparison of Mamba variants on sinusoidal test data,
sorted by ground-truth frequency — analogous to the MOIRAI
comparison_sortby_frequency.png plot.

Usage (needs GPU):
    python -m mamba_forecaster.plot_mamba_comparison

Can also run on CPU with --cpu flag.
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets
from mamba_forecaster.mamba_forecaster import MambaForecaster


def load_test_data(data_root, irregularity):
    base = os.path.join(data_root, f"sinusoidal_{irregularity}")
    hf = datasets.load_from_disk(os.path.join(base, "test"))
    samples = []
    for row in hf:
        values = np.array(row["target"], dtype=np.float32)
        timestamps = np.array(row["timestamp"], dtype=np.float32)
        delta_t = np.zeros_like(timestamps)
        delta_t[1:] = np.diff(timestamps)
        samples.append(dict(values=values, timestamps=timestamps, delta_t=delta_t))
    return samples


def run_mamba_inference(ckpt_path, samples, history, device):
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

        with torch.no_grad():
            preds = model(values, delta_t=delta_t)

        pred_mask = timestamps >= history
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
                ctx_times=ctx_times,
                ctx_vals=ctx_vals,
                pred_times=pred_times,
                pred_vals=pred_true,
                pred_hat=pred_hat,
                r2=r2,
                mse=mse,
            ))
    return results


def estimate_frequency(times, values):
    t_min = min(times)
    t_max = max(times)
    n_interp = 256
    t_uniform = np.linspace(t_min, t_max, n_interp)
    v_uniform = np.interp(t_uniform, times, values)
    v_uniform = v_uniform - np.mean(v_uniform)
    dt = (t_max - t_min) / (n_interp - 1)
    fft_vals = np.abs(np.fft.rfft(v_uniform))
    freqs = np.fft.rfftfreq(n_interp, d=dt)
    fft_vals[0] = 0
    dominant_freq = freqs[np.argmax(fft_vals)]
    return dominant_freq


def plot_comparison_by_frequency(all_results_list, labels, output_dir, n_plots=4):
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(all_results_list)
    ref = all_results_list[0]

    sample_freqs = []
    for idx, res in enumerate(ref):
        all_t = np.concatenate([res["ctx_times"], res["pred_times"]])
        all_v = np.concatenate([res["ctx_vals"], res["pred_vals"]])
        order = np.argsort(all_t)
        freq = estimate_frequency(all_t[order], all_v[order])
        sample_freqs.append(freq)
    sample_freqs = np.array(sample_freqs)

    n_total = len(ref)
    percentiles = (0.25, 0.5, 0.75, 1.0)
    sort_idx = np.argsort(sample_freqs)
    selected = [int(round(p * (n_total - 1))) for p in percentiles]
    selected = selected[:min(n_plots, len(selected))]

    fig, axes = plt.subplots(len(selected), n_models, figsize=(7 * n_models, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for row, sel in enumerate(selected):
        for col, (label, results) in enumerate(zip(labels, all_results_list)):
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
                f"(f={sample_freqs[sort_idx[sel]]:.2f} Hz)",
                fontsize=10,
            )
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)
            ax.set_xlabel("Time")

    fig.suptitle("Mamba variants — sorted by ground-truth frequency (low → high)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "mamba_comparison_sortby_frequency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_distributions(all_results_list, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(all_results_list)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 3)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for i in range(n_models):
        for j in range(i + 1, n_models):
            r2_i = [r["r2"] for r in all_results_list[i]]
            r2_j = [r["r2"] for r in all_results_list[j]]
            ax.scatter(r2_i, r2_j, alpha=0.3, s=12, label=f"{labels[i]} vs {labels[j]}")
    lims = [-1.5, 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("R² (model on x)")
    ax.set_ylabel("R² (model on y)")
    ax.set_title("Per-sample R² scatter", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, (results, label) in enumerate(zip(all_results_list, labels)):
        r2 = [r["r2"] for r in results]
        ax.hist(r2, bins=40, alpha=0.5,
                label=f"{label} (mean={np.mean(r2):.3f})", color=colors[i])
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i, (results, label) in enumerate(zip(all_results_list, labels)):
        mse = [r["mse"] for r in results]
        ax.hist(mse, bins=40, alpha=0.5,
                label=f"{label} (mean={np.mean(mse):.4f})", color=colors[i])
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "mamba_comparison_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data_nobs160"
    log_base = "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_nobs160"
    output_dir = os.path.join(log_base, "comparison_high_irreg")
    history = 7.0

    ckpt_vanilla = log_base + "/vanilla_mamba/high_irreg/seed1/checkpoints/best-v1.ckpt"
    ckpt_true_dt = log_base + "/mamba_true_dt/high_irreg/seed1/checkpoints/best-v1.ckpt"
    ckpt_hybrid = log_base + "/mamba_hybrid_dt/high_irreg/seed1/checkpoints/best-v1.ckpt"

    print("Loading test data...")
    samples = load_test_data(data_root, "high_irreg")
    print(f"  {len(samples)} test samples")

    print("Running Vanilla Mamba inference...")
    results_vanilla = run_mamba_inference(ckpt_vanilla, samples, history, device)
    print(f"  {len(results_vanilla)} samples, mean R²={np.mean([r['r2'] for r in results_vanilla]):.4f}")

    print("Running True Δt inference...")
    results_true_dt = run_mamba_inference(ckpt_true_dt, samples, history, device)
    print(f"  {len(results_true_dt)} samples, mean R²={np.mean([r['r2'] for r in results_true_dt]):.4f}")

    print("Running Hybrid Δt inference...")
    results_hybrid = run_mamba_inference(ckpt_hybrid, samples, history, device)
    print(f"  {len(results_hybrid)} samples, mean R²={np.mean([r['r2'] for r in results_hybrid]):.4f}")

    all_results = [results_vanilla, results_true_dt, results_hybrid]
    labels = ("Vanilla Mamba", "True $\\Delta t$", "Hybrid $\\Delta t$")

    print("Plotting frequency comparison...")
    plot_comparison_by_frequency(all_results, labels, output_dir, n_plots=4)

    print("Plotting distributions...")
    plot_distributions(all_results, labels, output_dir)

    print("Done!")
