"""Merge real-data IMTS results from local CSV (S5, RoMAE) + W&B (Mamba-MV).

Outputs:
  output/log/imts_benchmark_v2_real/aggregate/summary_real.csv
  output/log/imts_benchmark_v2_real/aggregate/summary_real.md      (T-PatchGNN-style table)
  output/log/imts_benchmark_v2_real/aggregate/plot_real_mse.pdf
  output/log/imts_benchmark_v2_real/aggregate/plot_real_mae.pdf

NaN-tolerant (USHCN R^2 blows up, but MSE/MAE are valid).
"""
from __future__ import annotations
import csv, glob, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import wandb

ROOT = Path('/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real')
OUT = ROOT / 'aggregate'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ['physionet', 'activity', 'ushcn']

# ---------- 1. Local CSV (S5, RoMAE) ----------
local_rows = []
for path in sorted(glob.glob(str(ROOT / '*' / '*' / 'default' / 'seed*' / 'test_metrics.csv'))):
    rel = Path(path).relative_to(ROOT)
    model, regime, _, seed_dir, _ = rel.parts
    seed = int(seed_dir.replace('seed', ''))
    with open(path) as f:
        row = next(csv.DictReader(f))
    local_rows.append({
        'source': 'local', 'model': model, 'dataset': regime, 'variant': 'default',
        'seed': seed,
        'mse': float(row.get('test_mse', 'nan')),
        'mae': float(row.get('test_mae', 'nan')),
        'r2':  float(row.get('test_r2', 'nan')),
    })
print(f"Local CSVs: {len(local_rows)} rows from {ROOT}")

# ---------- 2. W&B colleague runs ----------
print("Pulling colleague runs from W&B...")
api = wandb.Api()
wandb_runs = list(api.runs(
    "magicslabnorthwestern/TSKing",
    filters={"config.regime": {"$in": DATASETS}},
    per_page=200,
))

wandb_rows = []
for r in wandb_runs:
    cfg = r.config or {}; summ = r.summary or {}
    dr = (cfg.get('data_root') or '')
    if dr.startswith('/projects/b1094/StarEmbed/'):
        continue  # ours
    if r.state != 'finished':
        continue
    is_mv = ('dt_mode' in cfg) or ('mamba_arch' in cfg) or ('n_perv_layer' in cfg)
    model = 'mamba_mv' if is_mv else cfg.get('model', '?')
    ds = cfg.get('regime', '?')
    seed = cfg.get('seed', None)
    name = (r.name or '')
    # Variant: map by name prefix
    if name.startswith('mamba_mv_confirm_real') or name.startswith('mamba_mv_winner'):
        variant = 'hpo_tuned'
    elif name.startswith('mamba_mv_hpo_real'):
        variant = 'hpo_sweep'
    elif name.startswith('real_mv_local') or name.startswith('real_mv_'):
        variant = 'default'
    else:
        variant = cfg.get('variant', 'default')
    mse = summ.get('test/mse', summ.get('test_mse'))
    if mse is None: continue
    wandb_rows.append({
        'source': 'wandb_colleague', 'model': model, 'dataset': ds, 'variant': variant,
        'seed': seed,
        'mse': float(mse),
        'mae': float(summ.get('test/mae', summ.get('test_mae', float('nan')))),
        'r2':  float(summ.get('test/r2',  summ.get('test_r2', float('nan')))),
    })
print(f"W&B colleague rows: {len(wandb_rows)}")

# ---------- 3. Merge + aggregate ----------
all_rows = local_rows + wandb_rows
groups = defaultdict(list)
for r in all_rows:
    groups[(r['model'], r['variant'], r['dataset'])].append(r)

def agg(vals):
    arr = np.array([v for v in vals if not math.isnan(v)])
    if len(arr) == 0: return float('nan'), float('nan'), 0
    if len(arr) == 1: return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)

summary_rows = []
for (model, variant, ds), rs in sorted(groups.items()):
    mse_m, mse_s, n_mse = agg([r['mse'] for r in rs])
    mae_m, mae_s, _     = agg([r['mae'] for r in rs])
    r2_m,  r2_s,  _     = agg([r['r2'] for r in rs])
    summary_rows.append({
        'model': model, 'variant': variant, 'dataset': ds,
        'n_seeds': n_mse,
        'mse_mean': mse_m, 'mse_std': mse_s,
        'mae_mean': mae_m, 'mae_std': mae_s,
        'r2_mean':  r2_m,  'r2_std':  r2_s,
    })

# Write CSV
csv_path = OUT / 'summary_real.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader(); w.writerows(summary_rows)
print(f"\nWrote {csv_path}")

# Print + write markdown table (T-PatchGNN-style: MSE×10⁻³ for activity, ×10⁻¹ for ushcn)
SCALES = {'physionet': (1e-3, 1e-2), 'activity': (1e-3, 1e-2), 'ushcn': (1e-1, 1e-1)}
def fmt(mean, std, scale):
    if math.isnan(mean): return '—'
    if std == 0: return f"{mean/scale:.2f}"
    return f"{mean/scale:.2f}±{std/scale:.2f}"

print("\n=== T-PatchGNN-style summary (best per column = bold) ===")
header = f"{'Model':<26} | {'PhysioNet MSE×10⁻³':<22} {'PhysioNet MAE×10⁻²':<22} | {'Activity MSE×10⁻³':<22} {'Activity MAE×10⁻²':<22} | {'USHCN MSE×10⁻¹':<22} {'USHCN MAE×10⁻¹':<22}"
print(header)
print('-' * len(header))
# Order rows: S5, RoMAE, Mamba-MV (default), Mamba-MV (HPO)
order = [
    ('s5', 'default', 'S5'),
    ('romae', 'default', 'RoMAE'),
    ('mamba_mv', 'default', 'Mamba-MV (default)'),
    ('mamba_mv', 'hpo_tuned', 'Mamba-MV (HPO)'),
]
md_lines = ["| Model | PhysioNet MSE×10⁻³ | PhysioNet MAE×10⁻² | Activity MSE×10⁻³ | Activity MAE×10⁻² | USHCN MSE×10⁻¹ | USHCN MAE×10⁻¹ |",
            "|---|---|---|---|---|---|---|"]
table = {}
for model, variant, label in order:
    cells = []
    md_cells = [label]
    for ds in DATASETS:
        scale_mse, scale_mae = SCALES[ds]
        match = [r for r in summary_rows if r['model']==model and r['variant']==variant and r['dataset']==ds]
        if match:
            r = match[0]
            cell_mse = fmt(r['mse_mean'], r['mse_std'], scale_mse) + f" (n={r['n_seeds']})"
            cell_mae = fmt(r['mae_mean'], r['mae_std'], scale_mae)
        else:
            cell_mse = '—'; cell_mae = '—'
        cells.append(f"{cell_mse:<22} {cell_mae:<22}")
        md_cells.extend([cell_mse, cell_mae])
    table[label] = cells
    print(f"{label:<26} | {' | '.join(cells)}")
    md_lines.append('| ' + ' | '.join(md_cells) + ' |')

md_path = OUT / 'summary_real.md'
with open(md_path, 'w') as f:
    f.write('# Real-IMTS results (current snapshot)\n\n')
    f.write('Aggregated from local CSVs (S5, RoMAE) and W&B colleague runs (Mamba-MV).\n\n')
    f.write('\n'.join(md_lines) + '\n\n')
    f.write('Notes:\n')
    f.write('- USHCN R² is unreliable (per-variate variance can vanish on flat windows).\n')
    f.write('- T-PatchGNN paper Table 1 best-baseline numbers for reference:\n')
    f.write('  - Activity MSE×10⁻³ = 2.66, MAE×10⁻² = 3.15\n')
    f.write('  - USHCN  MSE×10⁻¹ = 5.00, MAE×10⁻¹ = 3.08\n')
print(f"Wrote {md_path}")

# ---------- 4. Plot ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bar chart: 2 subplots (MSE, MAE), x=model, color=dataset
def make_plot(metric_key, ylabel, fname):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4), sharey=False)
    width = 0.7
    for ax, ds in zip(axes, DATASETS):
        ds_rows = [r for r in summary_rows if r['dataset'] == ds]
        # Order
        ordered = []
        for model, variant, label in order:
            m = [r for r in ds_rows if r['model']==model and r['variant']==variant]
            if m: ordered.append((label, m[0]))
        labels = [x[0] for x in ordered]
        means  = [x[1][f'{metric_key}_mean'] for x in ordered]
        stds   = [x[1][f'{metric_key}_std']  for x in ordered]
        ns     = [x[1]['n_seeds'] for x in ordered]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, width=width, capsize=4,
                      color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'][:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_title(f"{ds.title()}  (n_seeds shown above bars)")
        ax.set_ylabel(ylabel)
        for xi, m, n in zip(x, means, ns):
            ax.text(xi, m, f"n={n}", ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f"Real-IMTS forecasting — {ylabel}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = OUT / fname
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Wrote {out} and .png")
    plt.close(fig)

make_plot('mse', 'Test MSE', 'plot_real_mse.pdf')
make_plot('mae', 'Test MAE', 'plot_real_mae.pdf')
