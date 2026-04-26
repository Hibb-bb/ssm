"""Compare Mamba-MV d=160 (param-matched to S5) vs baselines on Phase 3 high_irreg.

Two scale tiers:
- ~1.4M  : S5 (paper-native, default), Mamba-MV-d160 x {learned, replace, concat}
- ~7.8M  : RoMAE (paper-native, default), Mamba-MV-d384 x {learned, replace, concat}

All on Phase 3, multisin_high_irreg (async dense, fully irregular).
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2')
OUT_DIR = Path('/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/imts_benchmark/docs/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_seeds(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.DataFrame([pd.read_csv(f).iloc[0] for f in files])


def stats(df: pd.DataFrame) -> dict:
    return {
        'mse_mean': df['test_mse'].mean(), 'mse_std': df['test_mse'].std(ddof=0),
        'mae_mean': df['test_mae'].mean(), 'mae_std': df['test_mae'].std(ddof=0),
        'r2_mean':  df['test_r2'].mean(),  'r2_std':  df['test_r2'].std(ddof=0),
        'n_params': int(df['n_params'].iloc[0]),
        'n_seeds': len(df),
    }


# -------- collect --------
configs = [
    # Tier 1: ~1.4M
    ('S5',           '~1.4M', str(ROOT / 'phase3/s5/multisin_high_irreg/default/seed*/test_metrics.csv')),
    ('Mamba-MV (learned)', '~1.4M', str(ROOT / 'hpo_mamba_mv/phase3_winner_d160/multisin_high_irreg/dt-learned_*/seed*/test_metrics.csv')),
    ('Mamba-MV (replace)', '~1.4M', str(ROOT / 'hpo_mamba_mv/phase3_winner_d160/multisin_high_irreg/dt-replace_*/seed*/test_metrics.csv')),
    ('Mamba-MV (concat)',  '~1.4M', str(ROOT / 'hpo_mamba_mv/phase3_winner_d160/multisin_high_irreg/dt-concat_*/seed*/test_metrics.csv')),
    # Tier 2: ~7.8M
    ('RoMAE',        '~7.8M', str(ROOT / 'phase3/romae/multisin_high_irreg/default/seed*/test_metrics.csv')),
    ('Mamba-MV (learned)', '~7.8M', str(ROOT / 'hpo_mamba_mv/phase3_winner/multisin_high_irreg/dt-learned_*/seed*/test_metrics.csv')),
    ('Mamba-MV (replace)', '~7.8M', str(ROOT / 'hpo_mamba_mv/phase3_winner/multisin_high_irreg/dt-replace_*/seed*/test_metrics.csv')),
    ('Mamba-MV (concat)',  '~7.8M', str(ROOT / 'hpo_mamba_mv/phase3_winner/multisin_high_irreg/dt-concat_*/seed*/test_metrics.csv')),
]
rows = []
for name, scale, pat in configs:
    df = load_seeds(pat)
    s = stats(df)
    s['name'] = name
    s['scale'] = scale
    rows.append(s)
results = pd.DataFrame(rows)
print(results[['name', 'scale', 'n_params', 'n_seeds', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std', 'r2_mean', 'r2_std']].to_string(index=False))

# -------- plot --------
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
metrics = [('mse_mean', 'mse_std', 'Test MSE', True),
           ('mae_mean', 'mae_std', 'Test MAE', True),
           ('r2_mean',  'r2_std',  r'Test $R^2$', False)]

# x positions: 4 bars per tier, gap between tiers
n_per_tier = 4
gap = 1.0
xs_t1 = np.arange(n_per_tier)
xs_t2 = np.arange(n_per_tier) + n_per_tier + gap
xs_all = np.concatenate([xs_t1, xs_t2])

# Color palette: baselines gray, Mamba-MV dt_modes color-coded
def color_for(name):
    if name in ('S5', 'RoMAE'):
        return '#7f7f7f'   # gray
    if 'learned' in name:
        return '#1f77b4'   # blue
    if 'replace' in name:
        return '#d62728'   # red
    if 'concat' in name:
        return '#2ca02c'   # green
    return 'k'

colors = [color_for(n) for n in results['name'].tolist()]
labels = results['name'].tolist()

for ax, (m, s, label, lower_is_better) in zip(axes, metrics):
    means = results[m].values
    stds  = results[s].values
    bars = ax.bar(xs_all, means, yerr=stds, color=colors, edgecolor='black', linewidth=0.6,
                  capsize=4, error_kw={'elinewidth': 1, 'ecolor': '#333'})
    ax.set_xticks(xs_all)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(label)
    # annotate winner
    if lower_is_better:
        idx = int(np.argmin(means))
    else:
        idx = int(np.argmax(means))
    ax.bar(xs_all[idx], means[idx], color=colors[idx], edgecolor='black', linewidth=2.0, fill=False, zorder=5)
    # tier dividers + labels
    ax.axvline((xs_t1[-1] + xs_t2[0]) / 2, color='lightgray', linestyle='--', linewidth=0.8, zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text(xs_t1.mean(), ymax * 0.98, 'Param-matched ~1.4M', ha='center', va='top', fontsize=9, color='#444')
    ax.text(xs_t2.mean(), ymax * 0.98, 'Original scale ~7.8M', ha='center', va='top', fontsize=9, color='#444')
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)

fig.suptitle('Phase 3 (async dense) — multisin_high_irreg — 5 seeds, error bars = std (ddof=0)', y=0.995, fontsize=12)
fig.tight_layout()
out_png = OUT_DIR / 'phase3_high_irreg_d160_vs_baselines.png'
out_pdf = OUT_DIR / 'phase3_high_irreg_d160_vs_baselines.pdf'
fig.savefig(out_png, dpi=180, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'\nSaved: {out_png}')
print(f'Saved: {out_pdf}')
