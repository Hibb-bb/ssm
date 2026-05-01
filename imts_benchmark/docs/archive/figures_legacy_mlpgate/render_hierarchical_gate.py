"""Render the hierarchical-gate classification head as a paper-ready figure.

Two-stage pool:
    Stage 1: per-variate temporal gate, with availability mask removing
             null-state bins from the temporal softmax.
    Stage 2: variate-axis gate -> single pooled vector z.
Then LayerNorm -> Dropout -> Linear -> logits.

Output: hierarchical_gate.{png,pdf} next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 12
V = 4
C = 3


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


rng = np.random.default_rng(0)
gate_t = np.zeros((V, K))
gate_t[0] = softmax(np.linspace(-1, 3.5, K) + 0.4 * rng.standard_normal(K))   # late
gate_t[1] = softmax(np.linspace(3.0, -1, K) + 0.4 * rng.standard_normal(K))   # early
gate_t[2] = softmax(2.5 * np.exp(-((np.arange(K) - 2) ** 2) / 1.5)            # bimodal
                    + 2.5 * np.exp(-((np.arange(K) - 9) ** 2) / 1.5))
gate_t[3] = softmax(0.2 * rng.standard_normal(K))                              # uniform

avail = np.ones((V, K), dtype=bool)
avail[0, 0] = False
avail[1, -1] = False
avail[2, [4, 5]] = False
avail[3, 7] = False
gate_t_masked = gate_t.copy()
gate_t_masked[~avail] = 0.0
gate_t_masked = gate_t_masked / gate_t_masked.sum(axis=1, keepdims=True)

gate_v = softmax(np.array([2.0, 0.5, 1.6, -1.5]))

variate_colors = ["#3b6ea8", "#d96c2f", "#5da955", "#a64ca6"]
class_colors = ["#444444", "#888888", "#bbbbbb"]

# Use a clean 4-row vertical layout. Each row has its own axis with explicit
# title/labels; outer fig.suptitle placed above with extra top padding.
fig, axes = plt.subplots(
    4, 1, figsize=(8.5, 9.0), dpi=130,
    gridspec_kw=dict(height_ratios=[2.4, 4.4, 2.6, 1.5], hspace=0.75),
)
ax_h, ax_t, ax_v, ax_cls = axes

# ============================================================
# Row 0: H^(L) tensor.
# ============================================================
ax_h.set_xlim(-0.5, K - 0.5)
ax_h.set_ylim(-0.5, V - 0.5)
ax_h.invert_yaxis()
ax_h.set_xticks(range(K))
ax_h.set_xticklabels([f"$s_{{{k+1}}}$" for k in range(K)], fontsize=8)
ax_h.set_yticks(range(V))
ax_h.set_yticklabels([f"$v={v+1}$" for v in range(V)], fontsize=9)
ax_h.set_title(
    r"Encoder output  $H^{(L)} \in \mathbb{R}^{K \times V \times d}$",
    fontsize=11.5, pad=10, loc="left",
)
ax_h.set_xlabel("grid time $s_k$", fontsize=9, labelpad=2)
ax_h.tick_params(length=0)
for spine in ax_h.spines.values():
    spine.set_visible(False)

for v in range(V):
    for k in range(K):
        face = variate_colors[v] if avail[v, k] else "#f0f0f0"
        edge = "white" if avail[v, k] else "#bbbbbb"
        ax_h.add_patch(mpatches.FancyBboxPatch(
            (k - 0.42, v - 0.42), 0.84, 0.84,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=1.0,
            alpha=0.85 if avail[v, k] else 1.0,
        ))
        if not avail[v, k]:
            ax_h.text(k, v, r"$\varnothing$", ha="center", va="center",
                      fontsize=9, color="#888")

# Inline legend at right edge.
ax_h.text(
    K - 0.5, -1.15,
    r"colored = real evidence ($m_k^{(n)}=1$)    "
    r"$\varnothing$ = learned null state $h_\varnothing^{(n)}$ ($m_k^{(n)}=0$)",
    fontsize=8.5, color="#444", ha="right", va="center",
)

# ============================================================
# Row 1: Stage 1 -- per-variate temporal gate.
# ============================================================
ax_t.set_xlim(-0.5, K - 0.5)
ax_t.set_ylim(V + 0.05, -0.1)  # invert to match top-down ordering
ax_t.set_xticks(range(K))
ax_t.set_xticklabels([f"$s_{{{k+1}}}$" for k in range(K)], fontsize=8)
ax_t.set_yticks([v + 0.5 for v in range(V)])
ax_t.set_yticklabels([f"$\\alpha^{{t}}_{{k,{v+1}}}$" for v in range(V)],
                     fontsize=10)
ax_t.set_title(
    r"Stage 1 — per-variate temporal gate    "
    r"$\alpha^{t}_{k,n} = \mathrm{softmax}_k\,(\, m_k^{(n)} \cdot \mathrm{MLP}_t(H^{(L)}) \,)$",
    fontsize=11.5, pad=10, loc="left",
)
ax_t.set_xlabel("grid time $s_k$", fontsize=9, labelpad=2)
ax_t.spines["top"].set_visible(False)
ax_t.spines["right"].set_visible(False)

bar_height = 0.85
max_height = max(gate_t_masked.max(), 0.5)
for v in range(V):
    base = v + 0.95
    for k in range(K):
        h = gate_t_masked[v, k] / max_height * bar_height
        if avail[v, k]:
            color = variate_colors[v]
            alpha = 0.9
        else:
            color = "#dddddd"
            alpha = 1.0
        ax_t.add_patch(mpatches.Rectangle(
            (k - 0.35, base - h), 0.7, h,
            facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha,
        ))

# (Caption omitted — the bar shapes already convey "different channels can
# attend to different timesteps". A title-line annotation collided with the
# x-axis label.)

# ============================================================
# Row 2: Stage 2 -- per-variate vectors + variate gate -> z.
# ============================================================
ax_v.set_xlim(-0.5, V * 2.6 + 3.2)
ax_v.set_ylim(-0.6, 2.6)
ax_v.set_axis_off()
ax_v.set_title(
    r"Stage 2 — variate gate    "
    r"$\alpha^{v}_{n} = \mathrm{softmax}_n(\,\mathrm{MLP}_v(H^{(v)})\,)$,    "
    r"$z = \sum_n \alpha^{v}_{n}\, H^{(v)}_{n}$",
    fontsize=11.5, pad=12, loc="left",
)

vec_w = 0.45
vec_h = 1.1
for v in range(V):
    cx = v * 2.6 + 0.5
    # Variate gate weight as a horizontal bar at top.
    g = gate_v[v]
    bar_w = g * 1.6
    ax_v.add_patch(mpatches.Rectangle(
        (cx - bar_w / 2, 2.10), bar_w, 0.10,
        facecolor=variate_colors[v], edgecolor="white", linewidth=0.5,
    ))
    ax_v.text(cx, 2.32, f"$\\alpha^{{v}}_{{{v+1}}}={g:.2f}$",
              ha="center", va="bottom", fontsize=9.0,
              color=variate_colors[v])

    # Pooled per-variate vector.
    ax_v.add_patch(mpatches.FancyBboxPatch(
        (cx - vec_w / 2, 0.55), vec_w, vec_h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor=variate_colors[v], edgecolor="white", linewidth=1.0,
        alpha=0.85,
    ))
    ax_v.text(cx, 0.40, f"$H^{{(v)}}_{{{v+1}}}$",
              ha="center", va="top", fontsize=10,
              color=variate_colors[v])

# Output z to the right.
zx = V * 2.6 + 0.6
ax_v.annotate("", xy=(zx + 0.3, 1.10), xytext=(zx - 0.7, 1.10),
              arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555"))
ax_v.text(zx - 0.2, 1.32, r"$\sum$",
          ha="center", va="bottom", fontsize=14, color="#666")
ax_v.add_patch(mpatches.FancyBboxPatch(
    (zx + 0.4, 0.55), vec_w, vec_h,
    boxstyle="round,pad=0.01,rounding_size=0.06",
    facecolor="#666666", edgecolor="white", linewidth=1.0,
))
ax_v.text(zx + 0.4 + vec_w / 2, 0.40,
          r"$z \in \mathbb{R}^{d}$", ha="center", va="top",
          fontsize=10.5, color="#444", fontweight="bold")

# ============================================================
# Row 3: classifier head.
# ============================================================
ax_cls.set_xlim(0, 10.5)
ax_cls.set_ylim(0, 1.5)
ax_cls.set_axis_off()
ax_cls.set_title("Classifier", fontsize=11.5, pad=8, loc="left")

# z swatch on the left.
ax_cls.add_patch(mpatches.FancyBboxPatch(
    (0.2, 0.45), 0.5, 0.7,
    boxstyle="round,pad=0.01,rounding_size=0.05",
    facecolor="#666666", edgecolor="white", linewidth=1.0,
))
ax_cls.text(0.45, 1.30, "$z$", ha="center", va="bottom", fontsize=11,
            fontweight="bold")

# LN -> Dropout -> Linear boxes.
boxes = [("LayerNorm", 1.4), ("Dropout", 3.5), ("Linear$(d, C)$", 5.6)]
prev_x = 0.7
for lbl, x in boxes:
    ax_cls.add_patch(mpatches.FancyBboxPatch(
        (x, 0.55), 1.6, 0.45,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#eeeeee", edgecolor="#888888", linewidth=1.0,
    ))
    ax_cls.text(x + 0.8, 0.78, lbl, ha="center", va="center", fontsize=9.5)
    ax_cls.annotate("", xy=(x - 0.02, 0.78), xytext=(prev_x, 0.78),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888"))
    prev_x = x + 1.6

# Logits.
logits_x = 7.6
ax_cls.annotate("", xy=(logits_x - 0.05, 0.78),
                xytext=(prev_x, 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888"))
for c in range(C):
    h = 0.30 + 0.20 * c
    ax_cls.add_patch(mpatches.Rectangle(
        (logits_x + c * 0.55, 0.55), 0.40, h,
        facecolor=class_colors[c], edgecolor="white", linewidth=0.5,
    ))
ax_cls.text(logits_x + (C * 0.55) / 2 + 0.15, 1.30,
            r"logits $\in \mathbb{R}^{C}$", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold")

fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.04)

out_png = OUT_DIR / "hierarchical_gate.png"
out_pdf = OUT_DIR / "hierarchical_gate.pdf"
fig.savefig(out_png, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"saved {out_png}")
print(f"saved {out_pdf}")
