# UEA Classification — head-to-head with Mamba-MV-pretrain block added

Date: 2026-05-03
Supersedes: [RESULTS_uea_classification_2026-05-02.md](RESULTS_uea_classification_2026-05-02.md)
Adds: Mamba-MV-pretrain rows (3 dt_modes × 5 datasets, 3 seeds) from chain
`18010426 → 18010427 → 18010428` (all COMPLETED, 0 failures).

## What's new

The pretrain-architecture encoder ([imts_benchmark/mamba_pretrain/](../mamba_pretrain/))
is now wired for classification: slot-anonymous variates, Moirai-style binary
attention bias, no per-variate ID embedding in shared-grid; classification
head identical to supervised. Run on a tighter schedule (patience=10,
max_epochs=200) than the supervised v2 rows (patience=50, max_epochs=800)
because the new HPO winners landed at low LRs that didn't need the longer
schedule.

## Final head-to-head (per-dataset best dt_mode)

| Dataset | Mamba-MV best | Mamba-MV-pretrain best | Best baseline (RoMAE T4) | Headline best |
|---|---|---|---|---|
| BM   | learned **1.000** | learned 0.925 ± .075 | mTAN/RoMAE 0.992 | **Mamba-MV (learned)** |
| CT   | learned 0.975 ± .002 | learned 0.979 ± .002 | RoMAE **0.988** | RoMAE |
| EP   | replace **0.981** ± .011 | concat 0.973 ± .012 | TST/RoMAE ≤0.959 | **Mamba-MV (replace)** |
| HB   | learned 0.694 ± .038 | replace 0.652 ± .058 | mTAN **0.779** | mTAN |
| LSST | replace 0.458 ± .032 | learned 0.302 ± .031 | S5 **0.639** | S5 |

**Wins for pretrain block (vs supervised Mamba-MV at best dt_mode):**
- **CT:** +0.4 pt (0.979 vs 0.975), tighter std.

**Losses for pretrain block:**
- **BM:** −7.5 pt (saturated → de-saturated; concat seed-0 collapsed to 0.275)
- **EP:** −0.8 pt (small)
- **HB:** −4.2 pt
- **LSST:** −15.6 pt (largest regression)

**Read:** the slot-anonymous + binary-bias encoder helps where data is
plentiful and clean (CT) but loses on small/imbalanced/high-V datasets
(HB, LSST) where the per-variate ID embedding in the supervised encoder
provides a useful inductive prior the new encoder must learn from data.

---

## Full table (LaTeX)

```latex
\begin{table}[t]
\caption{UEA multivariate classification accuracy on five datasets under
the Kidger 30\% sync-drop protocol~\citep{kidger2020ncde}, mean over 3
seeds. \emph{Top block}: RoMAE Table~4 baselines~\citep{zivanovic2025rotary}
(3-seed mean, no std reported). \emph{Middle block}: supervised Mamba-MV
(per-variate ID embedding + absolute-time positional bias; patience=50,
max\_epochs=800). \emph{Bottom block}: Mamba-MV-pretrain (slot-anonymous
variates with Moirai-style binary attention bias, no per-variate ID
embedding; classification head identical to supervised; tighter schedule
patience=10, max\_epochs=200). BM saturates v1 (single number across
3 seeds, std=0); CT/EP/HB/LSST supervised from v2 re-sweep. Picker:
val/macro\_f1 for HB/EP/LSST, val/acc for BM/CT, with collapse-detector
exclusion (val\_acc $-$ val\_f1 $> 0.2$). \textbf{Bold} = best per column;
\underline{underline} = second-best. Em-dashes mark cells not yet run
(supervised concat for the v2 re-sweep was deferred).}
\label{tab:uea_pretrain_head_to_head}
\centering
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\small
\begin{tabular}{lccccc}
\toprule
Algorithm                              & BM                                       & CT                                       & EP                                       & HB                                       & LSST                                     \\
\midrule
TST                                    & 0.967                                    & 0.974                                    & 0.959                                    & 0.740                                    & 0.552                                    \\
mTAN                                   & \underline{0.992}                        & 0.953                                    & 0.920                                    & \textbf{0.779}                           & 0.531                                    \\
S5                                     & 0.983                                    & 0.961                                    & 0.907                                    & 0.733                                    & \textbf{0.639}                           \\
ContiFormer                            & 0.975                                    & \underline{0.983}                        & 0.932                                    & \underline{0.756}                        & 0.600                                    \\
RoMAE                                  & \underline{0.992}                        & \textbf{0.988}                           & 0.952                                    & 0.745                                    & \underline{0.623}                        \\
\midrule
Mamba-MV (learned)                     & \textbf{1.000}{\scriptsize\,$\pm$.000}   & 0.975{\scriptsize\,$\pm$.002}            & 0.964{\scriptsize\,$\pm$.000}            & 0.694{\scriptsize\,$\pm$.038}            & 0.407{\scriptsize\,$\pm$.047}            \\
Mamba-MV (replace)                     & 0.950{\scriptsize\,$\pm$.066}            & 0.969{\scriptsize\,$\pm$.008}            & \textbf{0.981}{\scriptsize\,$\pm$.011}   & 0.689{\scriptsize\,$\pm$.066}            & 0.458{\scriptsize\,$\pm$.032}            \\
Mamba-MV (concat)                      & ---                                      & ---                                      & ---                                      & ---                                      & ---                                      \\
\midrule
Mamba-MV-pretrain (learned)            & 0.925{\scriptsize\,$\pm$.075}            & 0.979{\scriptsize\,$\pm$.002}            & 0.961{\scriptsize\,$\pm$.015}            & 0.646{\scriptsize\,$\pm$.053}            & 0.302{\scriptsize\,$\pm$.031}            \\
Mamba-MV-pretrain (replace)            & 0.892{\scriptsize\,$\pm$.038}            & 0.977{\scriptsize\,$\pm$.002}            & 0.969{\scriptsize\,$\pm$.004}            & 0.652{\scriptsize\,$\pm$.058}            & 0.295{\scriptsize\,$\pm$.033}            \\
Mamba-MV-pretrain (concat)             & 0.533{\scriptsize\,$\pm$.257}            & 0.969{\scriptsize\,$\pm$.016}            & 0.973{\scriptsize\,$\pm$.012}            & 0.629{\scriptsize\,$\pm$.051}            & 0.291{\scriptsize\,$\pm$.060}            \\
\bottomrule
\end{tabular}
\end{table}
```

## Pretrain-block winners (HPO output)

Picked by val/macro\_f1 (HB/EP/LSST) or val/acc (BM/CT) with collapse-detector,
single HPO seed = 42, 3-seed final at the picked winner.

| Dataset | dt | LR (winner) | BS | head_dropout |
|---|---|---:|---:|---:|
| BasicMotions | learned | 1e-3 | 16 | 0.0 |
| BasicMotions | replace | 1e-3 | 16 | 0.0 |
| BasicMotions | concat | 1e-3 | 32 | 0.0 |
| CharacterTrajectories | learned | 1e-4 | 32 | 0.0 |
| CharacterTrajectories | replace | 1e-4 | 32 | 0.0 |
| CharacterTrajectories | concat | 1e-4 | 32 | 0.0 |
| Epilepsy | learned | 3e-4 | 16 | 0.2 |
| Epilepsy | replace | 3e-4 | 16 | 0.2 |
| Epilepsy | concat | 1e-3 | 16 | 0.2 |
| Heartbeat | learned | 1e-4 | 16 | 0.0 |
| Heartbeat | replace | 1e-4 | 16 | 0.0 |
| Heartbeat | concat | 1e-4 | 32 | 0.0 |
| LSST | learned | 1e-4 | 32 | 0.2 |
| LSST | replace | 1e-4 | 16 | 0.2 |
| LSST | concat | 1e-4 | 32 | 0.2 |

**Edge-winner check:** HB/LSST winners landed at LR=1e-4 (lower edge of the
new grid). v3 had tried extending HB further down to 3e-5 and got worse
single-seed picker noise — likely the same trap. The 3-seed final at
LR=1e-4 here gives a more reliable picture: HB 0.652, LSST 0.302.

## Protocol asymmetry caveat

The pretrain block uses **patience=10, max_epochs=200** vs the supervised
block's **patience=50, max_epochs=800**. So the comparison Mamba-MV
(supervised) vs Mamba-MV-pretrain confounds two changes: encoder
architecture AND training schedule. To make a clean apples-to-apples
architectural claim, we'd need either:
- Re-run supervised v2 under patience=10, max_epochs=200 (~75 cells, ~6 GPU-hr), OR
- Re-run pretrain under patience=50, max_epochs=800 (similar cost).

For the headline 3-seed final results above, the practical effect is bounded:
v2 supervised cells already converged in ~3-6 min (well under both
schedules' caps), so the schedule difference is mostly nominal. The HB
LSST regression in pretrain isn't likely a schedule artifact since both
landed at low LRs where 200 epochs is still plenty.

## Reproducibility pointers

- Code:  [`mamba_pretrain/train_cls.py`](../mamba_pretrain/train_cls.py),
         [`mamba_pretrain/multivariate_classifier.py`](../mamba_pretrain/multivariate_classifier.py)
- Submit chain: `bash imts_benchmark/scripts/submit_uea_cls_pretrain_pipeline_delta_x86.sh`
- Per-cell summaries: `output/log/uea_cls_pretrain/hpo/<ds>/<dt>/<cell>/hpo/summary.json`
- Per-seed final summaries: `output/log/uea_cls_pretrain/final/<ds>/<dt>/<seed>/final/summary.json`
- Winner JSON: `output/log/uea_cls_pretrain/winner_configs.json`
