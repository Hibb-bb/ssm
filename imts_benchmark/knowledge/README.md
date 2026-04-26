# knowledge/ — reference papers for the IMTS benchmark

This folder centralizes the four papers relevant to interpreting the Mamba-MV vs S5 vs RoMAE benchmark. Each `.md` file summarizes the paper's contents with specific section pointers to where a claim in our [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md) or [HPO_PLAN.md](../docs/HPO_PLAN.md) is grounded.

## Index

| Filename | Paper | Role in this project |
|---|---|---|
| [S5_smith_warrington_linderman_ICLR2023.md](S5_smith_warrington_linderman_ICLR2023.md) | Simplified State Space Layers for Sequence Modeling (Smith, Warrington, Linderman; ICLR 2023; arXiv:2208.04933) | S5 baseline; defines the true-Δt mechanism our wrapper uses (§3.3, §6.3) and the HiPPO-N init that drives per-variate strength (§3.2, §4.2) |
| [RoMAE_zivanovic_etal_2025.md](RoMAE_zivanovic_etal_2025.md) | Rotary Masked Autoencoders are Versatile Learners (Zivanovic et al.; 2025; arXiv:2505.20535) | RoMAE baseline; defines RoPEND + the variate-as-axial-RoPE-dim strategy (§4.2) used in our wrapper |
| [mamba_gu_dao_2023.md](mamba_gu_dao_2023.md) | Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao; 2023; arXiv:2312.00752) | **PENDING user attachment.** Defines the semantics of Δ as a selection gate, crucial to interpreting the `dt_mode=replace` vs `learned` reversal on Phase 4. |
| [starembed_draft_2026.md](starembed_draft_2026.md) | Data-centric pretraining on TSFMs for irregular time series (our own draft) | What we are writing. See Part 1 §9 of [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md) for recommended framing changes. |

## To add the actual PDFs
Drop them into this folder with the filenames below:

```
S5_smith_warrington_linderman_ICLR2023.pdf
RoMAE_zivanovic_etal_2025.pdf
mamba_gu_dao_2023.pdf
starembed_draft_2026.pdf
```

The `.md` summary stubs remain authoritative for the section-pointers used in our interpretation — the PDFs are reference-only.
