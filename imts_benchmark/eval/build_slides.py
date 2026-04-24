"""Build a compact slide deck (.pptx) summarizing Phase 2 / 3 / 4-2 results.

Layout: 16:9 (13.33" x 7.5"). White background, minimal text, concise bullets.

Slide order (Phase 4-2 first — most impressive result leads):
  1. Synthetic IMTS benchmark (setup)
  2. Models (correct param counts: S5 is 1.40M paper-native; others ~7.8M)
  3. Phase 4-2 — async + random-variate gap  (MSE + R² bars + winner-grid)
  4. Phase 3  — async dense                  (MSE + R² bars + winner-grid)
  5. Phase 2  — sync dense                   (MSE + R² bars + winner-grid)
  6. Next steps
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


DOCS = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/imts_benchmark/docs")
OUT = DOCS / "imts_benchmark_slides.pptx"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_title(slide, text: str, size: int = 32):
    tx = slide.shapes.add_textbox(Inches(0.4), Inches(0.18), Inches(12.53), Inches(0.8))
    tf = tx.text_frame
    tf.margin_left = Inches(0); tf.margin_right = Inches(0)
    tf.margin_top = Inches(0);  tf.margin_bottom = Inches(0)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size); p.font.bold = True
    p.font.color.rgb = RGBColor(0x14, 0x14, 0x14)


def add_bullets(slide, bullets, left, top, width, height, size: int = 18):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0); tf.margin_right = Inches(0)
    tf.margin_top = Inches(0);  tf.margin_bottom = Inches(0)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {b}"
        p.font.size = Pt(size)
        p.font.color.rgb = RGBColor(0x1f, 0x1f, 0x1f)
        p.space_after = Pt(6)
    return tx


def add_img(slide, path: Path, left, top, width=None, height=None):
    return slide.shapes.add_picture(str(path), left, top,
                                    width=width, height=height)


# ---------------- content -------------------------------------------------

def slide_data_setup(prs):
    s = blank(prs)
    add_title(s, "Synthetic IMTS benchmark")
    add_bullets(s, [
        "3 variates per sample:  v1, v2 ~ sin(2πf·t + φ)   ·   v3 = w·v1 + (1−w)·v2,  w ~ U(0.2, 0.8)",
        "Cross-variate info is load-bearing — single-variate models pay for ignoring it",
        "120 obs / variate   ·   history = 8.0   ·   forecast = [8, 10]",
        "4 irregularity regimes via frac_regular ∈ {1.0, 0.8, 0.3, 0.0}   →   Regular / Low / Med / High",
        "Phase 2 — sync timestamps (same ts across variates)",
        "Phase 3 — async timestamps (per-variate independent)",
        "Phase 4-2 — Phase 3 + random-variate long gap per sample",
    ], Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5), size=22)


def slide_models(prs):
    s = blank(prs)
    add_title(s, "Models")
    add_bullets(s, [
        "Mamba-MV (ours)  —  multivariate SSM:  per-variate SSM + variable-axis attention + fusion Mamba on shared grid",
        "     d_model = 256   ·   d_state = 16   ·   3 per-variate + 3 fusion layers   ·   7.79 M params",
        "     dt_modes:   learned (vanilla Mamba Δ)   /   replace (Δ = Δtrue)",
        "S5  (Smith+ ICLR '23)  —  per-variate diagonal SSM, paper-native config",
        "     d_model = 128   ·   state_dim = 256   ·   6 layers   ·   1.40 M params",
        "RoMAE  —  Rotary-PE masked autoencoder (2D positions: timestamp + variate-id)",
        "     encoder  d=288, 6-head, 7 layers   ·   decoder d=180, 3-head, 2 layers   ·   7.80 M params",
        "mTAN — dropped (training collapse across 5 configs)",
        "Shared recipe:  AdamW (lr=5e-4 default; S5 overrides to paper-native lr=1e-3)  ·  cosine 100→1600 steps  ·  patience=50  ·  n=5 seeds",
    ], Inches(0.5), Inches(1.25), Inches(12.3), Inches(6.0), size=16)


def slide_phase_results(prs, phase_title: str,
                        mse_img: Path, r2_img: Path, winner_img: Path,
                        bullets: list[str]):
    s = blank(prs)
    add_title(s, phase_title)
    # Compact bullets under title.
    add_bullets(s, bullets, Inches(0.4), Inches(1.0),
                Inches(12.5), Inches(1.1), size=16)
    # Bars: MSE + R² side-by-side, mid slide.
    add_img(s, mse_img,    Inches(0.25), Inches(2.2), width=Inches(6.4))
    add_img(s, r2_img,     Inches(6.7),  Inches(2.2), width=Inches(6.4))
    # Winner grid centered below.
    add_img(s, winner_img, Inches(2.33), Inches(5.35), width=Inches(8.7))


def slide_phase4_2(prs):
    slide_phase_results(
        prs,
        phase_title="Phase 4-2 — async + random-variate long gap",
        mse_img=DOCS / "phase4_2_results/phase4_2_metric_mse.png",
        r2_img=DOCS / "phase4_2_results/phase4_2_metric_r2.png",
        winner_img=DOCS / "phase4_2_results/phase4_2_winner_grid.png",
        bullets=[
            "Mamba-MV wins 10/12 (regime × freq) cells — multivariate SSM fills the gapped variate from the other two",
            "S5 (per-variate) can’t recover the gapped variate — loses across almost all cells",
            "Mamba dt_mode:  learned > replace here  (Δtrue spikes near gaps amplified by replace)",
        ],
    )


def slide_phase3(prs):
    slide_phase_results(
        prs,
        phase_title="Phase 3 — async dense (per-variate timestamps)",
        mse_img=DOCS / "phase3_results/phase3_metric_mse.png",
        r2_img=DOCS / "phase3_results/phase3_metric_r2.png",
        winner_img=DOCS / "phase3_results/phase3_winner_grid.png",
        bullets=[
            "S5 wins 7/12 cells (all Med/High irreg × all freqs) — HPO gate triggered (Mamba replace loses 3/3 on irregular regimes)",
            "Mamba-MV wins only the sync-like Regular column (all 3 freqs) and loses under async irregularity",
            "Training time — Mamba ~3.5 min  ·  RoMAE ~3.6 min  ·  S5 ~11 min  (Mamba cheapest by ~3×)",
        ],
    )


def slide_phase2(prs):
    slide_phase_results(
        prs,
        phase_title="Phase 2 — sync dense (baseline check)",
        mse_img=DOCS / "phase2_final/phase2_metric_mse.png",
        r2_img=DOCS / "phase2_final/phase2_metric_r2.png",
        winner_img=DOCS / "phase2_final/phase2_winner_grid.png",
        bullets=[
            "Mamba-MV wins the Regular column (all freqs); S5 wins Low+Med; RoMAE wins High",
            "Fair-comparison baseline — paper-native S5 + paper-native RoMAE at matched recipe",
        ],
    )


def slide_summary(prs):
    s = blank(prs)
    add_title(s, "Summary — who wins where")
    add_bullets(s, [
        "Mamba-MV wins Regular (all phases) AND almost everywhere under gaps (Phase 4-2)",
        "     →  multivariate-SSM + cross-variate fusion pays off when variates must be combined",
        "S5 wins async-no-gap irregularity (Phase 3 Med/High columns, any freq)",
        "     →  single-variate SSM with paper-native state=256 is a strong forecasting baseline",
        "RoMAE wins narrow niches (Phase 2 High, Phase 3 Low irreg at low/mid freq)",
        "Next question:  can HPO close the gap on Phase 3 Med/High (where Mamba replace loses 3/3)?",
    ], Inches(0.5), Inches(1.3), Inches(12.3), Inches(6.0), size=20)


def slide_next(prs):
    s = blank(prs)
    add_title(s, "Next steps")
    add_bullets(s, [
        "Mamba-MV HPO  —  sweep d_model, d_state, depth, LR (per HPO_PLAN.md, gated on Phase 3 result)",
        "Re-evaluate post-HPO on Phase 2 / 3 / 4-2",
        "Apply winning config to real IMTS data (astronomy / clinical)",
    ], Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.5), size=24)


def main():
    prs = new_deck()
    slide_data_setup(prs)
    slide_models(prs)
    slide_phase4_2(prs)    # lead with most impressive
    slide_phase3(prs)
    slide_phase2(prs)
    slide_summary(prs)
    slide_next(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB,  {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
