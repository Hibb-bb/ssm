"""One-shot converter: Time-IMM CSVs -> sparse-flat HF parquet.

Outputs the same schema as ``tpatchgnn_data/{regime}/{split}`` so that
``pretrain.val_imts.IMTSValDataset`` can consume them directly.

Reproduces the exact preprocessing in
``IMM-TSF/lib/parse_datasets.py:: ChunkedTimeSeriesDataset``
so reported MSE/MAE on the test split matches the IMM-TSF paper Table.
Specifically:

  - per-record GLOBAL z-score of values (their lines 103-111),
  - chunking with per-dataset (history, pred_window, stride, time_unit)
    from ``IMM-TSF/main.py::update_args_for_dataset`` (lines 788-836),
  - drop chunks whose history span has no text entry (their lines 217-221),
  - per-record temporal 60/20/20 split (their ``split_method="sample"``
    branch, lines 715-730), the default used by ``main.py:1227`` and
    ``main_all.py:128``.

After conversion, only the **test** split is wired into the live
pretraining val loop (see ``pretrain/datamodule.py``).  Train / val
splits are written for completeness but never loaded by us.

Usage
-----

    python convert_imm_tsf_to_sparse.py \
        --imm_tsf_root /home/ubuntu/imm_tsf_data \
        --out_root     /home/ubuntu/hongyu/ssm/imts_benchmark/data/imm_tsf_sparse

Run once.  Re-run only if the upstream data changes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import datasets
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-dataset config -- mirror IMM-TSF/main.py::update_args_for_dataset
# (https://github.com/blacksnail789521/IMM-TSF/blob/master/main.py#L788-L836)
# ---------------------------------------------------------------------------

# (history, pred_window, stride, time_unit)
DATASET_CONFIG: "OrderedDict[str, tuple[int, int, int, str]]" = OrderedDict([
    ("GDELT",        (14, 14, 14, "days")),
    ("RepoHealth",   (31, 31, 31, "days")),
    ("FNSPID",       (31, 31, 31, "days")),
    ("ClusterTrace", (12, 12, 12, "hours")),
    ("StudentLife",  (31, 31, 31, "days")),
    ("ILINet",       (36, 36,  4, "weeks")),
    ("CESNET",       ( 7,  7,  7, "days")),
    ("EPA-Air",      ( 7,  7,  7, "days")),
    # MIMIC excluded: data-use agreement disallows redistribution
    # (Time-IMM README, MIMIC Preprocessing section).
])

UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours":   3600.0,
    "days":    86400.0,
    "weeks":   604800.0,
}

# pandas freq strings, only used for metadata (model side ignores)
FREQ_STR = {
    "seconds": "S",
    "minutes": "T",
    "hours":   "H",
    "days":    "D",
    "weeks":   "W",
}


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def _load_record(rec_dir: Path, time_unit: str):
    """Mirror IMM-TSF/lib/parse_datasets.py:90-172, time-series side.

    Returns
    -------
    feat_cols : list[str]
    tt        : np.ndarray (T,) timestamps in ``time_unit``
    vals      : np.ndarray (T, V) raw (unnormalized) values, NaN-filled
    mask      : np.ndarray (T, V) bool, True = observation present
    text_tt   : np.ndarray (M,)  text timestamps in ``time_unit``,
                relative to the same record start as ``tt``
    """
    sec_per_unit = UNIT_SECONDS[time_unit]

    ts_path = rec_dir / "time_series.csv"
    if not ts_path.is_file():
        raise FileNotFoundError(ts_path)

    df = pd.read_csv(ts_path)
    df["_ts_raw"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("_ts_raw").reset_index(drop=True)

    # Canonicalize feature ordering: Time-IMM CSVs sometimes list the
    # same features in different orders across records (e.g. FNSPID's
    # 'volume' column is column 0 for some tickers and column -1 for
    # others). Sort lexicographically so a single canonical (V, mu, std,
    # n_obs_per_var) layout applies to every record in the dataset.
    feat_cols = sorted(
        c for c in df.columns if c not in ("date_time", "record_id", "_ts_raw")
    )

    base = df["_ts_raw"].min()
    secs = (df["_ts_raw"] - base).dt.total_seconds().values
    tt = (secs / sec_per_unit).astype(np.float64)

    # df[feat_cols] respects the order of feat_cols (which we sorted above),
    # so vals[:, v] always refers to feat_cols[v] regardless of CSV order.
    vals = df[feat_cols].values.astype(np.float32)
    mask = ~np.isnan(vals)

    # text side
    text_tt: np.ndarray
    text_path = rec_dir / "text.csv"
    if text_path.is_file():
        try:
            tdf = pd.read_csv(text_path)
            if "date_time" in tdf.columns:
                t_raw = pd.to_datetime(tdf["date_time"])
                txt_secs = (t_raw - base).dt.total_seconds().values
                text_tt = (txt_secs / sec_per_unit).astype(np.float64)
            else:
                text_tt = np.zeros(0, dtype=np.float64)
        except Exception:
            text_tt = np.zeros(0, dtype=np.float64)
    else:
        text_tt = np.zeros(0, dtype=np.float64)

    return feat_cols, tt, vals, mask, text_tt


def _record_norm_stats(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature global mean/std over the entire record, ignoring NaNs.

    Mirrors IMM-TSF/lib/parse_datasets.py:103-111 (which uses pandas'
    ``col.mean()`` / ``col.std()`` -- pandas drops NaNs by default).
    """
    V = vals.shape[1]
    mu = np.zeros(V, dtype=np.float32)
    sd = np.ones(V, dtype=np.float32)
    for v in range(V):
        col = vals[:, v]
        col = col[~np.isnan(col)]
        if col.size == 0:
            mu[v] = 0.0
            sd[v] = 1.0
            continue
        m = float(col.mean())
        s = float(col.std(ddof=1)) if col.size > 1 else 0.0
        if not np.isfinite(s) or s == 0.0:
            s = 1.0
        if not np.isfinite(m):
            m = 0.0
        mu[v] = m
        sd[v] = s
    return mu, sd


def _build_chunks_for_record(
    rec_id: str,
    tt: np.ndarray,
    vals: np.ndarray,
    mask: np.ndarray,
    text_tt: np.ndarray,
    history: int,
    pred_window: int,
    stride: int,
    enforce_text_filter: bool,
) -> list[dict]:
    """Mirror IMM-TSF chunking + text filter.

    Each chunk is dict with keys consumed downstream:
      ``chunk_id, sub_tt (T,), sub_vals (T, V), sub_mask (T, V),
      hist_count (int), pred_count (int)``.

    Skipped reasons match parse_datasets.py:182-227.
    """
    total = float(history + pred_window)
    H = float(history)

    if tt.size == 0:
        return []
    st = float(tt.min())
    t_max = float(tt.max())

    chunks: list[dict] = []
    cnt = 0
    drop_no_text = 0
    drop_empty_split = 0

    while st + total <= t_max + 1e-12:
        idx = np.where((tt >= st) & (tt < st + total))[0]
        if idx.size >= 2:
            sub_tt = tt[idx] - st
            sub_vals = vals[idx]
            sub_mask = mask[idx]

            hist_sel = sub_tt < H
            pred_sel = sub_tt >= H
            hist_mask = sub_mask[hist_sel]
            pred_mask = sub_mask[pred_sel]
            if hist_mask.sum() == 0 or pred_mask.sum() == 0:
                drop_empty_split += 1
                st += stride
                continue

            if enforce_text_filter:
                hist_end = st + H
                has_text = bool(np.any((text_tt >= st) & (text_tt < hist_end)))
                if not has_text:
                    drop_no_text += 1
                    st += stride
                    continue

            chunks.append({
                "chunk_id":   f"{rec_id}_chunk{cnt}",
                "sub_tt":     sub_tt.astype(np.float64, copy=False),
                "sub_vals":   sub_vals,
                "sub_mask":   sub_mask,
                "hist_count": int(hist_mask.sum()),
                "pred_count": int(pred_mask.sum()),
            })
            cnt += 1

        st += stride

    return chunks


def _chunk_to_sparse_row(
    chunk: dict,
    mu: np.ndarray,
    sd: np.ndarray,
    history: float,
) -> dict:
    """Convert a (T, V) chunk to sparse-flat per-variate format.

    Output schema matches ``tpatchgnn_data`` layout:
      item_id (str), target (list[float]), timestamp (list[float]),
      past_feat_dynamic_real (list[float]), n_obs_per_var (list[int]),
      history (float), value_mu_per_var (list[float]),
      value_std_per_var (list[float]).

    ``target`` is in **original units** -- the val Dataset z-scores it
    on the fly using the stored ``value_mu_per_var`` /
    ``value_std_per_var``.  This keeps downstream metric reporting in
    original units trivial (just multiply by std + add mu).
    """
    sub_tt = chunk["sub_tt"]
    sub_vals = chunk["sub_vals"]
    sub_mask = chunk["sub_mask"]
    T, V = sub_vals.shape

    target_flat: list[float] = []
    ts_flat:     list[float] = []
    deltat_flat: list[float] = []
    n_obs_per_var: list[int] = []

    for v in range(V):
        valid = np.where(sub_mask[:, v])[0]
        n_obs_per_var.append(int(valid.size))
        if valid.size == 0:
            continue
        var_ts = sub_tt[valid].astype(np.float32, copy=False)
        var_vals = sub_vals[valid, v].astype(np.float32, copy=False)
        # delta_t convention used by tpatchgnn_data: first element 0,
        # rest = consecutive differences.
        var_dt = np.zeros_like(var_ts)
        if var_ts.size > 1:
            var_dt[1:] = np.diff(var_ts)
        target_flat.extend(var_vals.tolist())
        ts_flat.extend(var_ts.tolist())
        deltat_flat.extend(var_dt.tolist())

    return {
        "item_id": str(chunk["chunk_id"]),
        "target":  target_flat,
        "timestamp": ts_flat,
        "past_feat_dynamic_real": deltat_flat,
        "n_obs_per_var": n_obs_per_var,
        "history": float(history),
        "value_mu_per_var":  mu.astype(np.float32).tolist(),
        "value_std_per_var": sd.astype(np.float32).tolist(),
    }


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

def convert_dataset(
    ds_name: str,
    ds_root: Path,
    out_root: Path,
    *,
    enforce_text_filter: bool,
    verbose: bool = True,
) -> dict:
    cfg = DATASET_CONFIG[ds_name]
    history, pred_window, stride, time_unit = cfg

    proc_dir = ds_root / "processed"
    if not proc_dir.is_dir():
        raise FileNotFoundError(proc_dir)
    rec_ids = sorted(d.name for d in proc_dir.iterdir() if d.is_dir())

    train_rows: list[dict] = []
    val_rows:   list[dict] = []
    test_rows:  list[dict] = []
    per_record_meta: dict[str, dict] = {}
    n_vars: int | None = None
    feat_cols_canonical: list[str] | None = None

    total_chunks = 0
    total_dropped_no_text = 0

    for rec_id in rec_ids:
        rec_dir = proc_dir / rec_id
        if not (rec_dir / "time_series.csv").is_file():
            continue
        try:
            feat_cols, tt, vals, mask, text_tt = _load_record(rec_dir, time_unit)
        except Exception as exc:
            if verbose:
                print(f"[{ds_name}] skipping {rec_id}: load failed ({exc})")
            continue

        if feat_cols_canonical is None:
            feat_cols_canonical = feat_cols
            n_vars = len(feat_cols)
        else:
            if feat_cols != feat_cols_canonical:
                raise RuntimeError(
                    f"{ds_name}: record {rec_id} has feature cols {feat_cols} "
                    f"that disagree with the first record's {feat_cols_canonical}. "
                    "Time-IMM datasets are supposed to be schema-uniform."
                )

        mu, sd = _record_norm_stats(vals)

        chunks = _build_chunks_for_record(
            rec_id=rec_id,
            tt=tt,
            vals=vals,
            mask=mask,
            text_tt=text_tt,
            history=history,
            pred_window=pred_window,
            stride=stride,
            enforce_text_filter=enforce_text_filter,
        )

        N = len(chunks)
        # parse_datasets.py:725-730 -- per-record temporal split.
        t_end = int(N * 0.6)
        v_end = int(N * 0.8)

        for i, chunk in enumerate(chunks):
            row = _chunk_to_sparse_row(chunk, mu=mu, sd=sd, history=float(history))
            if i < t_end:
                train_rows.append(row)
            elif i < v_end:
                val_rows.append(row)
            else:
                test_rows.append(row)

        per_record_meta[rec_id] = {
            "n_chunks": N,
            "n_train": t_end,
            "n_val":   max(v_end - t_end, 0),
            "n_test":  max(N - v_end, 0),
            "value_mu_per_var":  mu.astype(np.float32).tolist(),
            "value_std_per_var": sd.astype(np.float32).tolist(),
            "feat_cols": feat_cols,
        }

        total_chunks += N
        # No-text drops are surfaced via len mismatch; recompute:
        # (cheap to compute here vs threading out of _build_chunks_for_record)

        if verbose:
            print(
                f"[{ds_name}] rec={rec_id:>40s}  "
                f"V={len(feat_cols):2d}  T={len(tt):>6d}  "
                f"chunks={N:>4d}  (train/val/test = "
                f"{t_end}/{v_end-t_end}/{N-v_end})"
            )

    # Write per-split datasets
    out_ds_dir = out_root / ds_name
    out_ds_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        d = datasets.Dataset.from_list(rows) if rows else datasets.Dataset.from_dict({
            "item_id": [], "target": [], "timestamp": [],
            "past_feat_dynamic_real": [], "n_obs_per_var": [],
            "history": [], "value_mu_per_var": [], "value_std_per_var": [],
        })
        d.save_to_disk(str(out_ds_dir / split))

    # Write norm_stats.json (mirrors tpatchgnn_data style)
    norm_meta = {
        "dataset":     ds_name,
        "n_vars":      int(n_vars or 0),
        "history":     float(history),
        "pred_window": float(pred_window),
        "stride":      float(stride),
        "time_unit":   time_unit,
        "time_max":    float(history + pred_window),
        "freq":        FREQ_STR.get(time_unit, ""),
        "split_method": "sample",                 # to match the paper's main_all.py:128
        "norm_mode":   "precomputed_per_record",  # consumed by IMTSValDataset
        "feat_cols":   feat_cols_canonical or [],
        "per_record":  per_record_meta,
        # Provenance for reproducibility:
        "source": "IMM-TSF/Time-IMM (Chang et al. NeurIPS 2025 D&B)",
        "source_repo": "https://github.com/blacksnail789521/Time-IMM",
        "imm_tsf_text_filter_enforced": bool(enforce_text_filter),
    }
    with open(out_ds_dir / "norm_stats.json", "w") as f:
        json.dump(norm_meta, f, indent=2)

    summary = {
        "dataset":   ds_name,
        "n_records": len(per_record_meta),
        "n_train":   len(train_rows),
        "n_val":     len(val_rows),
        "n_test":    len(test_rows),
        "n_total":   total_chunks,
    }
    if verbose:
        print(f"[{ds_name}] DONE: {summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--imm_tsf_root",
        type=str,
        required=True,
        help="Root containing data/{dataset}/processed/{record}/time_series.csv "
             "(e.g. /home/ubuntu/imm_tsf_data, the cloned Time-IMM repo).",
    )
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output dir, will hold {dataset}/{train,val,test}/ HF datasets.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=f"Subset of datasets to convert (default: all 8). "
             f"Choices: {list(DATASET_CONFIG)}",
    )
    p.add_argument(
        "--no_text_filter",
        action="store_true",
        help="Skip the 'no text in history -> drop chunk' filter. Off by "
             "default to keep the chunk count byte-equal to IMM-TSF.",
    )
    args = p.parse_args()

    imm_root = Path(args.imm_tsf_root) / "data"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ds_list = args.datasets or list(DATASET_CONFIG)
    summaries = []
    for ds_name in ds_list:
        if ds_name not in DATASET_CONFIG:
            raise ValueError(f"unknown dataset {ds_name}; choices {list(DATASET_CONFIG)}")
        ds_root = imm_root / ds_name
        if not ds_root.is_dir():
            print(f"[skip] {ds_name}: missing dir {ds_root}")
            continue
        summary = convert_dataset(
            ds_name=ds_name,
            ds_root=ds_root,
            out_root=out_root,
            enforce_text_filter=not args.no_text_filter,
        )
        summaries.append(summary)

    print()
    print("==== conversion summary ====")
    for s in summaries:
        print(
            f"{s['dataset']:>14s}: records={s['n_records']:>3d}  "
            f"train={s['n_train']:>5d}  val={s['n_val']:>4d}  "
            f"test={s['n_test']:>4d}  total={s['n_total']:>5d}"
        )


if __name__ == "__main__":
    main()
