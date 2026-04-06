"""
Convert T-PatchGNN benchmark datasets (PhysioNet, Activity, USHCN) into
HuggingFace Datasets compatible with the MOIRAI training pipeline.

Sparse multivariate format: each record stores ALL variates' observed values
concatenated into flat 1D arrays, with `n_obs_per_var` as offsets. At training
time, `UnpackSparseMultivariate` splits them back into per-variate arrays.

Usage:
    python convert_tpatchgnn_data.py --dataset physionet
    python convert_tpatchgnn_data.py --dataset activity
    python convert_tpatchgnn_data.py --dataset ushcn
"""

import argparse
import json
import os
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import Features, Sequence, Value
from sklearn import model_selection

TPATCHGNN_ROOT = Path('/projects/b1094/StarEmbed/t-PatchGNN')


def load_physionet(tpatchgnn_root: Path, device='cpu'):
    """Load PhysioNet directly from processed .pt files (avoids torchvision dependency)."""
    processed = tpatchgnn_root / 'data' / 'physionet' / 'processed'
    data = []
    for prefix in ('set-a', 'set-b', 'set-c'):
        candidates = list(processed.glob(f'{prefix}_*.pt'))
        if not candidates:
            raise FileNotFoundError(f'No processed file found for {prefix} in {processed}')
        data.extend(torch.load(candidates[0], map_location='cpu'))
    return data, {'history': 24.0, 'n_vars': 41, 'time_unit': 'hours'}


def load_activity(tpatchgnn_root: Path, device='cpu'):
    """Load Activity directly from processed .pt file."""
    processed = tpatchgnn_root / 'data' / 'activity' / 'processed'
    data = torch.load(processed / 'data.pt', map_location='cpu')
    return list(data), {
        'history': 3000.0,
        'pred_window': 1000.0,
        'n_vars': 12,
        'time_unit': 'ms',
        'needs_chunking': True,
    }


def load_ushcn(tpatchgnn_root: Path, device='cpu'):
    """Load USHCN directly from processed .pt file."""
    processed = tpatchgnn_root / 'data' / 'ushcn' / 'processed'
    data = torch.load(processed / 'ushcn.pt', map_location='cpu')
    return list(data), {
        'history': 24.0,
        'n_months': 48,
        'pred_window': 1.0,
        'n_vars': 5,
        'time_unit': 'months',
        'needs_chunking': True,
    }


def chunk_activity(data, history, pred_window, device='cpu'):
    """Replicate Activity_time_chunk from T-PatchGNN."""
    chunk_data = []
    for record_id, tt, vals, mask in data:
        t_max = int(tt.max())
        for st in range(0, t_max - int(history), int(pred_window)):
            et = st + int(history) + int(pred_window)
            if et >= t_max:
                idx = torch.where((tt >= st) & (tt <= et))[0]
            else:
                idx = torch.where((tt >= st) & (tt < et))[0]
            new_id = f'{record_id}_{st // int(pred_window)}'
            chunk_data.append((new_id, tt[idx] - st, vals[idx], mask[idx]))
    return chunk_data


def chunk_ushcn(data, n_months, history, pred_window, device='cpu'):
    """Replicate USHCN_time_chunk from T-PatchGNN."""
    chunk_data = []
    for record_id, tt, vals, mask in data:
        for st in range(0, n_months - int(history) - int(pred_window) + 1, int(pred_window)):
            et = st + int(history) + int(pred_window)
            if et == n_months:
                indices = torch.where((tt >= st) & (tt <= et))[0]
            else:
                indices = torch.where((tt >= st) & (tt < et))[0]
            t_bias = float(st)
            chunk_data.append((record_id, tt[indices] - t_bias, vals[indices], mask[indices]))
    return chunk_data


def get_data_min_max(records, device='cpu'):
    """Replicate get_data_min_max from T-PatchGNN/lib/physionet.py."""
    inf = float('inf')
    data_min, data_max, time_max = None, None, -inf

    for item in records:
        tt = item[1]
        vals = item[2]
        mask = item[3]

        if tt.numel() == 0:
            continue

        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing = vals[:, i][mask[:, i] == 1]
            if len(non_missing) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(non_missing.min().item())
                batch_max.append(non_missing.max().item())

        batch_min = np.array(batch_min, dtype=np.float32)
        batch_max = np.array(batch_max, dtype=np.float32)

        if data_min is None:
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = np.minimum(data_min, batch_min)
            data_max = np.maximum(data_max, batch_max)

        time_max = max(time_max, tt.max().item())

    return data_min, data_max, time_max


def normalize_values(vals, mask, data_min, data_max):

    scale = data_max - data_min
    scale = scale + (scale == 0) * 1e-08
    normed = (vals - data_min) / scale
    normed[mask == 0] = 0.0
    return normed


def records_to_hf_dataset(
    records,
    n_vars,
    history,
    normalize_vals,
    data_min=None,
    data_max=None,
):
    """
    Convert a list of IMTS records into a HuggingFace Dataset.

    Sparse multivariate format: all variates' observed values are concatenated
    into flat arrays. `n_obs_per_var` stores the count per variate so the
    `UnpackSparseMultivariate` transform can reconstruct per-variate arrays.
    """
    rows = []
    for item in records:
        record_id = item[0]
        tt = item[1].cpu().numpy().astype(np.float32)
        vals = item[2].cpu().numpy().astype(np.float32)
        mask = item[3].cpu().numpy().astype(np.float32)

        if normalize_vals and data_min is not None:
            vals = normalize_values(vals, mask, data_min, data_max)

        all_target = []
        all_timestamp = []
        all_delta_t = []
        obs_per_var = []

        for d in range(n_vars):
            obs_idx = np.where(mask[:, d] > 0.5)[0]
            n_obs = len(obs_idx)
            obs_per_var.append(n_obs)

            if n_obs == 0:
                continue

            ts_d = tt[obs_idx]
            val_d = vals[obs_idx, d]

            delta_t = np.zeros_like(ts_d)
            if n_obs > 1:
                delta_t[1:] = np.diff(ts_d)

            all_target.extend(val_d.tolist())
            all_timestamp.extend(ts_d.tolist())
            all_delta_t.extend(delta_t.tolist())

        if len(all_target) == 0:
            continue

        rows.append(
            {
                'item_id': str(record_id),
                'target': all_target,
                'timestamp': all_timestamp,
                'past_feat_dynamic_real': all_delta_t,
                'n_obs_per_var': obs_per_var,
                'history': history,
            }
        )

    features = Features(
        {
            'item_id': Value('string'),
            'target': Sequence(Value('float32')),
            'timestamp': Sequence(Value('float32')),
            'past_feat_dynamic_real': Sequence(Value('float32')),
            'n_obs_per_var': Sequence(Value('int32')),
            'history': Value('float32'),
        }
    )

    return datasets.Dataset.from_dict(
        {k: [r[k] for r in rows] for k in rows[0].keys()},
        features=features,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['physionet', 'activity', 'ushcn'],
    )
    parser.add_argument(
        '--tpatchgnn_root',
        type=str,
        default=str(TPATCHGNN_ROOT),
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    tpatchgnn_root = Path(args.tpatchgnn_root)
    if args.output_root is None:
        output_root = Path(__file__).resolve().parent.parent.parent / 'tpatchgnn_data'
    else:
        output_root = Path(args.output_root)

    dataset_name = args.dataset
    print(f'[INFO] Converting dataset: {dataset_name}')
    print(f'[INFO] T-PatchGNN root: {tpatchgnn_root}')
    print(f'[INFO] Output root: {output_root}')

    if dataset_name == 'physionet':
        total_dataset, meta = load_physionet(tpatchgnn_root)
    elif dataset_name == 'activity':
        total_dataset, meta = load_activity(tpatchgnn_root)
    elif dataset_name == 'ushcn':
        total_dataset, meta = load_ushcn(tpatchgnn_root)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    history = meta['history']
    n_vars = meta['n_vars']

    seen_data, test_data = model_selection.train_test_split(
        total_dataset, train_size=0.8, random_state=42, shuffle=True,
    )
    train_data, val_data = model_selection.train_test_split(
        seen_data, train_size=0.75, random_state=42, shuffle=False,
    )

    print(f'[INFO] Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')

    if dataset_name == 'activity':
        pred_window = meta['pred_window']
        train_data = chunk_activity(train_data, history, pred_window)
        val_data = chunk_activity(val_data, history, pred_window)
        test_data = chunk_activity(test_data, history, pred_window)
        print(f'[INFO] After chunking: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')
    elif dataset_name == 'ushcn':
        n_months = meta['n_months']
        pred_window = meta['pred_window']
        train_data = chunk_ushcn(train_data, n_months, history, pred_window)
        val_data = chunk_ushcn(val_data, n_months, history, pred_window)
        test_data = chunk_ushcn(test_data, n_months, history, pred_window)
        print(f'[INFO] After chunking: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')

    seen_all = train_data + val_data
    data_min, data_max, time_max = get_data_min_max(seen_all)

    normalize_vals = dataset_name != 'ushcn'

    if dataset_name == 'activity':
        time_max = history + meta['pred_window']

    print(f'[INFO] data_min: {data_min}')
    print(f'[INFO] data_max: {data_max}')
    print(f'[INFO] time_max: {time_max}')
    print(f'[INFO] normalize_vals: {normalize_vals}')

    out_dir = output_root / dataset_name
    os.makedirs(out_dir, exist_ok=True)

    norm_stats = {
        'data_min': data_min.tolist(),
        'data_max': data_max.tolist(),
        'time_max': float(time_max),
        'normalize_vals': normalize_vals,
        'history': float(history),
        'n_vars': n_vars,
        'dataset': dataset_name,
    }
    with open(out_dir / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f'[INFO] Saved normalization stats to {out_dir / "norm_stats.json"}')

    for split_name, split_data in (('train', train_data), ('val', val_data), ('test', test_data)):
        print(f'[INFO] Converting {split_name} split ({len(split_data)} records)...')
        hf_ds = records_to_hf_dataset(
            split_data,
            n_vars=n_vars,
            history=history,
            normalize_vals=normalize_vals,
            data_min=data_min,
            data_max=data_max,
        )

        total_obs = sum(sum(r) for r in hf_ds['n_obs_per_var'])
        avg_obs = total_obs / len(hf_ds)
        print(f'[INFO]   -> {len(hf_ds)} records, avg {avg_obs:.0f} total obs/record')

        save_path = out_dir / split_name
        hf_ds.save_to_disk(str(save_path))
        print(f'[INFO]   -> Saved to {save_path}')

    print('[INFO] Done!')


if __name__ == '__main__':
    main()
