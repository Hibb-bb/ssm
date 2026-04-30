"""Thin wrapper to run a single (model, dataset, HP-cell) cell of IMM-TSF's
training loop on TIME-IMM data, then dump the test metrics as JSON.

Why this exists: _imm_tsf_repo's `main.py` hardcodes most params in its
`__main__` block (it's essentially a smoke-test script, not a clean CLI).
This wrapper invokes their `trainable(tunable_params, fixed_params, args)`
directly, with everything overridden from CLI flags.

Usage:
    python run_immtsf_baseline.py \
        --imm_tsf_repo /projects/bfrf/seojininus/ssm/_imm_tsf_repo \
        --time_imm_data /projects/bfrf/seojininus/ssm/_time_imm_repo/data \
        --dataset FNSPID --model tPatchGNN \
        --lr 1e-3 --patience 10 --batch_size 8 --seed 1 \
        --arch_hps '{"patch_size": 8, "K": 1}' \
        --output_dir /path/to/log/dir \
        --output_json /path/to/log/dir/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _build_args(repo_dir, time_imm_data, dataset, save_dir):
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)
    from main import get_args_from_parser

    saved_argv = sys.argv
    sys.argv = ["main.py"]
    args = get_args_from_parser()
    sys.argv = saved_argv

    args.dataset = dataset
    args.data_root = time_imm_data
    args.save = save_dir
    # CRITICAL: without this, update_args() in trainable() is a no-op and
    # args.model / args.history / args.pred_window all stay at argparse defaults.
    # See _imm_tsf_repo/main.py:936 — `if args.overwrite_args:` gates the entire
    # update_args_from_fixed_params + update_args_from_tunable_params +
    # update_args_for_dataset + update_args_for_model pipeline.
    args.overwrite_args = True
    # Pre-create the 'logs/' dir IMM-TSF's trainable() makes at line 1009 to
    # avoid a FileExistsError race when multiple parallel slurm tasks share cwd.
    os.makedirs(os.path.join(repo_dir, "logs"), exist_ok=True)
    return args


def _coerce(args, key, val):
    if not hasattr(args, key):
        return False
    cur = getattr(args, key)
    if cur is None:
        setattr(args, key, val)
        return True
    target_type = type(cur)
    try:
        setattr(args, key, target_type(val))
    except (TypeError, ValueError):
        setattr(args, key, val)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imm_tsf_repo", required=True)
    p.add_argument("--time_imm_data", required=True)
    p.add_argument("--dataset", required=True,
                   help="IMM-TSF dataset name (e.g., FNSPID, GDELT, ILINet)")
    p.add_argument("--model", required=True,
                   help="IMM-TSF model name (CRU, LatentODE, NeuralFlow, tPatchGNN)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--arch_hps",
        type=str,
        default="{}",
        help="JSON dict of architectural HP overrides, e.g. '{\"hid_dim\": 32}'",
    )
    p.add_argument("--enable_text", action="store_true", default=False)
    p.add_argument("--split_method", type=str, default="sample")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_json", required=True)
    cli = p.parse_args()

    arch_hps = json.loads(cli.arch_hps) if cli.arch_hps else {}

    os.makedirs(cli.output_dir, exist_ok=True)
    save_dir = os.path.join(cli.output_dir, "ckpts")
    os.makedirs(save_dir, exist_ok=True)

    args = _build_args(cli.imm_tsf_repo, cli.time_imm_data, cli.dataset, save_dir)
    args.seed = cli.seed
    args.epoch = cli.max_epochs

    fixed_params = {
        "dataset": cli.dataset,
        "model": cli.model,
        "batch_size": cli.batch_size,
        "enable_text": cli.enable_text,
        "use_text_embeddings": cli.enable_text,
        "split_method": cli.split_method,
        "TTF_module": "TTF_RecAvg",
        "MMF_module": "MMF_GR_Add",
        "llm_model_fusion": "GPT2",
        "llm_layers_fusion": None,
    }

    tunable_params = {
        "lr": cli.lr,
        "patience": cli.patience,
    }
    for k, v in arch_hps.items():
        tunable_params[k] = v

    print(f"[immtsf-wrap] dataset={cli.dataset} model={cli.model} "
          f"lr={cli.lr} patience={cli.patience} bs={cli.batch_size} seed={cli.seed} "
          f"arch_hps={arch_hps}")

    from utils.tools import set_seed
    set_seed(cli.seed)

    from main import trainable

    t0 = time.time()
    error = None
    try:
        test_res = trainable(tunable_params, fixed_params, args)
        ok = True
    except Exception as e:
        ok = False
        error = repr(e)
        traceback.print_exc()
        test_res = {}
    wall = time.time() - t0

    payload = {
        "ok": ok,
        "error": error,
        "wall_sec": wall,
        "dataset": cli.dataset,
        "model": cli.model,
        "lr": cli.lr,
        "patience": cli.patience,
        "batch_size": cli.batch_size,
        "seed": cli.seed,
        "arch_hps": arch_hps,
        "metrics": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in test_res.items()
            if not isinstance(v, (list, dict))
        } if isinstance(test_res, dict) else {},
    }
    with open(cli.output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[immtsf-wrap] ok={ok} wall={wall:.1f}s metrics={payload['metrics']}")
    print(f"[immtsf-wrap] result -> {cli.output_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
