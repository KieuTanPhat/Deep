import argparse
import csv
import os
import sys
from copy import deepcopy


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from config import config as base_config
from train_demo import _artifact_suffix, _canonical_model_name, _slugify, train


CURATED_TRIALS = [
    {
        "name": "balanced",
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "vit_dim": 384,
        "vit_depth": 2,
        "vit_heads": 6,
        "vit_dropout": 0.20,
        "classifier_dropout": 0.35,
        "freeze_backbone_epochs": 2,
        "label_smoothing": 0.02,
    },
    {
        "name": "low_lr_stable",
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "vit_dim": 384,
        "vit_depth": 2,
        "vit_heads": 6,
        "vit_dropout": 0.20,
        "classifier_dropout": 0.30,
        "freeze_backbone_epochs": 3,
        "label_smoothing": 0.01,
    },
    {
        "name": "deeper_vit",
        "lr": 1e-5,
        "weight_decay": 2e-4,
        "vit_dim": 384,
        "vit_depth": 4,
        "vit_heads": 6,
        "vit_dropout": 0.25,
        "classifier_dropout": 0.40,
        "freeze_backbone_epochs": 2,
        "label_smoothing": 0.02,
    },
    {
        "name": "wider_tokens",
        "lr": 8e-6,
        "weight_decay": 1e-4,
        "vit_dim": 512,
        "vit_depth": 2,
        "vit_heads": 8,
        "vit_dropout": 0.25,
        "classifier_dropout": 0.35,
        "freeze_backbone_epochs": 3,
        "label_smoothing": 0.01,
    },
    {
        "name": "less_regularized",
        "lr": 2e-5,
        "weight_decay": 5e-5,
        "vit_dim": 384,
        "vit_depth": 2,
        "vit_heads": 6,
        "vit_dropout": 0.10,
        "classifier_dropout": 0.20,
        "freeze_backbone_epochs": 1,
        "label_smoothing": 0.0,
    },
]


def _require_dataset(data_root, labels_root, tasks):
    missing = []
    for split in ("train", "valid"):
        for plane in ("axial", "coronal", "sagittal"):
            path = os.path.join(data_root, split, plane)
            if not os.path.isdir(path):
                missing.append(path)
    for task in tasks:
        for split in ("train", "valid"):
            path = os.path.join(labels_root, f"{split}-{task}.csv")
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing[:20])
        raise FileNotFoundError(f"Dataset/labels are missing. First missing paths:\n{joined}")


def _read_best_auc(csv_path):
    if not os.path.exists(csv_path):
        return 0.0
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return 0.0
    values = np.atleast_1d(data["val_auc"]).astype(float)
    return float(np.max(values))


def _metrics_csv_path(model_name, cfg):
    suffix = _artifact_suffix(cfg)
    eval_folder = os.path.join("evaluation", f"{model_name}_{cfg['task']}{suffix}")
    return os.path.join(eval_folder, f"{model_name}_{cfg['task']}_metrics.csv")


def _write_summary(summary_path, rows):
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    header = [
        "rank",
        "task",
        "trial",
        "seed",
        "best_val_auc",
        "lr",
        "weight_decay",
        "vit_dim",
        "vit_depth",
        "vit_heads",
        "vit_dropout",
        "classifier_dropout",
        "freeze_backbone_epochs",
        "label_smoothing",
        "exp_name",
    ]
    rows = sorted(rows, key=lambda row: row["best_val_auc"], reverse=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Run curated EfficientNet-B0 + ViT tuning trials.")
    parser.add_argument("--tasks", type=str, default="acl", help="Comma-separated tasks.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--labels-root", type=str, default="labels")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--target-slices", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42,1337,2027.")
    parser.add_argument("--trials", type=str, default=None, help="Comma-separated curated trial names.")
    parser.add_argument("--summary", type=str, default=os.path.join("evaluation", "efficientnetb0_vit_tuning_summary.csv"))
    args = parser.parse_args()

    model_name = _canonical_model_name("efficientnetb0_vit")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if not tasks:
        raise ValueError("At least one task is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")

    selected_trials = CURATED_TRIALS
    if args.trials:
        wanted = {name.strip() for name in args.trials.split(",") if name.strip()}
        selected_trials = [trial for trial in CURATED_TRIALS if trial["name"] in wanted]
        if not selected_trials:
            raise ValueError(f"No curated trial matched: {sorted(wanted)}")

    _require_dataset(args.data_root, args.labels_root, tasks)

    rows = []
    for task in tasks:
        for seed in seeds:
            for trial in selected_trials:
                cfg = deepcopy(base_config)
                cfg.update(trial)
                cfg["task"] = task
                cfg["seed"] = seed
                cfg["optimizer"] = "adamw"
                cfg["vit_pooling"] = "cls_attention"
                cfg["exp_name"] = _slugify(f"vit_{trial['name']}_seed{seed}")

                overrides = {
                    "max_epoch": args.max_epoch,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.grad_accum_steps,
                    "target_slices": args.target_slices,
                    "image_size": args.image_size,
                    "patience": args.patience,
                }
                for key, value in overrides.items():
                    if value is not None:
                        cfg[key] = value
                if args.grad_accum_steps is not None:
                    cfg["use_gradient_accumulation"] = int(args.grad_accum_steps > 1)

                print(f"\n=== Trial {trial['name']} | task={task} | seed={seed} ===")
                train(cfg, model_name=model_name, data_root=args.data_root, labels_root=args.labels_root)

                metrics_csv = _metrics_csv_path(model_name, cfg)
                best_auc = _read_best_auc(metrics_csv)
                rows.append(
                    {
                        "task": task,
                        "trial": trial["name"],
                        "seed": seed,
                        "best_val_auc": best_auc,
                        "lr": cfg["lr"],
                        "weight_decay": cfg["weight_decay"],
                        "vit_dim": cfg["vit_dim"],
                        "vit_depth": cfg["vit_depth"],
                        "vit_heads": cfg["vit_heads"],
                        "vit_dropout": cfg["vit_dropout"],
                        "classifier_dropout": cfg["classifier_dropout"],
                        "freeze_backbone_epochs": cfg["freeze_backbone_epochs"],
                        "label_smoothing": cfg["label_smoothing"],
                        "exp_name": cfg["exp_name"],
                    }
                )
                _write_summary(args.summary, rows)
                print(f"Best AUC so far: {max(row['best_val_auc'] for row in rows):.4f}")

    _write_summary(args.summary, rows)
    best = sorted(rows, key=lambda row: row["best_val_auc"], reverse=True)[0]
    print("\nBest trial")
    print(best)
    print(f"Summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
