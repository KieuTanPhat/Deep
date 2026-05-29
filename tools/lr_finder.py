
import argparse
import math
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_data
from config import config as base_config
from models import Densenet121, EfficientNetB0, EfficientNetB0ViT


def _build_model(name: str):
    name = name.lower()
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
    if name in {"efficientnetb0_vit", "efficientnetb0-vit", "efficientnetb0vit"}:
        return EfficientNetB0ViT()
    raise ValueError(f"Unsupported model: {name}")


def lr_finder(model, loader, criterion, device, lr_start, lr_end, num_iters):
    model.train()
    lrs = np.logspace(math.log10(lr_start), math.log10(lr_end), num_iters)
    losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    iterator = iter(loader)
    for i in range(num_iters):
        try:
            images, label = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, label = next(iterator)

        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        lr = lrs[i]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 10 == 0:
            print(f"iter {i}/{num_iters} | lr={lr:.2e} | loss={loss.item():.4f}")

    return lrs, np.array(losses)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="efficientnetb0_vit",
        choices=["densenet121", "efficientnetb0", "efficientnetb0_vit", "efficientnetb0-vit", "efficientnetb0vit"],
    )
    ap.add_argument("--lr-start", type=float, default=1e-6)
    ap.add_argument("--lr-end", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--task", type=str, default=None)
    args = ap.parse_args()

    cfg = dict(base_config)
    if args.task is not None:
        cfg["task"] = args.task

    train_loader, _, train_wts, _ = load_data(
        cfg["task"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        target_slices=cfg["target_slices"],
        image_size=cfg["image_size"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args.model)
    if device == "cuda":
        model = model.cuda()
        train_wts = train_wts.cuda()

    criterion = nn.BCEWithLogitsLoss(pos_weight=train_wts)
    if device == "cuda":
        criterion = criterion.cuda()

    lrs, losses = lr_finder(
        model,
        train_loader,
        criterion,
        device,
        args.lr_start,
        args.lr_end,
        args.iters,
    )

    out_dir = "evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lr_finder_{args.model}_{cfg['task']}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved LR finder plot to: {out_path}")


if __name__ == "__main__":
    main()

