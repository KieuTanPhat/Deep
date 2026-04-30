import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_image_standardize(img: torch.Tensor) -> torch.Tensor:
    return (img - img.mean()) / (img.std() + 1e-6)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Lambda(per_image_standardize),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(per_image_standardize),
        ]
    )
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.efficientnet_b0(weights=None)
    except Exception:
        model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == target).float().mean().item())


def collect_images(data_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


class RotationDataset(Dataset):
    """Self-supervised dataset for 1-class data: predict image rotation (0/90/180/270)."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) * 4

    def __getitem__(self, index):
        base_idx = index // 4
        rot_label = index % 4
        path = self.image_paths[base_idx]
        img = Image.open(path).convert("RGB")
        img = img.rotate(rot_label * 90, expand=True)
        if self.transform is not None:
            img = self.transform(img)
        return img, rot_label


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    accs = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                loss_for_step = loss / grad_accum_steps

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss_for_step).backward()
                else:
                    loss_for_step.backward()

                do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
                if do_step:
                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, labels))

    if len(losses) == 0:
        return 0.0, 0.0
    return float(np.mean(losses)), float(np.mean(accs))


def _three_plane_state_from_backbone(backbone_state: dict) -> dict:
    state = {}
    for key, value in backbone_state.items():
        state[f"axial.{key}"] = value
        state[f"coronal.{key}"] = value
        state[f"sagittal.{key}"] = value

    fc = nn.Linear(3 * 1280, 1).state_dict()
    state["fc.weight"] = fc["weight"]
    state["fc.bias"] = fc["bias"]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_pretrained")
    parser.add_argument("--output-dir", type=str, default="model_pretrained")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf, val_tf = build_transforms(args.image_size)
    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    image_folder_ds = datasets.ImageFolder(root=args.data_dir, transform=train_tf)
    num_classes = len(image_folder_ds.classes)

    if num_classes >= 2:
        print(f"Pretrain mode: supervised ({num_classes} classes)")

        full_ds_train_tf = datasets.ImageFolder(root=args.data_dir, transform=train_tf)
        full_ds_val_tf = datasets.ImageFolder(root=args.data_dir, transform=val_tf)
        targets = np.array(full_ds_train_tf.targets)
        indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=targets,
        )

        train_set = Subset(full_ds_train_tf, train_idx.tolist())
        val_set = Subset(full_ds_val_tf, val_idx.tolist())
        train_classes = full_ds_train_tf.classes
    else:
        print("Pretrain mode: self-supervised rotation (detected <2 classes)")
        image_paths = collect_images(args.data_dir)
        if len(image_paths) == 0:
            raise RuntimeError(f"No images found under: {args.data_dir}")

        indices = np.arange(len(image_paths))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.seed,
            shuffle=True,
        )
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]

        train_set = RotationDataset(train_paths, transform=train_tf)
        val_set = RotationDataset(val_paths, transform=val_tf)
        num_classes = 4
        train_classes = ["rot0", "rot90", "rot180", "rot270"]

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_best.pth")
    last_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_last.pth")
    features_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_features.pth")

    print(f"Device: {device}")
    print(f"Train classes: {train_classes}")
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    print("Training EfficientNetB0 backbone (weights=None)...")

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=max(1, args.grad_accum_steps),
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=1,
        )
        scheduler.step()

        print(
            "Epoch [{}/{}] | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f} | lr {:.6f}".format(
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer.param_groups[0]["lr"],
            )
        )

        backbone_state = model.features.state_dict()
        three_plane_state = _three_plane_state_from_backbone(backbone_state)
        state = {
            "epoch": epoch,
            "model_state_dict": three_plane_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "pretrain_classes": train_classes,
            "source": "pretrained.py",
        }
        torch.save(state, last_ckpt_path)
        torch.save(backbone_state, features_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(state, best_ckpt_path)
            print(f"  New best val acc: {best_val_acc:.4f} -> saved")

    elapsed = time.time() - started
    print(f"Done. Total time: {elapsed:.2f}s")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Backbone features checkpoint: {features_ckpt_path}")
    print("Checkpoint format is compatible with train_demo.py (axial/coronal/sagittal keys).")


if __name__ == "__main__":
    main()
