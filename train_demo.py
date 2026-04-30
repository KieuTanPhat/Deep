import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from config import config as base_config
from models import Densenet121, EfficientNetB0
from utils import _get_lr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _build_model(name: str):
    name = name.lower()
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
    raise ValueError(f"Unsupported model: {name}")


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    phase="train",
    scaler=None,
    use_amp=False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    y_true = []
    y_prob = []
    losses = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, leave=False)

    for batch in iterator:
        if batch is None:
            continue
        images, label = batch

        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=bool(use_amp and device != "cpu")):
                output = model(images)
                loss = criterion(output, label)
            if is_train:
                if scaler is not None and bool(use_amp and device != "cpu"):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.item())

        probas = torch.sigmoid(output).detach().cpu().view(-1).numpy().tolist()
        labels = label.detach().cpu().view(-1).numpy().tolist()

        y_prob.extend(probas)
        y_true.extend(labels)

    if len(losses) == 0:
        return 0.0, 0.5, 0.0, [], [], []

    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = metrics.accuracy_score(y_true, y_pred)
    loss_mean = float(np.mean(losses))
    return loss_mean, float(auc), float(acc), y_true, y_prob, y_pred


def _append_csv(csv_path, row, header):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _plot_curves(csv_path, out_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    epochs = np.atleast_1d(data["epoch"])
    train_loss = np.atleast_1d(data["train_loss"])
    val_loss = np.atleast_1d(data["val_loss"])
    train_auc = np.atleast_1d(data["train_auc"])
    val_auc = np.atleast_1d(data["val_auc"])
    train_acc = np.atleast_1d(data["train_acc"])
    val_acc = np.atleast_1d(data["val_acc"])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, train_auc, label="train_auc")
    plt.plot(epochs, val_auc, label="val_auc")
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, out_path):
    if len(y_true) == 0:
        return
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_roc(y_true, y_prob, out_path):
    if len(y_true) == 0:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
    except Exception:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(config: dict, model_name: str, data_root: str = "data", labels_root: str = "labels"):
    save_folder = os.path.join("weights", config["task"])
    os.makedirs(save_folder, exist_ok=True)

    eval_folder = os.path.join("evaluation", f"{model_name}_{config['task']}")
    os.makedirs(eval_folder, exist_ok=True)

    csv_path = os.path.join(eval_folder, f"{model_name}_{config['task']}_metrics.csv")
    best_model_path = os.path.join(save_folder, f"{model_name}_best_model.pth")
    last_model_path = os.path.join(save_folder, f"{model_name}_last_checkpoint.pth")

    print("Starting to Train Model...")
    train_loader, val_loader, train_wts, val_wts = load_data(
        config["task"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        target_slices=config["target_slices"],
        image_size=config["image_size"],
        data_root=data_root,
        label_root=labels_root,
    )

    print("Initializing Model...")
    model = _build_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()

    print("Initializing Loss Method...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)
    if device == "cuda":
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    print("Setup the Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.3, threshold=1e-4
    )
    use_amp = bool(device == "cuda")
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    starting_epoch = config["starting_epoch"]
    num_epochs = config["max_epoch"]
    best_val_auc = float(0)
    patience = config.get("patience", 5)
    epochs_no_improve = 0

    if os.path.exists(last_model_path):
        print(f"Found checkpoint at {last_model_path}. Loading...")
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        starting_epoch = checkpoint.get("epoch", starting_epoch) + 1
        best_val_auc = checkpoint.get("best_val_auc", best_val_auc)
        print(f"Resuming from epoch {starting_epoch} | Best AUC {best_val_auc:.4f}")

    writer = SummaryWriter(comment=f"model={model_name} lr={config['lr']} task={config['task']}")
    t_start_training = time.time()

    header = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_acc",
        "val_loss",
        "val_auc",
        "val_acc",
        "lr",
    ]

    for epoch in range(starting_epoch, num_epochs):
        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        train_loss, train_auc, train_acc, _, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_auc, val_acc, _, _, _ = _run_epoch(
            model,
            val_loader,
            val_criterion,
            optimizer=None,
            device=device,
            phase="val",
            scaler=scaler,
            use_amp=use_amp,
        )

        writer.add_scalar("Train/Avg Loss", train_loss, epoch)
        writer.add_scalar("Train/AUC_epoch", train_auc, epoch)
        writer.add_scalar("Train/Acc_epoch", train_acc, epoch)
        writer.add_scalar("Val/Avg Loss", val_loss, epoch)
        writer.add_scalar("Val/AUC_epoch", val_auc, epoch)
        writer.add_scalar("Val/Acc_epoch", val_acc, epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time
        print(
            "Epoch [{}/{}] | train loss {:.4f} | train auc {:.4f} | train acc {:.4f} | "
            "val loss {:.4f} | val auc {:.4f} | val acc {:.4f} | time {:.2f} s".format(
                epoch, num_epochs, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, delta
            )
        )
        print("-" * 30)
        writer.flush()

        _append_csv(
            csv_path,
            [epoch, train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, current_lr],
            header,
        )

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            epochs_no_improve = 0
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model for {model_name}...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "model_name": model_name,
                },
                best_model_path,
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "model_name": model_name,
            },
            last_model_path,
        )
        print(f"Checkpoint saved to {last_model_path}")

        if not improved:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    t_end_training = time.time()
    print(f"Training finished. Total time: {t_end_training - t_start_training:.2f} s")
    writer.flush()
    writer.close()

    # Load best model for final evaluation/plots
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    _, _, _, y_true, y_prob, y_pred = _run_epoch(
        model,
        val_loader,
        val_criterion,
        optimizer=None,
        device=device,
        phase="val",
        scaler=scaler,
        use_amp=use_amp,
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{model_name}_{config['task']}_curves.png"))
    _plot_confusion_matrix(y_true, y_pred, os.path.join(eval_folder, f"{model_name}_{config['task']}_confusion.png"))
    _plot_roc(y_true, y_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_roc.png"))

    print(f"Metrics saved to: {csv_path}")
    print(f"Plots saved to: {eval_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetb0",
        choices=["densenet121", "efficientnetb0"],
        help="Choose model to train",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="abnormal,acl,meniscus",
        help="Comma-separated tasks to train (default: abnormal,acl,meniscus)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Directory containing train/valid MRI folders (default: ./data).",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default="labels",
        help="Directory containing train-*.csv and valid-*.csv (default: ./labels).",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        cfg = dict(base_config)
        cfg["task"] = task
        print("Training Configuration")
        print(cfg)
        train(
            config=cfg,
            model_name=args.model,
            data_root=args.data_root,
            labels_root=args.labels_root,
        )
    print("Training Ended...")
