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


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        return checkpoint
    return None


def _unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _load_model_state_dict(model, state_dict, strict=False):
    if not isinstance(state_dict, dict):
        raise ValueError("state_dict must be a dict.")

    target_model = _unwrap_model(model)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return target_model.load_state_dict(state_dict, strict=strict)


def _get_model_state_dict_for_save(model):
    return _unwrap_model(model).state_dict()


def _try_warmstart_from_abnormal(model, config, task, last_model_path, device):
    if os.path.exists(last_model_path):
        return
    if not bool(config.get("warmstart_from_abnormal", 1)):
        return

    warmstart_tasks = set(config.get("warmstart_tasks", ["acl", "meniscus"]))
    if task not in warmstart_tasks:
        return

    abnormal_path = str(config.get("abnormal_warmstart_path", "")).strip()
    if not abnormal_path:
        print(f"Skip warm-start for task={task}: abnormal_warmstart_path is empty.")
        return

    abnormal_path = os.path.abspath(os.path.expandvars(os.path.expanduser(abnormal_path)))
    if not os.path.exists(abnormal_path):
        print(f"Skip warm-start for task={task}: checkpoint not found at {abnormal_path}")
        return

    try:
        checkpoint = torch.load(abnormal_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            print(f"Skip warm-start for task={task}: invalid checkpoint format at {abnormal_path}")
            return

        missing, unexpected = _load_model_state_dict(model, state_dict, strict=False)
        loaded_count = len(model.state_dict()) - len(missing)
        if loaded_count == 0:
            print(
                f"Skip warm-start for task={task}: no compatible tensors in {abnormal_path} "
                f"(unexpected={len(unexpected)})"
            )
            return
        print(
            f"Warm-started task={task} from abnormal checkpoint: {abnormal_path} "
            f"(loaded={loaded_count}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
    except Exception as exc:
        print(f"Skip warm-start for task={task}: failed loading {abnormal_path} ({exc})")


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    phase="train",
    scaler=None,
    use_amp=False,
    grad_accum_steps=1,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    grad_accum_steps = max(1, int(grad_accum_steps))

    y_true = []
    y_prob = []
    losses = []
    total_batches = len(loader)

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, leave=False)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(iterator):
        if batch is None:
            continue
        images, label = batch

        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=bool(use_amp and device != "cpu")):
                output = model(images)
                loss = criterion(output, label)
            if is_train:
                loss_for_backward = loss / grad_accum_steps
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == total_batches)
                if scaler is not None and bool(use_amp and device != "cpu"):
                    scaler.scale(loss_for_backward).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        probas = torch.sigmoid(output).detach().cpu().view(-1).numpy().tolist()
        labels = label.detach().cpu().view(-1).numpy().tolist()

        y_prob.extend(probas)
        y_true.extend(labels)

    if len(losses) == 0:
        return 0.0, [], []

    loss_mean = float(np.mean(losses))
    return loss_mean, y_true, y_prob


def _compute_metrics(y_true, y_prob, threshold=0.5):
    if len(y_true) == 0:
        return {
            "auc": 0.5,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "threshold": float(threshold),
            "y_pred": [],
        }

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    return {
        "auc": float(auc),
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "y_pred": y_pred,
    }


def _find_best_threshold_by_f1(y_true, y_prob, num_thresholds=101):
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.0, 1.0, num=num_thresholds)
    for threshold in thresholds:
        score = metrics.f1_score(y_true, [1 if p >= threshold else 0 for p in y_prob], zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def _append_csv(csv_path, row, header):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _ensure_csv_header(csv_path, header):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return
    existing_header = first_line.split(",")
    if existing_header == header:
        return
    backup_path = f"{csv_path}.bak_{int(time.time())}"
    os.replace(csv_path, backup_path)
    print(f"Detected old metrics CSV format. Backed up to: {backup_path}")


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
    train_loader, val_loader, test_loader, train_wts, val_wts, test_wts = load_data(
        config["task"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        target_slices=config["target_slices"],
        image_size=config["image_size"],
        data_root=data_root,
        label_root=labels_root,
        include_test=True,
    )

    print("Initializing Model...")
    model = _build_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"DataParallel enabled on {torch.cuda.device_count()} GPUs.")
        else:
            print("DataParallel disabled: only 1 GPU is available.")
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()
        if test_wts is not None:
            test_wts = test_wts.cuda()

    _try_warmstart_from_abnormal(
        model=model,
        config=config,
        task=config["task"],
        last_model_path=last_model_path,
        device=device,
    )

    print("Initializing Loss Method...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)
    test_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=test_wts) if test_wts is not None else val_criterion
    if device == "cuda":
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()
        if test_wts is not None:
            test_criterion = test_criterion.cuda()

    print("Setup the Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.3, threshold=1e-4
    )
    use_amp = bool(device == "cuda")
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    if not bool(config.get("use_gradient_accumulation", 0)):
        grad_accum_steps = 1
    grad_accum_steps = max(1, grad_accum_steps)
    effective_batch_size = config["batch_size"] * grad_accum_steps
    print(
        f"Batch size: {config['batch_size']} | Grad accumulation steps: {grad_accum_steps} | "
        f"Effective batch size: {effective_batch_size}"
    )

    starting_epoch = config["starting_epoch"]
    num_epochs = config["max_epoch"]
    best_val_auc = float(0)
    patience = config.get("patience", 5)
    epochs_no_improve = 0

    if os.path.exists(last_model_path):
        print(f"Found checkpoint at {last_model_path}. Loading...")
        checkpoint = torch.load(last_model_path, map_location=device)
        _load_model_state_dict(model, checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_monitor") == "val_auc":
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("Skip loading old scheduler state because it was not configured for val_auc.")
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
        "train_precision",
        "train_recall",
        "train_f1",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_best_threshold",
        "val_best_f1",
        "val_best_precision",
        "val_best_recall",
        "val_best_acc",
        "lr",
    ]
    _ensure_csv_header(csv_path, header)

    for epoch in range(starting_epoch, num_epochs):
        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        train_loss, train_true, train_prob = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_true, val_prob = _run_epoch(
            model,
            val_loader,
            val_criterion,
            optimizer=None,
            device=device,
            phase="val",
            scaler=scaler,
            use_amp=use_amp,
        )

        train_metrics = _compute_metrics(train_true, train_prob, threshold=0.5)
        val_metrics = _compute_metrics(val_true, val_prob, threshold=0.5)
        val_best_threshold, _ = _find_best_threshold_by_f1(val_true, val_prob)
        val_best_metrics = _compute_metrics(val_true, val_prob, threshold=val_best_threshold)

        writer.add_scalar("Train/Avg Loss", train_loss, epoch)
        writer.add_scalar("Train/AUC_epoch", train_metrics["auc"], epoch)
        writer.add_scalar("Train/Acc_epoch", train_metrics["acc"], epoch)
        writer.add_scalar("Train/Precision_epoch", train_metrics["precision"], epoch)
        writer.add_scalar("Train/Recall_epoch", train_metrics["recall"], epoch)
        writer.add_scalar("Train/F1_epoch", train_metrics["f1"], epoch)
        writer.add_scalar("Val/Avg Loss", val_loss, epoch)
        writer.add_scalar("Val/AUC_epoch", val_metrics["auc"], epoch)
        writer.add_scalar("Val/Acc_epoch", val_metrics["acc"], epoch)
        writer.add_scalar("Val/Precision_epoch", val_metrics["precision"], epoch)
        writer.add_scalar("Val/Recall_epoch", val_metrics["recall"], epoch)
        writer.add_scalar("Val/F1_epoch", val_metrics["f1"], epoch)
        writer.add_scalar("Val/BestThreshold_F1", val_best_threshold, epoch)
        writer.add_scalar("Val/BestF1_epoch", val_best_metrics["f1"], epoch)

        scheduler.step(val_metrics["auc"])

        t_end = time.time()
        delta = t_end - epoch_start_time
        print(
            "Epoch [{}/{}] | train loss {:.4f} | train auc {:.4f} | train acc {:.4f} | "
            "train p/r/f1 {:.4f}/{:.4f}/{:.4f} | val loss {:.4f} | val auc {:.4f} | "
            "val p/r/f1@0.5 {:.4f}/{:.4f}/{:.4f} | val best_thr {:.2f} f1 {:.4f} | time {:.2f} s".format(
                epoch,
                num_epochs,
                train_loss,
                train_metrics["auc"],
                train_metrics["acc"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_loss,
                val_metrics["auc"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_best_threshold,
                val_best_metrics["f1"],
                delta,
            )
        )
        print("-" * 30)
        writer.flush()

        _append_csv(
            csv_path,
            [
                epoch,
                train_loss,
                train_metrics["auc"],
                train_metrics["acc"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_loss,
                val_metrics["auc"],
                val_metrics["acc"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_best_threshold,
                val_best_metrics["f1"],
                val_best_metrics["precision"],
                val_best_metrics["recall"],
                val_best_metrics["acc"],
                current_lr,
            ],
            header,
        )

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            epochs_no_improve = 0
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model for {model_name}...")
            torch.save(
                {
                    "model_state_dict": _get_model_state_dict_for_save(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "model_name": model_name,
                    "scheduler_monitor": "val_auc",
                },
                best_model_path,
            )

        torch.save(
            {
                "model_state_dict": _get_model_state_dict_for_save(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "model_name": model_name,
                "scheduler_monitor": "val_auc",
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
        _load_model_state_dict(model, checkpoint["model_state_dict"], strict=True)

    model.eval()
    _, val_true, val_prob = _run_epoch(
        model,
        val_loader,
        val_criterion,
        optimizer=None,
        device=device,
        phase="val",
        scaler=scaler,
        use_amp=use_amp,
    )

    best_threshold, _ = _find_best_threshold_by_f1(val_true, val_prob)
    val_final_metrics = _compute_metrics(val_true, val_prob, threshold=best_threshold)
    print(
        "Final VALID metrics | thr {:.2f} | auc {:.4f} | acc {:.4f} | precision {:.4f} | recall {:.4f} | f1 {:.4f}".format(
            best_threshold,
            val_final_metrics["auc"],
            val_final_metrics["acc"],
            val_final_metrics["precision"],
            val_final_metrics["recall"],
            val_final_metrics["f1"],
        )
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{model_name}_{config['task']}_curves.png"))
    _plot_confusion_matrix(
        val_true,
        val_final_metrics["y_pred"],
        os.path.join(eval_folder, f"{model_name}_{config['task']}_confusion.png"),
    )
    _plot_roc(val_true, val_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_roc.png"))

    test_metrics_csv = os.path.join(eval_folder, f"{model_name}_{config['task']}_test_metrics.csv")
    if test_loader is not None:
        _, test_true, test_prob = _run_epoch(
            model,
            test_loader,
            test_criterion,
            optimizer=None,
            device=device,
            phase="test",
            scaler=scaler,
            use_amp=use_amp,
        )
        test_metrics = _compute_metrics(test_true, test_prob, threshold=best_threshold)
        print(
            "Final TEST metrics  | thr {:.2f} | auc {:.4f} | acc {:.4f} | precision {:.4f} | recall {:.4f} | f1 {:.4f}".format(
                best_threshold,
                test_metrics["auc"],
                test_metrics["acc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
            )
        )
        _append_csv(
            test_metrics_csv,
            [
                config["task"],
                best_threshold,
                test_metrics["auc"],
                test_metrics["acc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
            ],
            ["task", "threshold", "auc", "acc", "precision", "recall", "f1"],
        )
        _plot_confusion_matrix(
            test_true,
            test_metrics["y_pred"],
            os.path.join(eval_folder, f"{model_name}_{config['task']}_test_confusion.png"),
        )
        _plot_roc(test_true, test_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_test_roc.png"))
    else:
        print("Skip TEST evaluation: test split not found.")

    print(f"Metrics saved to: {csv_path}")
    if test_loader is not None:
        print(f"Test metrics saved to: {test_metrics_csv}")
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
        help="Directory containing train/valid/test MRI folders (default: ./data).",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default="labels",
        help="Directory containing train-*.csv, valid-*.csv and test-*.csv (default: ./labels).",
    )
    parser.add_argument(
        "--abnormal-pth",
        type=str,
        default=None,
        help="Absolute/relative path to abnormal checkpoint used to warm-start acl/meniscus.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        cfg = dict(base_config)
        cfg["task"] = task
        if args.abnormal_pth:
            cfg["abnormal_warmstart_path"] = args.abnormal_pth
        print("Training Configuration")
        print(cfg)
        train(
            config=cfg,
            model_name=args.model,
            data_root=args.data_root,
            labels_root=args.labels_root,
        )
    print("Training Ended...")
