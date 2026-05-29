import argparse
import csv
import os
import random
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from config import config as base_config
from models import Densenet121, EfficientNetB0, EfficientNetB0_ViT, EfficientNetViT
from utils import _get_lr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MODEL_ALIASES = {
    "densenet121": "densenet121",
    "efficientnetb0": "efficientnetb0",
    "efficientnetb0_vit": "efficientnetb0_vit",
    "efficientnetb0-vit": "efficientnetb0_vit",
    "efficientnetb0vit": "efficientnetb0_vit",
    "effectionnetb0_vit": "efficientnetb0_vit",
    "effectionnectb0_vit": "efficientnetb0_vit",
    "efficientnetvit": "efficientnetvit",
}


def _canonical_model_name(name: str):
    key = name.lower().replace(" ", "").replace("__", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_ALIASES[key]


def _build_model(name: str, config: dict):
    name = name.lower()
    name = _canonical_model_name(name)
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
    if name == "efficientnetb0_vit":
        return EfficientNetB0_ViT(
            vit_dim=int(config.get("vit_dim", 256)),
            vit_depth=int(config.get("vit_depth", 2)),
            vit_heads=int(config.get("vit_heads", 4)),
            vit_mlp_ratio=float(config.get("vit_mlp_ratio", 2.0)),
            vit_dropout=float(config.get("vit_dropout", 0.2)),
            classifier_dropout=float(config.get("classifier_dropout", 0.35)),
            max_slices=max(64, int(config.get("target_slices", 24))),
            pooling=str(config.get("vit_pooling", "cls_attention")),
        )
    if name == "efficientnetvit":
        return EfficientNetViT(max_slices=max(64, int(config.get("target_slices", 24))))
    raise ValueError(f"Unsupported model: {name}")


def _slugify(value: str):
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _artifact_suffix(config: dict):
    exp_name = _slugify(config.get("exp_name", ""))
    if exp_name in {"", "test", "default"}:
        return ""
    return f"_{exp_name}"


def _set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if not strict:
        # Loại bỏ key có shape không khớp để tránh RuntimeError khi warm-start
        # giữa các model khác kiến trúc (vd: EfficientNetB0 → EfficientNetB0_ViT).
        model_sd = target_model.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items()
            if k not in model_sd or v.shape == model_sd[k].shape
        }

    return target_model.load_state_dict(state_dict, strict=strict)


def _get_model_state_dict_for_save(model):
    return _unwrap_model(model).state_dict()


def _set_backbone_frozen(model, freeze):
    model = _unwrap_model(model)
    if hasattr(model, "freeze_feature_extractors"):
        model.freeze_feature_extractors(freeze=freeze)
        state = "frozen" if freeze else "trainable"
        print(f"Backbone feature extractors are now {state}.")


def _build_optimizer(model, config):
    model_for_groups = _unwrap_model(model)
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])

    if hasattr(model_for_groups, "backbone_parameters") and hasattr(model_for_groups, "head_parameters"):
        backbone_params = [p for p in model_for_groups.backbone_parameters() if p.requires_grad]
        head_params = [p for p in model_for_groups.head_parameters() if p.requires_grad]
        params = [
            {
                "params": backbone_params,
                "lr": lr * float(config.get("backbone_lr_mult", 0.3)),
            },
            {"params": head_params, "lr": lr},
        ]
    else:
        params = model.parameters()

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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
    max_grad_norm=0.0,
    label_smoothing=0.0,
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
                loss_label = label
                if is_train and float(label_smoothing) > 0:
                    smoothing = float(label_smoothing)
                    loss_label = label * (1.0 - smoothing) + 0.5 * smoothing
                loss = criterion(output, loss_label)
            if is_train:
                loss_for_backward = loss / grad_accum_steps
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == total_batches)
                if scaler is not None and bool(use_amp and device != "cpu"):
                    scaler.scale(loss_for_backward).backward()
                    if should_step:
                        if float(max_grad_norm) > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if should_step:
                        if float(max_grad_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
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
    model_name = _canonical_model_name(model_name)
    _set_seed(config.get("seed", None))

    suffix = _artifact_suffix(config)
    exp_name = _slugify(config.get("exp_name", ""))
    if suffix:
        save_folder = os.path.join("weights", config["task"], exp_name)
    else:
        save_folder = os.path.join("weights", config["task"])
    os.makedirs(save_folder, exist_ok=True)

    eval_folder = os.path.join("evaluation", f"{model_name}_{config['task']}{suffix}")
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
    model = _build_model(model_name, config)
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
    optimizer = _build_optimizer(model, config)
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
    freeze_backbone_epochs = int(config.get("freeze_backbone_epochs", 0))
    backbone_is_frozen = None

    for epoch in range(starting_epoch, num_epochs):
        should_freeze_backbone = bool(freeze_backbone_epochs > 0 and epoch < freeze_backbone_epochs)
        if should_freeze_backbone != backbone_is_frozen:
            _set_backbone_frozen(model, should_freeze_backbone)
            backbone_is_frozen = should_freeze_backbone

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
            max_grad_norm=float(config.get("max_grad_norm", 0.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
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
        default="efficientnetb0_vit",
        help="Choose model to train: densenet121, efficientnetb0 or efficientnetb0_vit.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch_size. This is the micro-batch size kept in GPU memory.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override config gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--target-slices",
        type=int,
        default=None,
        help="Override config target_slices to reduce/increase per-volume memory.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override config image_size.",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Optional run name for separate artifacts.")
    parser.add_argument("--max-epoch", type=int, default=None, help="Override config max_epoch.")
    parser.add_argument("--lr", type=float, default=None, help="Override config learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override config weight_decay.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"], help="Optimizer.")
    parser.add_argument("--backbone-lr-mult", type=float, default=None, help="LR multiplier for CNN backbones.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping max norm.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Binary label smoothing amount.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None, help="Warm-up epochs with frozen CNN backbones.")
    parser.add_argument("--vit-dim", type=int, default=None, help="Transformer hidden dimension.")
    parser.add_argument("--vit-depth", type=int, default=None, help="Transformer encoder layers.")
    parser.add_argument("--vit-heads", type=int, default=None, help="Transformer attention heads.")
    parser.add_argument("--vit-mlp-ratio", type=float, default=None, help="Transformer MLP expansion ratio.")
    parser.add_argument("--vit-dropout", type=float, default=None, help="Transformer dropout.")
    parser.add_argument("--classifier-dropout", type=float, default=None, help="Classifier dropout.")
    parser.add_argument(
        "--vit-pooling",
        type=str,
        default=None,
        choices=["cls", "mean", "max", "attention", "cls_attention"],
        help="Pooling mode for transformer tokens.",
    )
    args = parser.parse_args()

    model_name = _canonical_model_name(args.model)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        cfg = dict(base_config)
        cfg["task"] = task
        if args.abnormal_pth:
            cfg["abnormal_warmstart_path"] = args.abnormal_pth
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        if args.grad_accum_steps is not None:
            cfg["gradient_accumulation_steps"] = args.grad_accum_steps
            cfg["use_gradient_accumulation"] = int(args.grad_accum_steps > 1)
        if args.target_slices is not None:
            cfg["target_slices"] = args.target_slices
        if args.image_size is not None:
            cfg["image_size"] = args.image_size
        override_map = {
            "exp_name": args.exp_name,
            "max_epoch": args.max_epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "backbone_lr_mult": args.backbone_lr_mult,
            "max_grad_norm": args.max_grad_norm,
            "label_smoothing": args.label_smoothing,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "vit_dim": args.vit_dim,
            "vit_depth": args.vit_depth,
            "vit_heads": args.vit_heads,
            "vit_mlp_ratio": args.vit_mlp_ratio,
            "vit_dropout": args.vit_dropout,
            "classifier_dropout": args.classifier_dropout,
            "vit_pooling": args.vit_pooling,
        }
        for key, value in override_map.items():
            if value is not None:
                cfg[key] = value
        print("Training Configuration")
        print(cfg)
        train(
            config=cfg,
            model_name=model_name,
            data_root=args.data_root,
            labels_root=args.labels_root,
        )
    print("Training Ended...")
