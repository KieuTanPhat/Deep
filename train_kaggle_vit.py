
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
try:
    from torch.amp import GradScaler, autocast
    def _make_scaler(enabled): return GradScaler('cuda', enabled=enabled)
    def _autocast(enabled): return autocast('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def _make_scaler(enabled): return GradScaler(enabled=enabled)
    def _autocast(enabled): return autocast(enabled=enabled)
from sklearn import metrics
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ─── Đường dẫn Kaggle (thay đổi nếu cần) ────────────────────────────────────
# Dataset: Nhom5_DeepLearning_Dataset
# Slug Kaggle tự động: nhom5-deeplearning-dataset
KAGGLE_DATA_ROOT   = "/kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/data"
KAGGLE_LABELS_ROOT = "/kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/labels"
KAGGLE_OUTPUT_ROOT = "/kaggle/working"
# ─────────────────────────────────────────────────────────────────────────────

from dataset import load_data
from config import config as base_config
from models import EfficientNetViT
from utils import _get_lr


MODEL_NAME = "efficientnetvit"

# Config mặc định cho Kaggle (tối ưu VRAM 16GB)
KAGGLE_CONFIG_OVERRIDES = {
    "batch_size": 4,
    "target_slices": 24,
    "image_size": 224,
    "num_workers": 2,
    "use_gradient_accumulation": 1,
    "gradient_accumulation_steps": 8,  # Effective batch = 4 * 8 = 32
    "max_epoch": 50,
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "patience": 5,
}


# ─── Helpers dùng chung với train_demo.py ────────────────────────────────────

def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        return checkpoint
    return None


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_model_state_dict(model, state_dict, strict=False):
    target = _unwrap_model(model)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return target.load_state_dict(state_dict, strict=strict)


def _get_model_state_dict_for_save(model):
    return _unwrap_model(model).state_dict()


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu",
               phase="train", scaler=None, use_amp=False, grad_accum_steps=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    grad_accum_steps = max(1, int(grad_accum_steps))

    y_true, y_prob, losses = [], [], []
    total = len(loader)
    iterator = tqdm(loader, desc=phase, leave=False) if tqdm else loader

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for idx, batch in enumerate(iterator):
        if batch is None:
            continue
        images, label = batch
        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        with torch.set_grad_enabled(is_train):
            with _autocast(enabled=bool(use_amp and device != "cpu")):
                output = model(images)
                loss = criterion(output, label)

            if is_train:
                loss_scaled = loss / grad_accum_steps
                should_step = ((idx + 1) % grad_accum_steps == 0) or ((idx + 1) == total)
                if scaler and use_amp and device != "cpu":
                    scaler.scale(loss_scaled).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_scaled.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        y_prob.extend(torch.sigmoid(output).detach().cpu().view(-1).tolist())
        y_true.extend(label.detach().cpu().view(-1).tolist())

    return float(np.mean(losses)) if losses else 0.0, y_true, y_prob


def _compute_metrics(y_true, y_prob, threshold=0.5):
    if not y_true:
        return {"auc": 0.5, "acc": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "threshold": threshold, "y_pred": []}
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


def _find_best_threshold(y_true, y_prob, steps=101):
    if not y_true or len(set(y_true)) < 2:
        return 0.5, 0.0
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0, 1, steps):
        f1 = metrics.f1_score(y_true, [1 if p >= thr else 0 for p in y_prob], zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def _append_csv(path, row, header):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _plot_curves(csv_path, out_path):
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        epochs = np.atleast_1d(data["epoch"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, np.atleast_1d(data["train_loss"]), label="train")
        axes[0].plot(epochs, np.atleast_1d(data["val_loss"]), label="val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, np.atleast_1d(data["train_auc"]), label="train")
        axes[1].plot(epochs, np.atleast_1d(data["val_auc"]), label="val")
        axes[1].set_title("AUC"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception as e:
        print(f"Bỏ qua vẽ đồ thị: {e}")


def _plot_confusion_matrix(y_true, y_pred, out_path):
    if not y_true:
        return
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(out_path); plt.close()


def _plot_roc(y_true, y_prob, out_path):
    if not y_true:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception as e:
        print(f"Bỏ qua vẽ ROC: {e}")


# ─── Hàm train chính ─────────────────────────────────────────────────────────

def train(config: dict, data_root: str, labels_root: str, freeze_epochs: int = 0):
    task = config["task"]
    save_folder = os.path.join(KAGGLE_OUTPUT_ROOT, "weights", task)
    eval_folder = os.path.join(KAGGLE_OUTPUT_ROOT, "evaluation", f"{MODEL_NAME}_{task}")
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    csv_path       = os.path.join(eval_folder, f"{MODEL_NAME}_{task}_metrics.csv")
    best_model_path = os.path.join(save_folder, f"{MODEL_NAME}_best_model.pth")
    last_model_path = os.path.join(save_folder, f"{MODEL_NAME}_last_checkpoint.pth")

    print(f"\n{'='*60}")
    print(f"  Task: {task.upper()}  |  Model: EfficientNetB0 + ViT")
    print(f"  freeze_backbone cho {freeze_epochs} epoch đầu")
    print(f"{'='*60}")

    print("Đang load dữ liệu...")
    train_loader, val_loader, test_loader, train_wts, val_wts, test_wts = load_data(
        task,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        target_slices=config["target_slices"],
        image_size=config["image_size"],
        data_root=data_root,
        label_root=labels_root,
        include_test=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Khởi tạo model — freeze backbone trong freeze_epochs đầu để tiết kiệm VRAM
    print("Khởi tạo EfficientNetViT...")
    model = EfficientNetViT(
        nhead=8,
        num_layers=4,
        dropout=0.1,
        freeze_backbone=(freeze_epochs > 0),
        max_slices=config["target_slices"] + 8,  # Buffer nhỏ
    )

    if device == "cuda":
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"DataParallel: {torch.cuda.device_count()} GPU")
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()
        if test_wts is not None:
            test_wts = test_wts.cuda()

    criterion     = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)
    test_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=test_wts) if test_wts is not None else val_criterion
    if device == "cuda":
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.3, threshold=1e-4
    )
    use_amp = (device == "cuda")
    scaler  = _make_scaler(enabled=use_amp)
    grad_accum = int(config.get("gradient_accumulation_steps", 1)) if config.get("use_gradient_accumulation") else 1
    grad_accum = max(1, grad_accum)
    print(f"AMP: {use_amp} | batch={config['batch_size']} | grad_accum={grad_accum} | effective_bs={config['batch_size']*grad_accum}")

    # Resume checkpoint nếu có
    starting_epoch = config["starting_epoch"]
    best_val_auc   = 0.0
    patience_cnt   = 0

    if os.path.exists(last_model_path):
        print(f"Tìm thấy checkpoint: {last_model_path}")
        ckpt = torch.load(last_model_path, map_location=device)
        _load_model_state_dict(model, ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_monitor") == "val_auc":
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        starting_epoch = ckpt.get("epoch", starting_epoch) + 1
        best_val_auc   = ckpt.get("best_val_auc", 0.0)
        print(f"Resume từ epoch {starting_epoch} | Best AUC: {best_val_auc:.4f}")

    writer = None
    if _HAS_TB:
        writer = SummaryWriter(
            log_dir=os.path.join(KAGGLE_OUTPUT_ROOT, "runs", f"{MODEL_NAME}_{task}"),
            comment=f"lr={config['lr']} task={task}",
        )

    header = [
        "epoch", "train_loss", "train_auc", "train_acc",
        "train_precision", "train_recall", "train_f1",
        "val_loss", "val_auc", "val_acc",
        "val_precision", "val_recall", "val_f1",
        "val_best_thr", "val_best_f1", "lr",
    ]

    t0 = time.time()
    for epoch in range(starting_epoch, config["max_epoch"]):
        current_lr = _get_lr(optimizer)

        # Unfreeze backbone sau freeze_epochs epoch đầu
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"\nEpoch {epoch}: Unfreeze backbone + giảm LR xuống {config['lr']/10:.2e}")
            _unwrap_model(model).unfreeze_backbones()
            for pg in optimizer.param_groups:
                pg["lr"] = config["lr"] / 10
            # Thêm backbone params vào optimizer
            backbone_params = []
            for net in [
                _unwrap_model(model).axial_backbone,
                _unwrap_model(model).coronal_backbone,
                _unwrap_model(model).sagittal_backbone,
            ]:
                backbone_params.extend(net.parameters())
            optimizer.add_param_group({"params": backbone_params, "lr": config["lr"] / 20})

        ep_start = time.time()
        train_loss, tr_true, tr_prob = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer,
            device=device, phase="train", scaler=scaler,
            use_amp=use_amp, grad_accum_steps=grad_accum,
        )
        val_loss, val_true, val_prob = _run_epoch(
            model, val_loader, val_criterion,
            device=device, phase="val", scaler=scaler, use_amp=use_amp,
        )

        tr_m    = _compute_metrics(tr_true, tr_prob)
        val_m   = _compute_metrics(val_true, val_prob)
        best_thr, _ = _find_best_threshold(val_true, val_prob)
        val_best_m  = _compute_metrics(val_true, val_prob, threshold=best_thr)

        scheduler.step(val_m["auc"])

        if writer:
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/AUC",  tr_m["auc"], epoch)
            writer.add_scalar("Val/Loss",   val_loss, epoch)
            writer.add_scalar("Val/AUC",    val_m["auc"], epoch)
            writer.add_scalar("Val/BestThr_F1", best_thr, epoch)

        print(
            "Epoch [{:3d}/{:3d}] | "
            "train loss {:.4f} auc {:.4f} | "
            "val loss {:.4f} auc {:.4f} f1@0.5 {:.4f} best_f1 {:.4f} (thr={:.2f}) | "
            "lr {:.2e} | {:.1f}s".format(
                epoch, config["max_epoch"],
                train_loss, tr_m["auc"],
                val_loss, val_m["auc"], val_m["f1"],
                val_best_m["f1"], best_thr,
                current_lr, time.time() - ep_start,
            )
        )

        _append_csv(csv_path, [
            epoch, train_loss, tr_m["auc"], tr_m["acc"],
            tr_m["precision"], tr_m["recall"], tr_m["f1"],
            val_loss, val_m["auc"], val_m["acc"],
            val_m["precision"], val_m["recall"], val_m["f1"],
            best_thr, val_best_m["f1"], current_lr,
        ], header)

        improved = val_m["auc"] > best_val_auc
        if improved:
            best_val_auc = val_m["auc"]
            patience_cnt = 0
            print(f"  *** Best AUC mới: {best_val_auc:.4f} → lưu {best_model_path}")
            torch.save({
                "model_state_dict": _get_model_state_dict_for_save(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch, "best_val_auc": best_val_auc,
                "model_name": MODEL_NAME, "scheduler_monitor": "val_auc",
            }, best_model_path)
        else:
            patience_cnt += 1

        torch.save({
            "model_state_dict": _get_model_state_dict_for_save(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "best_val_auc": best_val_auc,
            "model_name": MODEL_NAME, "scheduler_monitor": "val_auc",
        }, last_model_path)

        if patience_cnt >= config.get("patience", 7):
            print(f"Early stopping sau {patience_cnt} epoch không cải thiện.")
            break

    print(f"\nTổng thời gian train: {(time.time()-t0)/60:.1f} phút")

    # Đánh giá cuối với best model
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        _load_model_state_dict(model, ckpt["model_state_dict"], strict=True)

    model.eval()
    _, val_true, val_prob = _run_epoch(
        model, val_loader, val_criterion, device=device, phase="val",
        scaler=scaler, use_amp=use_amp,
    )
    best_thr, _ = _find_best_threshold(val_true, val_prob)
    final_m = _compute_metrics(val_true, val_prob, threshold=best_thr)
    print(
        "Final VAL | thr={:.2f} | auc={:.4f} | acc={:.4f} | "
        "precision={:.4f} | recall={:.4f} | f1={:.4f}".format(
            best_thr, final_m["auc"], final_m["acc"],
            final_m["precision"], final_m["recall"], final_m["f1"],
        )
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{MODEL_NAME}_{task}_curves.png"))
    _plot_confusion_matrix(val_true, final_m["y_pred"],
                           os.path.join(eval_folder, f"{MODEL_NAME}_{task}_confusion.png"))
    _plot_roc(val_true, val_prob, os.path.join(eval_folder, f"{MODEL_NAME}_{task}_roc.png"))

    # Test set
    if test_loader is not None:
        _, te_true, te_prob = _run_epoch(
            model, test_loader, test_criterion, device=device, phase="test",
            scaler=scaler, use_amp=use_amp,
        )
        te_m = _compute_metrics(te_true, te_prob, threshold=best_thr)
        print(
            "Final TEST| thr={:.2f} | auc={:.4f} | acc={:.4f} | "
            "precision={:.4f} | recall={:.4f} | f1={:.4f}".format(
                best_thr, te_m["auc"], te_m["acc"],
                te_m["precision"], te_m["recall"], te_m["f1"],
            )
        )
        _append_csv(
            os.path.join(eval_folder, f"{MODEL_NAME}_{task}_test_metrics.csv"),
            [task, best_thr, te_m["auc"], te_m["acc"],
             te_m["precision"], te_m["recall"], te_m["f1"]],
            ["task", "threshold", "auc", "acc", "precision", "recall", "f1"],
        )
        _plot_confusion_matrix(te_true, te_m["y_pred"],
                               os.path.join(eval_folder, f"{MODEL_NAME}_{task}_test_confusion.png"))
        _plot_roc(te_true, te_prob, os.path.join(eval_folder, f"{MODEL_NAME}_{task}_test_roc.png"))

    if writer:
        writer.flush(); writer.close()

    print(f"Kết quả lưu tại: {eval_folder}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 + ViT trên MRNet (Kaggle)")
    parser.add_argument("--tasks", type=str, default="abnormal,acl,meniscus",
                        help="Danh sách task, phân cách bằng dấu phẩy")
    parser.add_argument("--data-root", type=str, default=None,
                        help=f"Thư mục data (mặc định: {KAGGLE_DATA_ROOT})")
    parser.add_argument("--labels-root", type=str, default=None,
                        help=f"Thư mục labels (mặc định: {KAGGLE_LABELS_ROOT})")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Số epoch đóng băng backbone EfficientNetB0 đầu tiên (mặc định: 5)")
    parser.add_argument("--target-slices", type=int, default=None,
                        help="Override số slice (mặc định theo KAGGLE_CONFIG_OVERRIDES)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    data_root   = args.data_root   or KAGGLE_DATA_ROOT
    labels_root = args.labels_root or KAGGLE_LABELS_ROOT

    # Kiểm tra thư mục tồn tại
    if not os.path.isdir(data_root):
        print(f"[CẢNH BÁO] data_root không tồn tại: {data_root}")
        print("  → Đang chạy trên máy local? Hãy truyền --data-root và --labels-root")
        sys.exit(1)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        cfg = dict(base_config)
        cfg.update(KAGGLE_CONFIG_OVERRIDES)
        cfg["task"] = task

        if args.target_slices is not None:
            cfg["target_slices"] = args.target_slices
        if args.lr is not None:
            cfg["lr"] = args.lr
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size

        print("\nCấu hình training:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

        train(
            config=cfg,
            data_root=data_root,
            labels_root=labels_root,
            freeze_epochs=args.freeze_epochs,
        )

    print("\n=== Hoàn thành tất cả tasks ===")
