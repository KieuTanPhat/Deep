# DeepLearning_v - Train Model MRI Gối

Thư mục này chứa pipeline huấn luyện model deep learning để chẩn đoán chấn thương gối từ MRI theo 3 mặt cắt:

- `axial`
- `coronal`
- `sagittal`

Các task hiện hỗ trợ:

- `abnormal`: phát hiện bất thường tổng quát
- `acl`: phát hiện tổn thương dây chằng chéo trước
- `meniscus`: phát hiện tổn thương sụn chêm

## Cấu Trúc Thư Mục

```text
DeepLearning_v/
├── config.py                  # Cấu hình train mặc định
├── train_demo.py              # Script train chính đang dùng                  # Script train cũ cho MRNet
├── dataset/
│   └── dataset.py             # Dataset loader, preprocessing, DataLoader
├── models/
│   ├── EfficientNetB0.py      # Model EfficientNet-B0 3 mặt cắt
│   ├── Densenet121.py
│   └── MRnet.py
├── preprocessing/             # Augmentation, resize, sampling, normalization
├── utils/                     # Hàm hỗ trợ train/evaluate cũ
├── tools/                     # Công cụ phụ trợ
├── weights/                   # Checkpoint .pth sau khi train
└── evaluation/                # CSV metrics, ROC, confusion matrix, curves
```

## Cấu Trúc Dữ Liệu

Mặc định dữ liệu được đọc từ:

```text
data/
├── train/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
├── valid/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
└── test/
    ├── axial/
    ├── coronal/
    └── sagittal/
```

Mỗi file ảnh MRI là `.npy`, ví dụ:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

Label được đọc từ thư mục `labels/`:

```text
labels/
├── train-abnormal.csv
├── valid-abnormal.csv
├── test-abnormal.csv
├── train-acl.csv
├── valid-acl.csv
├── test-acl.csv
├── train-meniscus.csv
├── valid-meniscus.csv
└── test-meniscus.csv
```

Định dạng mỗi dòng CSV:

```csv
id,label
```

Trong code hiện tại CSV được đọc với `header=None`, nên nếu file có header thật thì cần bỏ header hoặc chỉnh loader.

## Cấu Hình Chính

Cấu hình nằm trong `config.py`.

Các tham số đáng chú ý:

```python
'lr': 2e-5
'batch_size': 16
'image_size': 224
'target_slices': 24
'weight_decay': 1e-4
'patience': 5
'use_gradient_accumulation': 1
'gradient_accumulation_steps': 2
```

Với cấu hình mặc định, batch thực tế đưa qua GPU là `16`, nhưng effective batch size là:

```text
16 * 2 = 32
```

Nếu muốn tắt gradient accumulation:

```python
'use_gradient_accumulation': 0
'gradient_accumulation_steps': 1
```

## Train Model

Chạy tất cả task bằng EfficientNet-B0:

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\DeepLearning_v
py -3.13 train_demo.py --model efficientnetb0 --tasks abnormal,acl,meniscus --data-root data --labels-root labels
```

Chạy riêng task `abnormal`:

```powershell
py -3.13 train_demo.py --model efficientnetb0 --tasks abnormal --data-root data --labels-root labels
```

Chạy riêng task `acl`:

```powershell
py -3.13 train_demo.py --model efficientnetb0 --tasks acl --data-root data --labels-root labels
```

Chạy riêng task `meniscus`:

```powershell
py -3.13 train_demo.py --model efficientnetb0 --tasks meniscus --data-root data --labels-root labels
```

## Warm-Start ACL/Meniscus Từ Abnormal

`train_demo.py` hỗ trợ khởi tạo ACL và meniscus từ checkpoint abnormal.

Ví dụ:

```powershell
py -3.13 train_demo.py --model efficientnetb0 --tasks acl,meniscus --abnormal-pth weights/abnormal/efficientnetb0_best_model.pth
```

Các task được warm-start nằm trong `config.py`:

```python
'warmstart_tasks': ['acl', 'meniscus']
'warmstart_from_abnormal': 1
```

## Output Sau Khi Train

Checkpoint được lưu vào:

```text
weights/<task>/
├── efficientnetb0_best_model.pth
└── efficientnetb0_last_checkpoint.pth
```

Kết quả đánh giá được lưu vào:

```text
evaluation/efficientnetb0_<task>/
├── efficientnetb0_<task>_metrics.csv
├── efficientnetb0_<task>_test_metrics.csv
├── efficientnetb0_<task>_curves.png
├── efficientnetb0_<task>_roc.png
├── efficientnetb0_<task>_confusion.png
├── efficientnetb0_<task>_test_roc.png
└── efficientnetb0_<task>_test_confusion.png
```

## Training Pipeline Hiện Tại

Pipeline trong `train_demo.py` đang dùng:

- `BCEWithLogitsLoss`
- `Adam`
- `ReduceLROnPlateau` theo `val_auc`
- Early stopping theo `val_auc`
- Mixed precision khi có CUDA
- Gradient accumulation có thể bật/tắt trong `config.py`
- Lưu best checkpoint theo validation AUC
- Tìm best threshold theo F1 trên validation set
- Đánh giá test set bằng threshold tốt nhất từ validation

## Preprocessing

Dataset loader trong `dataset/dataset.py` xử lý mỗi volume như sau:

1. Load `.npy` cho 3 mặt cắt.
2. Lấy mẫu đều về `target_slices`.
3. Augmentation khi train.
4. Center crop nếu ảnh lớn hơn `image_size`.
5. Resize an toàn về `image_size x image_size` nếu cần.
6. Normalize về thang giống pipeline train.
7. Chuyển ảnh grayscale thành 3 kênh để đưa vào EfficientNet.

Lưu ý: `WeightedRandomSampler` đã được bỏ để tránh xử lý lệch class hai lần khi vẫn dùng `pos_weight` trong loss.

## Scheduler Và Early Stopping

Scheduler hiện dùng:

```python
ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=3,
    factor=0.3,
    threshold=1e-4,
)
```

Và được step bằng:

```python
scheduler.step(val_metrics["auc"])
```

Điều này giúp scheduler nhất quán với tiêu chí lưu best model và early stopping, đều dựa trên `val_auc`.

## Gợi Ý Khi Train Abnormal

Với `abnormal`, validation AUC có thể cao nhưng threshold `0.5` không phải lúc nào phù hợp. Sau khi train, nên xem:

```text
evaluation/efficientnetb0_abnormal/efficientnetb0_abnormal_metrics.csv
```

Cột quan trọng:

- `val_auc`
- `val_best_threshold`
- `val_best_f1`
- `val_best_precision`
- `val_best_recall`

Threshold tốt nhất của abnormal có thể thấp hơn `0.5`, ví dụ khoảng `0.2`.

## Ghi Chú

File `requirements.txt` hiện là bản cũ theo môi trường ban đầu. Nếu chạy bằng Python 3.13, nên đảm bảo các package chính đã có:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tensorboard`
- `tqdm`

Kết quả model chỉ phục vụ nghiên cứu/hỗ trợ kỹ thuật, không thay thế chẩn đoán y khoa.
