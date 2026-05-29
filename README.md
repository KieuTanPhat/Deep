# DeepLearning-v3 MRI Knee Training

Pipeline train binary classifiers for knee MRI across three planes:

- `axial`
- `coronal`
- `sagittal`

Supported tasks:

- `abnormal`
- `acl`
- `meniscus`

## Main Upgrade: EfficientNetB0 + ViT

The original EfficientNetB0 baseline extracts per-slice CNN features and then uses max pooling over slices. That is fast, but it discards slice order and weak cross-plane interactions.

This repo now adds `efficientnetb0_vit`:

1. EfficientNet-B0 extracts a 1280-d feature for every MRI slice.
2. Slice tokens receive learnable slice-position and plane embeddings.
3. A ViT-style Transformer encoder learns slice order plus axial/coronal/sagittal interactions.
4. CLS + attention pooling feeds a regularized classifier head.

This is the recommended model when trying to beat the previous EfficientNetB0 AUC baseline around `0.941`.

## Data Layout

```text
data/
  train/
    axial/
    coronal/
    sagittal/
  valid/
    axial/
    coronal/
    sagittal/
  test/
    axial/
    coronal/
    sagittal/

labels/
  train-abnormal.csv
  valid-abnormal.csv
  test-abnormal.csv
  train-acl.csv
  valid-acl.csv
  test-acl.csv
  train-meniscus.csv
  valid-meniscus.csv
  test-meniscus.csv
```

Each MRI volume is a `.npy` file such as:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

CSV files are read with `header=None`, so remove headers if your CSV has them.

## Train The Hybrid Model

Train all tasks:

```powershell
python train_demo.py --model efficientnetb0_vit --tasks abnormal,acl,meniscus --data-root data --labels-root labels
```

Train ACL only:

```powershell
python train_demo.py --model efficientnetb0_vit --tasks acl --data-root data --labels-root labels
```

Warm-start ACL and meniscus from the abnormal hybrid checkpoint:

```powershell
python train_demo.py --model efficientnetb0_vit --tasks acl,meniscus --abnormal-pth weights/abnormal/efficientnetb0_vit_best_model.pth
```

Useful memory-safe overrides:

```powershell
python train_demo.py --model efficientnetb0_vit --tasks acl --batch-size 2 --grad-accum-steps 16 --target-slices 24
```

## Automatic Tuning

Run curated trials and let the script rank them by validation AUC:

```powershell
python tools/tune_efficientnetb0_vit.py --tasks acl --data-root data --labels-root labels --seeds 42,1337,2027
```

Quick smoke tuning:

```powershell
python tools/tune_efficientnetb0_vit.py --tasks acl --max-epoch 3 --patience 2 --seeds 42
```

The tuning summary is saved to:

```text
evaluation/efficientnetb0_vit_tuning_summary.csv
```

Recommended process for highest AUC:

1. Tune on `valid` only; do not pick hyperparameters from the test set.
2. Pick the best validation AUC trial from the summary CSV.
3. Re-run that trial with 2-3 seeds.
4. Evaluate the selected checkpoint on `test`.
5. Compare final `test_auc` against the original `0.941` baseline.

## Key Training Defaults

Main defaults live in `config.py`:

```python
'model': use --model efficientnetb0_vit
'lr': 2e-5
'optimizer': 'adamw'
'backbone_lr_mult': 0.3
'batch_size': 4
'gradient_accumulation_steps': 8
'target_slices': 24
'vit_dim': 384
'vit_depth': 2
'vit_heads': 6
'vit_dropout': 0.20
'classifier_dropout': 0.35
'freeze_backbone_epochs': 2
'label_smoothing': 0.02
'max_grad_norm': 1.0
```

Outputs:

```text
weights/<task>/efficientnetb0_vit_best_model.pth
weights/<task>/efficientnetb0_vit_last_checkpoint.pth
evaluation/efficientnetb0_vit_<task>/
```

For tuning runs, artifacts are separated by `exp_name`, for example:

```text
weights/acl/vit_balanced_seed42/
evaluation/efficientnetb0_vit_acl_vit_balanced_seed42/
```

## Notes

- AUC cannot be guaranteed without the actual dataset and a full training run.
- The repo does not include `data/` or `labels/`, so local code-only checks cannot reproduce the final AUC.
- This project is for research/technical support and is not a medical diagnosis system.
