import os

config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 2e-5,
    'starting_epoch' : 0,
    # Micro-batch per optimizer forward/backward step.
    # Keep this small on 14-16GB GPUs; gradient accumulation controls effective batch size.
    'batch_size' : 4,
    'log_val' : 10,
    'task' : 'acl', # "meniscus" and  "acl" are the other options
    'weight_decay' : 1e-4,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    'seed' : 42,
    'optimizer' : 'adamw',
    'backbone_lr_mult' : 0.3,
    'max_grad_norm' : 1.0,
    'label_smoothing' : 0.02,
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 24,
    'num_workers' : 2,
    'use_gradient_accumulation' : 1,
    # Effective batch size = batch_size * gradient_accumulation_steps = 4 * 8 = 32.
    'gradient_accumulation_steps' : 8,
    # EfficientNet-B0 + ViT fusion defaults. The transformer keeps slice order
    # and cross-plane interactions that max pooling loses.
    'vit_dim' : 384,
    'vit_depth' : 2,
    'vit_heads' : 6,
    'vit_mlp_ratio' : 2.0,
    'vit_dropout' : 0.20,
    'classifier_dropout' : 0.35,
    'vit_pooling' : 'cls_attention',
    'freeze_backbone_epochs' : 2,
    # Warm-start ACL/Meniscus from Abnormal checkpoint (useful on Kaggle).
    # You can override by env var ABNORMAL_WARMSTART_PTH or CLI --abnormal-pth.
    'abnormal_warmstart_path' : os.environ.get(
        'ABNORMAL_WARMSTART_PTH',
        '/kaggle/working/weights/abnormal/efficientnetb0_best_model.pth',
    ),
    'warmstart_tasks' : ['acl', 'meniscus'],
    'warmstart_from_abnormal' : 1,
}
