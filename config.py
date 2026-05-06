import os

config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 2e-5,
    'starting_epoch' : 0,
    'batch_size' : 8,
    'log_val' : 10,
    'task' : 'acl', # "meniscus" and  "acl" are the other options
    'weight_decay' : 1e-4,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 24,
    'num_workers' : 2,
    # Warm-start ACL/Meniscus from Abnormal checkpoint (useful on Kaggle).
    # You can override by env var ABNORMAL_WARMSTART_PTH or CLI --abnormal-pth.
    'abnormal_warmstart_path' : os.environ.get(
        'ABNORMAL_WARMSTART_PTH',
        '/kaggle/working/weights/abnormal/efficientnetb0_best_model.pth',
    ),
    'warmstart_tasks' : ['acl', 'meniscus'],
    'warmstart_from_abnormal' : 1,
}
