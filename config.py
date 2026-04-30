config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 1e-5,
    'starting_epoch' : 0,
    'batch_size' : 4,
    'log_val' : 10,
    'task' : 'acl', # "meniscus" and  "acl" are the other options
    'weight_decay' : 1e-4,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 160,
    'target_slices' : 24,
    'num_workers' : 4
}
