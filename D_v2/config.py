config = {
    'batch_size': 512,
    'num_workers': 16,
    'start_epoch': 0,
    'epochs': 100,
    'data_dir_path': "dataset",
    'log_dir_path': "checkpoint",
    'dataset': "CIFAR10",
    'save_num_epochs': 1,
    'img_size': 32,
    'optimizer': "Adam",
    'weight_decay': 1.0e-6,
    'temperature': 0.1,
    'resnet': "resnet50",
    'normalize': True,
    'projection_dim': 64
}


debug_options = {
}

log_keys = [
        'model_name',
]
