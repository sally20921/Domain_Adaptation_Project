config = {
        'image_size':224,
        'batch_size': 128,
        'epochs': 100,
        'use_inputs': ["CIFAR10"], #STL10
        'pretrain': True,
        'model_name': "resnet18", #resnet50
        'projection_dim': 64,
        'optimizer': "Adam",
        'weight_decay': 1e-6,
        'temperature': 0.5,
        'logistic_batch_size': 256,
        'logistic_epoch':  500,
        'data_path': 'data',
        'num_workers': 10,
        'learning_rate': 1e-4,
        'loss_name': 'nt_xent_loss',
        'metrics': [],
        'log_cmd': False,
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None, 
        'shuffle': True
}

debug_options = {
}

log_keys = [
        'model_name',
]

