config = {
        'batch_sizes': (32,24) #train, test
        'model_name': 'resnet18',
        'log_path': 'data/log',
        'use_inputs': ['images'],
        'data_path': 'data',
        'max_epochs': 1,
        'num_wokers': 0,
        'layers': 3,
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'loss_name': 'cross_entropy_loss',
        'optimizer': 'sgd',
        'metrics': [],
        'log_cmd': False,
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None, 
        'shuffle': (True, False) #train, test
}

debug_options = {

}

log_keys = [
        'model_name',
]
