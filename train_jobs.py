TRAIN_JOBS = {
    'classic': {
        'model_name': 'mobilenet_classic',
        'architecture_name': 'mobilenet_original',
        'lr': 0.001,
        'weight_decay': 0.0005,
        'batch_size': 2,
        'epochs': 2
    },

    'delta_cnn': {
        'model_name': 'mobilenet_deltacnn',
        'architecture_name': 'mobilenet_classic',
        'lr': 0.001,
        'weight_decay': 0.0005,
        'batch_size': 16,
        'epochs': 10,
    }
}