import os
from nnfabrik.builder import get_data


def init_loaders(basepath, **kwargs):
    # as filenames, we'll select all 7 datasets
    filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]

    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {
        'paths': filenames,
        'normalize': True,
        'batch_size': 128,
    }

    dataset_config.update(kwargs)

    dataloaders = get_data(dataset_fn, dataset_config)
    return dataloaders
