import os
from nnfabrik.builder import get_data


def init_loaders(basepath, single=False):
    if single:
        filenames = ['notebooks/data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip', ]
    else:
        filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]

    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {
        'paths': filenames,
        'normalize': True,
        'include_behavior': False,
        'include_eye_position': False,
        'batch_size': 128,
        'scale':.25,
    }

    dataloaders = get_data(dataset_fn, dataset_config)
    return dataloaders
