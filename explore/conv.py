import os

import torch
import matplotlib.pyplot as plt

from nnfabrik.builder import get_data


def get_loader(file_names):

    if type(file_names) != list:
        file_names = [file_names]

    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {'paths': file_names,
                     'normalize': True,
                     # 'include_behavior': True,
                     # 'include_eye_position': True,
                     'batch_size': 128,
                     'exclude': None,
                     'file_tree': True,
                     'scale': 1,
                     'add_behavior_as_channels': False,
                     }

    dataloaders = get_data(dataset_fn, dataset_config)
    key = list(dataloaders['train'].keys())[0]
    train_loader = dataloaders['train'][key]
    val_loader = dataloaders['validation'][key]
    test_loader = dataloaders['test'][key]
    final_test_loader = dataloaders['final_test'][key]
    return train_loader, val_loader, test_loader, final_test_loader


def main():
    print(os.listdir(''))
    data_file = 'notebooks/data/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip'
    train_loader, val_loader, _, _ = get_loader(data_file)

    batch = next(iter(train_loader))
    import ipdb; ipdb.set_trace()
    return

    for i in range(3):
        image = batch.images[i].cpu().numpy()[0]
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()