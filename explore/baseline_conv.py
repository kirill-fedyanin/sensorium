"""
What is the question
how exactly are the sampled?
are nearby similar?
"""
import os
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer
from sensorium.training import standard_trainer
from explore.baseline_conv_model import init_model


def init_loaders(basepath, single=False, **kwargs):
    # as filenames, we'll select all 7 datasets
    filenames = sorted([os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ])
    if single:
        filenames = filenames[:1]

    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {
        'paths': filenames,
        'normalize': True,
        'include_behavior': False,
        'include_eye_position': False,
        'batch_size': 128,
    }

    dataset_config.update(kwargs)

    dataloaders = get_data(dataset_fn, dataset_config)
    return dataloaders


def plt_activations(dataloader):
    batch = next(iter(dataloader))

    positions = dataloader.dataset.neurons.cell_motor_coordinates
    print(positions.shape)
    # print(np.std(positions[:, 2]))

    for i, r in enumerate(batch.responses):
        r = r.cpu().numpy()
        if i == 20:
            break
        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], c=r, s=20, alpha=0.3, cmap='Reds')
        # plt.legend()
        plt.colorbar()
        plt.show()



def main():
    print('started')
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    basepath = "./notebooks/data/"
    dataloaders = init_loaders(
        basepath, single=True, scale=0.25
    )

    plt_activations(dataloaders['train']['21067-10-18'])
    return



    # ys = []
    # for batch in tqdm(dataloaders['train']['21067-10-18']):
    #     ys.append(batch.responses)
    #     pass
    # ys = torch.cat(ys).cpu().detach().numpy()
    # return

    model = init_model(dataloaders, seed=seed).cuda()

    validation_score, trainer_output, state_dict = standard_trainer(
        model,
        dataloaders,
        seed=seed,
        max_iter=10,
        verbose=True,
        lr_decay_steps=4,
        avg_loss=False,
        # lr_init=0.009,
        lr_init=0.009,
        track_training=True,
    )

    # trainer(model, dataloaders, seed=seed)
    print(validation_score)
    print(trainer_output)
    torch.save(model.state_dict(), f'model_checkpoints/smoll_generalization_model_{seed}.pth')


if __name__ == '__main__':
    main()
