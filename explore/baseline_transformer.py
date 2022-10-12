import os
from argparse import ArgumentParser

import torch
from torch import nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer
from neuralpredictors.measures.np_functions import corr, fev

from sensorium.training import standard_trainer
from explore.baseline_transformer_model import init_model

from neuralpredictors.measures.modules import PoissonLoss


def init_loaders(basepath, single=False, **kwargs):
    # as filenames, we'll select all 7 datasets
    filenames = sorted([os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ])
    if single:
        filenames = filenames[:1]

    dataset_fn = 'sensorium.datasets.static_loaders'
    dataset_config = {
        'paths': filenames,
        'normalize': False,
        'include_behavior': False,
        'include_eye_position': False,
    }

    dataset_config.update(kwargs)

    dataloaders = get_data(dataset_fn, dataset_config)
    return dataloaders


def train(model, dataloaders, mouse_id):
    pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)
    criterion = PoissonLoss()

    for epoch in range(5):
        losses = []
        for batch in tqdm(dataloaders['train'][mouse_id]):
            model(batch.images)
            # optimizer.zero_grad()
            # pred = model(batch.images)
            # loss = criterion(pred, batch.responses)
            # loss.backward()
            # optimizer.step()
            # losses.append(loss.item())
        print(np.mean(losses))



def main():
    print('started')
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    seed = args.seed
    basepath = "./notebooks/data/"
    if args.debug:
        dataloaders = init_loaders(
            basepath, single=True, scale=0.25, batch_size=4, image_n=100
        )
        max_iter = 5
    else:
        dataloaders = init_loaders(
            basepath, single=True, scale=0.1, batch_size=16
        )
        max_iter = 200

    mouse_id = '21067-10-18'
    loader = dataloaders['train'][mouse_id]
    batch = next(iter(loader))
    model = init_model(dataloaders, seed=seed).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    validation_score, trainer_output, state_dict = standard_trainer(
        model,
        dataloaders,
        seed=seed,
        max_iter=max_iter,
        verbose=True,
        lr_decay_steps=4,
        avg_loss=False,
        lr_init=0.009,
        track_training=(not args.debug),
    )

    print(validation_score)
    print(trainer_output)
    torch.save(model.state_dict(), f'model_checkpoints/detr_generalization_model_{seed}.pth')


if __name__ == '__main__':
    main()
