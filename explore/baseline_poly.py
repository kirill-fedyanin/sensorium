
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
from explore.baseline_conv_model import init_model


def init_loaders(basepath, single=None, **kwargs):
    filenames = [file for file in os.listdir(basepath) if ".zip" in file]
    if single is not None:
        filenames = [f for f in filenames if f.startswith(f'static{single}')]
    filenames = sorted([os.path.join(basepath, file) for file in filenames])

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


def make_hist(predictions, ys):
    bins = np.arange(0, 3, 0.1)
    plt.hist(predictions, bins=bins, alpha=0.5)
    plt.hist(ys, bins=bins, alpha=0.5)
    plt.show()


def torch_corr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    eps = 1e-8

    cost = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2))+eps)*(torch.sqrt(torch.sum(vy ** 2))+eps))
    return cost


class Power(nn.Module):
    """
    x_ = x**a
    """
    def __init__(self, model=None):
        super().__init__()
        self.coefs = nn.Parameter(torch.tensor([1.0]))
        self.model = model


    def forward(self, x):
        if self.model is not None:
            with torch.no_grad():
                x = self.model(x)
        return x ** self.coefs[0]


class LogPower(nn.Module):
    """
    x_ = x**a + b*(log(1+x)**c)
    """
    def __init__(self, a=1.0, b=0.01, c=1.0, model=None):
        super().__init__()
        self.coefs = nn.Parameter(torch.tensor([a, b, c]))
        self.model = model

    def forward(self, x):
        if self.model is not None:
            with torch.no_grad():
                x = self.model(x)
        return self.coefs[1]*(torch.log(1+x)**self.coefs[2])


def tune_activations(dataloader, model):
    # head = LogPower(model=model).cuda()
    head = Power(model=model).cuda()

    optimizer = torch.optim.SGD(head.parameters(), lr=5e-2)

    for epoch in range(100):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()
            predictions = head(batch.images)
            loss = -torch_corr(predictions, batch.responses)
            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
        print(epoch)
        print(head.coefs)
        print(np.mean(losses))

    preds = []
    ys = []
    for batch in dataloader:
        preds.append(head(batch.images).detach().cpu().numpy())
        ys.append(batch.responses.cpu().numpy())

    preds = np.concatenate(preds).flatten()
    ys = np.concatenate(ys).flatten()
    print(np.mean(preds))
    print(np.std(preds))

    print(np.mean(ys))
    print(np.std(ys))
    bins = np.linspace(0, 2, 50)
    print('corr', corr(preds, ys))
    plt.figure(figsize=(16, 9))
    plt.hist(ys, bins=bins, alpha=0.5)
    plt.hist(preds, bins=bins, alpha=0.5)
    plt.show()



def plt_activations(dataloader, model=None):
    batch = next(iter(dataloader))

    positions = dataloader.dataset.neurons.cell_motor_coordinates
    print(positions.shape)
    # print(np.std(positions[:, 2]))
    idx = np.arange(len(positions)) #[positions[:, 2] < 220]
    positions = positions[idx]

    if model:
        preds = model(batch.images).detach().cpu().numpy()

    for i, r in enumerate(batch.responses):
        r = r.cpu().numpy()
        if i == 5:
            break
        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.5, cmap='Reds')
        plt.show()
        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], c=r[idx], s=10, cmap='Reds')
        # if model:
        #     plt.scatter(positions[:, 0], positions[:, 1], c=preds[i][idx], s=20, alpha=0.5, cmap='Blues')
        # plt.legend()
        plt.colorbar()
        plt.show()

        # plt.figure()
        # bins = np.arange(0, 10, 0.1)
        # plt.hist(preds[i][idx], bins=bins, alpha=0.5)
        # plt.hist(r[idx], bins=bins, alpha=0.5)
        # plt.show()


def main():
    print('started')
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    basepath = "./notebooks/data/"
    # mouse_id = '21067-10-18'
    # mouse_id = '22846-10-16'
    mouse_id = '23343-5-17'
    dataloaders = init_loaders(
        basepath, single=mouse_id, scale=0.25
    )

    # ys = []
    # for batch in tqdm(dataloaders['train']['21067-10-18']):
    #     ys.append(batch.responses)
    #     pass
    # ys = torch.cat(ys).cpu().detach().numpy()
    # return

    model = init_model(dataloaders, seed=seed).cuda()
    model_path = f'model_checkpoints/smoll_generalization_model_{seed}_{mouse_id}.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        validation_score, trainer_output, state_dict = standard_trainer(
            model,
            dataloaders,
            seed=seed,
            max_iter=200,
            verbose=True,
            lr_decay_steps=4,
            avg_loss=False,
            # lr_init=0.009,
            lr_init=0.009,
            track_training=True,
        )

        print(validation_score)
        print(trainer_output)
        torch.save(model.state_dict(), model_path)
    import ipdb; ipdb.set_trace()
    tune_activations(dataloaders['train'][mouse_id], model)


if __name__ == '__main__':
    main()
