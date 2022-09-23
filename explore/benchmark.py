"""
"""

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer
import os
from sensorium.utility import get_correlations, get_signal_correlations, get_fev
from sensorium.utility.measure_helpers import get_df_for_scores

basepath = "./notebooks/data/"


def init_loaders(single=False):
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


def benchmark(dataloaders, model, tier='validation'):
    correlation_to_average = get_signal_correlations(
        model, dataloaders, tier=tier, device="cuda", as_dict=True
    )

    measure_attribute = "Correlation to Average"
    df = get_df_for_scores(
        session_dict=correlation_to_average,
        measure_attribute=measure_attribute,
    )
    print(df)
    fig = plt.figure(figsize=(15,8))
    sns.boxenplot(x="dataset", y=measure_attribute, data=df, )
    plt.xticks(rotation = 45)
    sns.despine(trim=True)
    plt.show()

    feves = get_fev(model, dataloaders, tier="test", device="cuda", as_dict=True)
    measure_attribute = "FEVE"
    df = get_df_for_scores(
        session_dict=feves,
        measure_attribute=measure_attribute,
    )

    fig = plt.figure(figsize=(15, 8))
    sns.boxenplot(x="dataset", y=measure_attribute, data=df, )
    plt.xticks(rotation=45)
    plt.ylim([-.1, 1])
    sns.despine(trim=True)
    plt.show()


def init_sota(dataloaders, checkpoint_path):
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {
        'pad_input': False,
        'layers': 4,
        'input_kern': 9,
        'gamma_input': 6.3831,
        'gamma_readout': 0.0076,
        'hidden_kern': 7,
        'hidden_channels': 64,
        'depth_separable': True,
        'grid_mean_predictor': {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1,
            'hidden_features': 30,
            'final_tanh': True
        },
        'init_sigma': 0.1,
        'init_mu_range': 0.3,
        'gauss_type': 'full',
        'shifter': False,
        'stack': -1,
    }

    model = get_model(model_fn=model_fn,
                      model_config=model_config,
                      dataloaders=dataloaders,
                      seed=42,)

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def ln_model(dataloaders, checkpoint_path):
    model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
    model_config = {'pad_input': False,
                    'stack': -1,
                    'layers': 3,
                    'input_kern': 9,
                    'gamma_input': 6.3831,
                    'gamma_readout': 0.0076,
                    'hidden_kern': 7,
                    'hidden_channels': 64,
                    'grid_mean_predictor': {
                        'type': 'cortex',
                        'input_dimensions': 2,
                        'hidden_layers': 1,
                        'hidden_features': 30,
                        'final_tanh': True
                    },
                    'depth_separable': True,
                    'init_sigma': 0.1,
                    'init_mu_range': 0.3,
                    'gauss_type': 'full',
                    'linear': True
                    }
    model = get_model(model_fn=model_fn,
                      model_config=model_config,
                      dataloaders=dataloaders,
                      seed=42, )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def main():
    # just change the model here
    dataloaders = init_loaders()
    model = init_sota(
        dataloaders,
        "notebooks/model_tutorial/model_checkpoints/pretrained/generalization_model.pth"
    )

    # dataloaders = init_loaders(single=True)
    # model = ln_model(
    #     dataloaders,
    #     "model_checkpoints/notebook_examples/sensorium_ln_model.pth"
    # )

    benchmark(dataloaders, model, tier='test')


if __name__ == '__main__':
    main()