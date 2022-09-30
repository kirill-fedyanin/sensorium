import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data
import os
from sensorium.utility import get_signal_correlations, get_fev
from sensorium.utility.measure_helpers import get_df_for_scores
from sensorium.models.model_initialization import sota, SotaEnsemble
from sensorium.datasets.dataset_initialization import init_loaders

basepath = "./notebooks/data/"


def benchmark(dataloaders, model, tier='validation'):
    correlation_to_average = get_signal_correlations(
        model, dataloaders, tier=tier, device="cuda", as_dict=True
    )

    measure_attribute = "Correlation to Average"
    df = get_df_for_scores(
        session_dict=correlation_to_average,
        measure_attribute=measure_attribute,
    )
    print(df[measure_attribute].mean())
    fig = plt.figure(figsize=(15,8))
    sns.boxenplot(x="dataset", y=measure_attribute, data=df, )
    plt.xticks(rotation = 45)
    sns.despine(trim=True)
    plt.show()

    # feves = get_fev(model, dataloaders, tier="test", device="cuda", as_dict=True)
    # measure_attribute = "FEVE"
    # df = get_df_for_scores(
    #     session_dict=feves,
    #     measure_attribute=measure_attribute,
    # )
    #
    # fig = plt.figure(figsize=(15, 8))
    # sns.boxenplot(x="dataset", y=measure_attribute, data=df, )
    # plt.xticks(rotation=45)
    # plt.ylim([-.1, 1])
    # sns.despine(trim=True)
    # plt.show()


def main():
    # just change the model here
    dataloaders = init_loaders(basepath)
    # checkpoint = 'model_checkpoints/generalization_model.pth'
    # print(checkpoint)
    # model = sota(dataloaders, checkpoint)


    checkpoints = [f'model_checkpoints/generalization_model_{n}.pth' for n in range(41, 61)]
    model = SotaEnsemble(dataloaders, checkpoints).cuda()

    # dataloaders = init_loaders(single=True)
    # model = ln_model(
    #     dataloaders,
    #     "model_checkpoints/notebook_examples/sensorium_ln_model.pth"
    # )


    benchmark(dataloaders, model, tier='test')


if __name__ == '__main__':
    main()