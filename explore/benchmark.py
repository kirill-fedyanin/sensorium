"""
Lvl 4
benchmark plus
"""
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data
import os
from sensorium.utility import get_signal_correlations, get_fev
from sensorium.utility.measure_helpers import get_df_for_scores
from sensorium.models.model_initialization import sota, SotaEnsemble
from sensorium.datasets.dataset_initialization import init_loaders
from sensorium.models.model_initialization import init_model
from sensorium.utility import submission


def benchmark(dataloaders, model, tier='validation', show_feves=False):
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

    if show_feves:
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


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./notebooks/data/")
    parser.add_argument("--model", type=str, default='generalization')
    parser.add_argument("--checkpoint_path", type=str, default='model_checkpoints/generalization_model.pth')
    parser.add_argument("--show_feves", default=False, action='store_true')
    parser.add_argument('--plus', default=False, action='store_true')
    parser.add_argument('--submission', default=False, action='store_true')
    parser.add_argument('--note', default='', type=str)
    args = parser.parse_args()


    basepath = "./notebooks/data/"
    dataloaders = init_loaders(
        basepath, scale=0.25, include_behavior=args.plus, include_eye_position=args.plus
    )

    model = init_model(args.model, args.checkpoint_path, dataloaders)

    benchmark(dataloaders, model, tier='test', show_feves=args.show_feves)

    if submission:
        save_directory = f"./submission_files/{args.model}{args.note}"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # generate the submission file
        dataset_name = '26872-17-20'  # the test one?
        submission.generate_submission_file(
            trained_model=model,
            dataloaders=dataloaders,
            data_key=dataset_name,
            path=save_directory,
            device="cuda"
        )


if __name__ == '__main__':
    main()
