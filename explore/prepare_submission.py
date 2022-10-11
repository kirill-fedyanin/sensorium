import os.path
from argparse import ArgumentParser
from sensorium.models.model_initialization import sota, SotaEnsemble
from sensorium.datasets.dataset_initialization import init_loaders
from sensorium.utility import submission
from sensorium.models.model_initialization import init_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./notebooks/data/")
    parser.add_argument("--model", type=str, default='generalization')
    parser.add_argument("--checkpoint_path", type=str, default='model_checkpoints/generalization_model.pth')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = '26872-17-20'
    dataloaders = init_loaders(args.data_path, scale=0.25)
    model = init_model(args.model, args.checkpoint_path, dataloaders)

    save_directory = f"./submission_files/{args.model}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # generate the submission file
    submission.generate_submission_file(
        trained_model=model,
        dataloaders=dataloaders,
        data_key=dataset_name,
        path=save_directory,
        device="cuda"
    )


if __name__ == '__main__':
    main()
