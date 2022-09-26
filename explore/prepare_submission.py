import os.path
from argparse import ArgumentParser
from sensorium.models.model_initialization import sota, SotaEnsemble
from sensorium.datasets.dataset_initialization import init_loaders
from sensorium.utility import submission

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_label', type=str, default='generalization_model')
    return parser.parse_args()



def main():
    args = parse_args()
    dataset_name = '26872-17-20'
    basepath = "./notebooks/data/"
    # just change the model here

    dataloaders = init_loaders(basepath)
    # checkpoint = f'model_checkpoints/{args.model_label}.pth'
    # print(checkpoint)
    # model = sota(dataloaders, checkpoint)
    checkpoints = [f'model_checkpoints/generalization_model_{n}.pth' for n in range(41, 61)]
    model = SotaEnsemble(dataloaders, checkpoints).cuda()


    save_directory = f"./submission_files/{args.model_label}"
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
