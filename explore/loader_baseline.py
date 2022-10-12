"""
lvl 9
How many images and how many records by id?
"""
from pathlib import Path
import os

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader,  Dataset
import matplotlib.pyplot as plt


def get_loader(base_dir):
    dataset = SensoirumDataset(base_dir, list(range(0, 1000)))
    train_loader = DataLoader(dataset, batch_size=17)
    return train_loader


def get_tensor(base_dir, tier='train'):
    base_dir = Path(base_dir)
    image_dir = base_dir / 'data/images'
    tiers = np.load(base_dir / 'meta/trials/tiers.npy')
    image_ids = np.load(base_dir / 'meta/trials/frame_image_id.npy')
    images = np.array([
        np.load(image_dir / name) for name in os.listdir(image_dir)
    ])
    images = images[tiers == tier]
    print(len(images))
    tester = {}
    for i, image in enumerate(images):
        tester[np.sum(image)] = 1
    print(len(tester))

    for i, image in enumerate(images):
        plt.imshow(image[0], cmap='gray')
        plt.show()


class SensoirumDataset(Dataset):
    def __init__(self, base_dir, tier='train', transformations=None):
        base_dir = Path(base_dir)
        self.image_path = base_dir / 'data/images'
        self.response_path = base_dir / 'data/responses'
        # tiers = np.load(base_dir / "meta/tiers.npy")
        # self.idx
        self.transformations = transformations

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        image = torch.Tensor(np.load(self.image_path / f'{idx}.npy')) / 255 - 0.5
        response = torch.Tensor(np.load(self.response_path / f'{idx}.npy'))
        return image, response


def explore_meta(basic_dir):
    print(os.listdir(basic_dir / 'meta'))
    trials = basic_dir / 'meta/trials'
    print(os.listdir(trials))

    for name in os.listdir(trials):
        print(name)
        print(len(np.load(trials / name)))
        print(np.load(trials / name))
        print()


def get_repeats(base_dir):
    trials = base_dir / 'meta/trials'
    tiers = np.load(trials / 'tiers.npy')
    image_ids = np.load(trials / 'frame_image_id.npy')
    print(len(image_ids))
    image_ids = image_ids[tiers == 'test']
    print(image_ids)
    print(len(image_ids))
    print(len(np.unique(image_ids)))



def main():
    base_dir = Path('data') / os.listdir('data')[0]
    get_repeats(base_dir)
    # get_tensor(base_dir, 'test')

    # explore_meta(data_dir)

    # loader =  get_loader(data_dir)
    # print(loader)
    #
    # for batch in  loader:
    #     print(batch)
    #     import ipdb; ipdb.set_trace()
    #     break


if __name__ == '__main__':
    main()

