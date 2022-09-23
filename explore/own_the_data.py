"""
So, we got the zip archive, right? Let's see what it consists from 
There is a folder for each of seven mice.
Each folder has data and meta. Meta is mostly params for normalization (mean/std)

In data there are images and responses, plus supplemental data about pupil centers and behavior
Each sample is presented as separate *.npy file
the response look gamma decay
images are grayscale, each croppped to 144x256
pupil is two number in strange range (like 150-250 for x,y)
behavior is three floats not clear without documentation
"""
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt

basic_path = Path('data')
basic_path = basic_path / 'static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6'
data_path = basic_path / 'data'
response_path = data_path / 'responses'
image_path = data_path / 'images'
pupil_path = data_path / 'pupil_center'
behavior_path = data_path / 'behavior'

for i in range(20):
    behave = np.load(behavior_path / f'{i}.npy')
    print(*behave, sep='\t\t')


# centers = np.stack([np.load(pupil_path / f"{i}.npy") for i in range(5000)])
#
# print(np.min(centers[:, 0]))
# print(np.max(centers[:, 0]))
# print(np.min(centers[:, 1]))
# print(np.max(centers[:, 1]))
#
# plt.hist(centers[:, 1], bins=20)
# plt.show()


# for i in range(50):
#     image = np.load(image_path / f'{i}.npy')
#     print(image.shape)
#     plt.imshow(image[0], cmap='gray')
#     plt.show()

