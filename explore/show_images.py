from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt


base_dir = Path('/home/kirill/apps/sensorium/notebooks/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6/data/images')
print(os.listdir(base_dir))


for i in range(20):
    img = np.load(base_dir / f'{i}.npy')
    plt.imshow(img[0], cmap='gray')
    plt.show()