from neuralpredictors.data.transforms import StaticTransform
from torchvision.transforms import GaussianBlur
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from scipy.ndimage.filters import gaussian_filter



class StaticGaussianBlur(StaticTransform):
    def __call__(self, x):
        sigma = 3
        x.images = gaussian_filter(x.images, sigma=sigma)
        return x
