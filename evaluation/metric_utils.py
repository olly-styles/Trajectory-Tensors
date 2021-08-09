from scipy.ndimage.measurements import center_of_mass
import numpy as np
from skimage.feature import peak_local_max


def get_tensor_centroid(tensor):
    assert len(tensor.shape) == 2
    return center_of_mass(tensor)
