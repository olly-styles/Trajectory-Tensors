# Internal
from experiments.utils import smooth_trajectory_tensor, normalize_trajectory_tensor
from evaluation.metric_utils import get_tensor_centroid

# External
import numpy as np


tensor = np.array([[[[1, 1], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]])


def test_correct_dimentions_smoothed():
    sigma = 1
    smoothed_tensor = smooth_trajectory_tensor(tensor, sigma)

    # Camera dimention
    assert np.all(tensor[0] != smoothed_tensor[0])
    assert np.all(tensor[1] == smoothed_tensor[1])

    # Time dimention
    assert np.all(tensor[0][1] != smoothed_tensor[0][1])


def test_normalized_tensor_range_and_shape():
    sigma = 1
    smoothed_tensor = smooth_trajectory_tensor(tensor, sigma)
    normalized_tensor = normalize_trajectory_tensor(smoothed_tensor)

    assert smoothed_tensor.shape == normalized_tensor.shape
    assert normalized_tensor.max() == 1
    assert normalized_tensor.min() == 0


def test_get_tensor_centroid():
    tensor = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    assert get_tensor_centroid(tensor) == (1, 1)
    tensor = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 2]])
    assert get_tensor_centroid(tensor) == (1.1, 1.6)
