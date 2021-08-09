import numpy as np
from scipy.ndimage import gaussian_filter


def ranks_to_scores(predictions, num_ranks):
    """
    Converts ranked camera predictions into scores
    """
    unordered_scores = np.tile(np.arange(0, num_ranks) / (num_ranks - 1), (len(predictions), 1))
    scores = np.zeros(unordered_scores.shape)
    for row in range(len(unordered_scores)):
        scores[row] = unordered_scores[row][predictions[row][::-1].argsort()]
    return scores


def append_missing_cameras(predictions, num_cameras):
    """
    If any cameras are missing from the predictions, these are appended
    """
    all_cameras = np.arange(1, num_cameras + 1)
    missing_cameras = np.setdiff1d(all_cameras, predictions)
    return np.append(predictions, missing_cameras)


def smooth_trajectory_tensor(trajectory_tensor, sigma):
    smoothed_trajectory_tensor = trajectory_tensor.copy().astype(np.float32)
    non_zero_camera_indexes = np.argwhere(trajectory_tensor.sum(axis=1).sum(axis=1).sum(axis=1) > 0)
    for non_zero_index in non_zero_camera_indexes:
        smoothed_trajectory_tensor[non_zero_index[0]] = gaussian_filter(
            trajectory_tensor[non_zero_index[0]].copy().astype(np.float32), sigma=sigma
        )
    return smoothed_trajectory_tensor


def normalize_trajectory_tensor(trajectory_tensor):
    # The will result in some divide by 0 errors which are supressed.
    with np.errstate(divide="ignore", invalid="ignore"):
        trajectory_tensor = (
            trajectory_tensor
            / trajectory_tensor.max(axis=1).max(axis=1).max(axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
        )
        trajectory_tensor = np.nan_to_num(trajectory_tensor)
    return trajectory_tensor
