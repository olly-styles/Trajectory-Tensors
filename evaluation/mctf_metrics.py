import numpy as np
from sklearn.metrics import average_precision_score
from scipy.spatial import distance
from evaluation.metric_utils import get_tensor_centroid
from global_config.global_config import BASE_HEATMAP_SIZE


class Error(Exception):
    """Base class for other exceptions"""

    pass


class ValueTooSmallError(Error):
    """Raised when the input value is too small"""

    pass


class ValueTooLargeError(Error):
    """Raised when the input value is too large"""

    pass


def get_ap(targets, predictions):
    return average_precision_score(targets.flatten(), predictions.flatten())


def get_dataset_siou_when(targets, predictions, return_mean=True):
    """
    Input shapes: (N, camera, timestep)
    Returns: WHEN SIOU mean or list of SIOUs for the N data samples
    """

    all_sious = np.array(list(map(lambda gt, pred: get_siou_when(gt, pred), targets, predictions)))

    if return_mean:
        return np.mean(all_sious)
    else:
        return all_sious


def get_siou_when(label, prediction):
    """
    Input shapes: (camera, timestep)
    Returns: WHEN SIOU
    """
    true_positive_cameras = np.argwhere(label.sum(axis=1) > 0).flatten()
    sious = []
    for camera in true_positive_cameras:
        timestep_SIOUs = []
        soft_intersection = np.sum(prediction[camera] * label[camera])
        soft_union = ((np.sum(prediction[camera])) + (np.sum(label[camera]))) - soft_intersection

        if soft_union == 0:
            sious.append(0)
        else:
            siou = soft_intersection / soft_union
            sious.append(siou)

        try:
            if (siou) > 1:
                raise SIOUTooLargeError
            if (siou) < 0:
                raise SIOUTooSmallError
        except SIOUTooLargeError:
            print("SIOU is larger than 1")
        except SIOUTooSmallError:
            print("SIOU is smaller than 0")

    return np.mean(sious)


def get_siou_where(label, prediction):
    """
    Input shapes: (camera, timestep, height, width)
    Returns: WHERE SIOU
    """

    true_positive_cameras = np.argwhere(label.sum(axis=3).sum(axis=2).sum(axis=1) > 0).flatten()
    sious = []
    for camera in true_positive_cameras:
        timestep_SIOUs = []
        true_positive_timesteps = np.argwhere(label[camera].sum(axis=2).sum(axis=1) > 0).flatten()
        for timestep in true_positive_timesteps:
            soft_intersection = np.sum(prediction[camera, timestep] * label[camera, timestep])
            soft_union = (
                (np.sum(prediction[camera, timestep])) + (np.sum(label[camera, timestep]))
            ) - soft_intersection

            if soft_union == 0:
                sious.append(0)
            else:
                siou = soft_intersection / soft_union
                sious.append(siou)
                try:
                    if (siou) > 1:
                        raise SIOUTooLargeError
                    if (siou) < 0:
                        raise SIOUTooSmallError
                except SIOUTooLargeError:
                    print("SIOU is larger than 1")
                except SIOUTooSmallError:
                    print("SIOU is smaller than 0")

    return np.mean(sious)


def get_dataset_siou_where(targets, predictions, return_mean=True):
    """
    Input shapes: (N, camera, timestep, height, width)
    Returns: WHEN SIOU mean or list of SIOUs for the N data samples
    """

    all_sious = np.array(list(map(lambda gt, pred: get_siou_where(gt, pred), targets, predictions)))

    if return_mean:
        return np.mean(all_sious)
    else:
        return all_sious


def get_tensor_ade(target_tensor, predicted_tensor, grid_cell_size):
    """
    Input shapes: (camera, timestep, height, width)
    Returns: Tensor ADE
    """
    assert len(target_tensor.shape) == 4
    assert len(predicted_tensor.shape) == 4
    true_positive_cameras = np.argwhere(target_tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 0).flatten()
    camera_ADEs = []
    for camera in true_positive_cameras:
        timestep_DEs = []
        true_positive_timesteps = np.argwhere(target_tensor[camera].sum(axis=2).sum(axis=1) > 0).flatten()
        for timestep in true_positive_timesteps:
            target_centroid = get_tensor_centroid(target_tensor[camera, timestep])
            predicted_centroid = get_tensor_centroid(predicted_tensor[camera, timestep])
            # Take the mid point if not defined
            if np.isnan(predicted_centroid[0]) or np.isnan(predicted_centroid[1]):
                predicted_centroid = (BASE_HEATMAP_SIZE[0] / 2, BASE_HEATMAP_SIZE[1] / 2)
            timestep_DEs.append(distance.euclidean(target_centroid, predicted_centroid) * grid_cell_size)
        camera_ADEs.append(np.mean(timestep_DEs))
    return np.mean(camera_ADEs)


def get_tensor_fde(target_tensor, predicted_tensor, grid_cell_size):
    """
    Input shapes: (camera, timestep, height, width)
    Returns: Tensor FDE
    """
    true_positive_cameras = np.argwhere(target_tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 0).flatten()
    camera_FDEs = []
    for camera in true_positive_cameras:
        camera_last_timestep = np.argwhere(target_tensor[camera].sum(axis=2).sum(axis=1) > 0)[-1][-1]
        target_centroid = get_tensor_centroid(target_tensor[camera, camera_last_timestep])
        predicted_centroid = get_tensor_centroid(predicted_tensor[camera, camera_last_timestep])
        # Take the mid point if not defined
        if np.isnan(predicted_centroid[0]) or np.isnan(predicted_centroid[1]):
            predicted_centroid = (BASE_HEATMAP_SIZE[0] / 2, BASE_HEATMAP_SIZE[1] / 2)
        camera_FDEs.append(distance.euclidean(target_centroid, predicted_centroid) * grid_cell_size)
    return np.mean(camera_FDEs)


def get_dataset_tensor_ade(targets, predictions, grid_cell_size, return_mean=True):
    """
    Input shapes: (N, camera, timestep, height, width)
    Returns: WHEN SIOU mean or list of SIOUs for the N data samples
    """
    predictions = np.nan_to_num(predictions)

    all_tensor_ades = np.array(
        list(map(lambda gt, pred: get_tensor_ade(gt, pred, grid_cell_size), targets, predictions))
    )

    if return_mean:
        return np.mean(all_tensor_ades)
    else:
        return all_tensor_ades


def get_dataset_tensor_fde(targets, predictions, grid_cell_size, return_mean=True):
    """
    Input shapes: (N, camera, timestep, height, width)
    Returns: mean or list of FDEs for the N data samples
    """
    predictions = np.nan_to_num(predictions)

    all_tensor_fdes = np.array(
        list(map(lambda gt, pred: get_tensor_fde(gt, pred, grid_cell_size), targets, predictions))
    )

    if return_mean:
        return np.mean(all_tensor_fdes)
    else:
        return all_tensor_fdes
