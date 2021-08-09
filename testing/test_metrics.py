from evaluation.mctf_metrics import (
    get_siou_when,
    get_siou_where,
    get_dataset_siou_when,
    get_dataset_siou_where,
    get_tensor_ade,
    get_tensor_fde,
)
import numpy as np
import scipy.ndimage as ndimage


def test_siou_when():
    gt_data = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    pred_data = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert get_siou_when(gt_data, pred_data) == 0.5

    gt_data = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    pred_data = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    assert get_siou_when(gt_data, pred_data) == 1

    gt_data = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    pred_data = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    assert get_siou_when(gt_data, pred_data) == (2 / 3)

    gt_data = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    pred_data = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    assert get_siou_when(gt_data, pred_data) == 0


def test_dataset_siou_when():
    gt_data = np.array([[[0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 0, 0]]])
    pred_data = np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 0, 0]]])
    assert get_dataset_siou_when(gt_data, pred_data) == 0.75

    gt_data = np.array([[[0, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]])
    pred_data = np.array([[[0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]]])
    assert get_dataset_siou_when(gt_data, pred_data) == (1 / 3)

    gt_data = np.array([[[0, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]])
    pred_data = np.array([[[0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]]])
    SIOUs = get_dataset_siou_when(gt_data, pred_data, return_mean=False)
    assert SIOUs.shape == (2,)
    assert SIOUs[0] == 2 / 3
    assert SIOUs[1] == 0


def test_siou_where():
    gt_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )

    assert get_siou_where(gt_data, pred_data) == 1

    gt_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )

    assert get_siou_where(gt_data, pred_data) == 0
    gt_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 0, 1], [0, 0, 1], [0, 0, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    assert get_siou_where(gt_data, pred_data) == 0.75
    gt_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            [[[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0.5]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    assert get_siou_where(gt_data, pred_data) == 0.625


def test_tensor_ade():
    gt_data = np.array(
        [
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
        ]
    )
    grid_cell_size = 10
    assert get_tensor_ade(gt_data, pred_data, grid_cell_size) == 7.5


def test_tensor_fde():
    gt_data = np.array(
        [
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
        ]
    )
    grid_cell_size = 10
    assert get_tensor_fde(gt_data, pred_data, grid_cell_size) == 0
    gt_data = np.array(
        [
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1], [0, 0, 1]]],
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
        ]
    )
    grid_cell_size = 10
    assert get_tensor_fde(gt_data, pred_data, grid_cell_size) == 10
    gt_data = np.array(
        [
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ]
    )
    pred_data = np.array(
        [
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1], [0, 0, 1]]],
            [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
        ]
    )
    grid_cell_size = 10
    assert get_tensor_fde(gt_data, pred_data, grid_cell_size) == 5
