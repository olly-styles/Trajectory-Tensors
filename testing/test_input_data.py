# Internal
from global_config.global_config import (
    DATA_PATH,
    INPUT_TRAJECTORY_LENGTH,
    NUM_CAMERAS,
    BASE_HEATMAP_SIZE,
    WHICH_TARGETS_PATH,
    WHEN_TARGETS_PATH,
    WHERE_TARGETS_PATH,
    NUM_DAYS,
)

# External
import numpy as np
import os
import pytest

coordinate_trajectories = []
trajectory_tensors = {}
trajectory_tensors["single_view"] = {}
trajectory_tensors["multi_view"] = {}
which_targets = []
when_targets = []
where_targets = []


for day in range(1, 21):
    coordinate_trajectory_path = os.path.join(
        os.path.join(DATA_PATH, "inputs", "input_coordinate_trajectories", "inputs_day_" + str(day) + ".npy")
    )
    single_view_data_path = os.path.join(
        os.path.join(
            DATA_PATH,
            "inputs",
            "trajectory_tensors_single_view",
            "size_9",
            "trajectory_tensors_day_" + str(day) + ".npy",
        )
    )
    multi_view_tensor_data_path = os.path.join(
        os.path.join(DATA_PATH, "inputs", "trajectory_tensors", "size_9", "trajectory_tensors_day_" + str(day) + ".npy")
    )
    which_target_path = os.path.join(WHICH_TARGETS_PATH, "targets_day_" + str(day) + ".npy")
    when_target_path = os.path.join(os.path.join(WHEN_TARGETS_PATH, "targets_day_" + str(day) + ".npy"))
    where_target_path = os.path.join(os.path.join(WHERE_TARGETS_PATH, "targets_day_" + str(day) + ".npy"))

    coordinate_trajectories.append(np.load(coordinate_trajectory_path))
    which_targets.append(np.load(which_target_path))
    when_targets.append(np.load(when_target_path))
    where_targets.append(np.load(where_target_path))

    trajectory_tensors["multi_view"]["day_" + str(day)] = np.load(multi_view_tensor_data_path)
    trajectory_tensors["single_view"]["day_" + str(day)] = np.load(single_view_data_path)


@pytest.mark.parametrize("data", coordinate_trajectories)
def test_coordinate_trajectories_range(data):
    """
    Test the processed coordinate trajectories are between 0 and 1
    """
    assert data.max() <= 1
    assert data.min() >= 0


@pytest.mark.parametrize("data", coordinate_trajectories)
def test_coordinate_trajectory_format(data):
    """
    Test that bounding boxes are of the format x1, y1, x2, y2
    """
    assert np.sum(data[:, :, 0] > data[:, :, 2]) == 0
    assert np.sum(data[:, :, 1] > data[:, :, 3]) == 0


def test_trajectory_tensor_format():
    for view in trajectory_tensors:
        for day in trajectory_tensors[view]:
            tensor = trajectory_tensors[view][day]
            print(view, day, tensor.shape)
            if len(tensor.shape) == 1:
                continue
            assert tensor.shape[1:] == ((NUM_CAMERAS, INPUT_TRAJECTORY_LENGTH) + BASE_HEATMAP_SIZE)


def test_multi_view_trajectory_tensor():
    """
    Tests that the number of multi-view trajectory tensors with detections in multiple camera views
    is more than 0
    """
    for day in trajectory_tensors["multi_view"]:
        tensor = trajectory_tensors["multi_view"][day]
        assert np.sum((tensor.sum(axis=4).sum(axis=3).sum(axis=2) > 1).sum(axis=1) > 1) > 0


def test_three_or_more_view_trajectory_tensor():
    """
    Tests that the number of multi-view trajectory tensors with detections in more than 2 camera views
    is more than 0
    """
    for day in trajectory_tensors["multi_view"]:
        tensor = trajectory_tensors["multi_view"][day]
        assert np.sum((tensor.sum(axis=4).sum(axis=3).sum(axis=2) > 1).sum(axis=1) > 2) > 0


def test_no_more_than_4_view_trajectory_tensor():
    """
    Tests that the number of multi-view trajectory tensors with detections in more than 4 camera views
    is 0
    """
    for day in trajectory_tensors["multi_view"]:
        tensor = trajectory_tensors["multi_view"][day]
        assert np.sum((tensor.sum(axis=4).sum(axis=3).sum(axis=2) > 1).sum(axis=1) > 4) == 0


def test_trajectory_tensor_departure_entrance_different_cameras():
    for view in ["single_view", "multi_view"]:
        for day_num in range(1, len(trajectory_tensors[view]) + 1):
            day = "day_" + str(day_num)
            tensors = trajectory_tensors[view][day]
            for idx in range(tensors.shape[0]):
                which_target_cameras = np.argwhere(which_targets[day_num - 1][idx])
                when_target_cameras = np.argwhere(when_targets[day_num - 1][idx].sum(axis=1))
                where_target_cameras = np.argwhere(where_targets[day_num - 1][idx].sum(axis=3).sum(axis=2).sum(axis=1))
                tensor = tensors[idx]
                non_zero_inputs = np.argwhere(tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 1)
                for non_zero_input in non_zero_inputs:
                    assert non_zero_input not in which_target_cameras
                    assert non_zero_input not in when_target_cameras
                    assert non_zero_input not in where_target_cameras


def test_single_view_multi_view_same_camera():
    """Tests that the input camera in a single view is contained in the multi view cameras"""
    for day_num in range(1, NUM_DAYS + 1):
        print(day_num)
        day = "day_" + str(day_num)
        single_view_tensors = trajectory_tensors["single_view"][day]
        multi_view_tensors = trajectory_tensors["multi_view"][day]
        assert single_view_tensors.shape[0] == multi_view_tensors.shape[0]
        for idx in range(multi_view_tensors.shape[0]):
            single_view_tensor = single_view_tensors[idx]
            multi_view_tensor = multi_view_tensors[idx]
            single_view_camera = np.argwhere(single_view_tensor.sum(axis=3).sum(axis=2).sum(axis=1))
            multi_view_cameras = np.argwhere(multi_view_tensor.sum(axis=3).sum(axis=2).sum(axis=1))
            assert len(single_view_camera) == 1
            assert single_view_camera in multi_view_cameras
