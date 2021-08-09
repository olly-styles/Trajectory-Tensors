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
import pickle

trajectory_tensors = []
which_targets = []
when_targets = []
where_targets = []
departure_cameras = []

for day in range(1, 21):
    multi_target_tensor_data_path = os.path.join(
        os.path.join(
            DATA_PATH,
            "multi_target",
            "inputs",
            "trajectory_tensors",
            "size_9",
            "trajectory_tensors_day_" + str(day) + ".pickle",
        )
    )
    which_target_path = os.path.join(
        DATA_PATH, "multi_target", "targets", "which", "targets_day_" + str(day) + ".pickle"
    )
    when_target_path = os.path.join(DATA_PATH, "multi_target", "targets", "when", "targets_day_" + str(day) + ".pickle")
    where_target_path = os.path.join(
        DATA_PATH, "multi_target", "targets", "where", "targets_day_" + str(day) + ".pickle"
    )
    departure_cameras_path = os.path.join(
        DATA_PATH, "multi_target", "inputs", "departure_cameras", "departure_cameras_day_" + str(day) + ".pickle"
    )
    try:
        with open(multi_target_tensor_data_path, "rb") as fp:
            trajectory_tensors.append(pickle.load(fp))
        with open(which_target_path, "rb") as fp:
            which_targets.append(pickle.load(fp))
        with open(when_target_path, "rb") as fp:
            when_targets.append(pickle.load(fp))
        with open(where_target_path, "rb") as fp:
            where_targets.append(pickle.load(fp))
        with open(departure_cameras_path, "rb") as fp:
            departure_cameras.append(pickle.load(fp))
    except ValueError:
        continue


def test_data_loaded_correctly():
    assert len(trajectory_tensors) > 0
    assert len(which_targets) > 0
    assert len(when_targets) > 0
    assert len(where_targets) > 0
    assert len(trajectory_tensors) == len(which_targets)


def inputs_and_targets_same_length():
    for one_day_tensors, one_day_targets in zip(trajectory_tensors, which_targets):
        assert len(one_day_tensors) == len(one_day_targets)
        for multi_target_input, multi_target_label in zip(one_day_tensors, one_day_targets):
            assert len(multi_target_input) == len(multi_target_label)


def test_trajectory_tensor_format():
    for one_day_tensor in trajectory_tensors:
        for multi_target_input in one_day_tensor:
            for tensor in multi_target_input:
                assert tensor.shape == ((NUM_CAMERAS, INPUT_TRAJECTORY_LENGTH) + BASE_HEATMAP_SIZE)


def test_multi_target_multi_view_trajectory_tensor():
    """
    Tests that the number of multi-target multi-view trajectory tensors with detections in multiple
    camera views is more than 0
    """
    total_multi_view = 0
    for one_day_tensor in trajectory_tensors:
        for multi_target_input in one_day_tensor:
            for tensor in multi_target_input:
                total_multi_view += np.sum((tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 1))
    assert total_multi_view > 0


def test_all_inputs_non_zero():
    for i, one_day_tensor in enumerate(trajectory_tensors):
        for j, multi_target_input in enumerate(one_day_tensor):
            for k, tensor in enumerate(multi_target_input):
                print(i, j, k)
                assert np.sum((tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 0)) > 0


def test_trajectory_tensor_departure_entrance_different_cameras():
    for one_day_tensor, one_day_targets in zip(trajectory_tensors, which_targets):
        for multi_target_input, multi_target_label in zip(one_day_tensor, one_day_targets):
            for tensor, target in zip(multi_target_input, multi_target_label):
                non_zero_inputs = np.argwhere(tensor.sum(axis=3).sum(axis=2).sum(axis=1) > 1)
                which_target = np.argwhere(target)
                for non_zero_input in non_zero_inputs:
                    assert non_zero_input not in which_target


def test_which_targets():
    """
    Test all inputs have the right number of targets and some have multiple targets
    """
    num_multi_targets = 0
    for one_day_targets in which_targets:
        for multi_target_label in one_day_targets:
            for target in multi_target_label:
                # All inputs have a label
                assert target.sum() > 0
                # No inputs have more than 3 targets
                assert target.sum() < 4


def test_input_target_different():
    """
    Test departure and entrance cameras are different for all labels
    """
    for one_day_which_targets, one_day_when_targets, one_day_where_targets, one_day_departure_cameras in zip(
        which_targets, when_targets, where_targets, departure_cameras
    ):
        for (
            multi_target_which_label,
            multi_target_when_label,
            multi_target_where_label,
            multi_target_departure_cameras,
        ) in zip(one_day_which_targets, one_day_when_targets, one_day_where_targets, one_day_departure_cameras):
            for which_target, when_target, where_target, all_departure_cameras in zip(
                multi_target_which_label,
                multi_target_when_label,
                multi_target_where_label,
                multi_target_departure_cameras,
            ):
                which_target_cameras = np.argwhere(which_target) + 1
                when_target_cameras = np.argwhere(when_target.sum(axis=1) > 1) + 1
                where_target_cameras = np.argwhere(where_target.sum(axis=3).sum(axis=2).sum(axis=1) > 1) + 1
                print(which_target_cameras, when_target_cameras, where_target_cameras)
                assert which_target_cameras == when_target_cameras == where_target_cameras
                assert all_departure_cameras not in which_target_cameras
                assert all_departure_cameras not in when_target_cameras
                assert all_departure_cameras not in where_target_cameras


def test_when_targets():
    """
    Test all inputs have the right number of when targets and some have multiple targets
    """
    num_multi_targets = 0
    for one_day_targets in when_targets:
        for multi_target_label in one_day_targets:
            for target in multi_target_label:
                assert np.sum((target.sum(axis=1) > 0)) > 0
                assert np.sum((target.sum(axis=1) > 0)) < 4


def test_where_targets():
    """
    Test all inputs have the right number of when targets and some have multiple targets
    """
    num_multi_targets = 0
    for one_day_targets in where_targets:
        for multi_target_label in one_day_targets:
            for target in multi_target_label:
                assert np.sum((target.sum(axis=3).sum(axis=2).sum(axis=1) > 0)) > 0
                assert np.sum((target.sum(axis=3).sum(axis=2).sum(axis=1) > 0)) < 4
