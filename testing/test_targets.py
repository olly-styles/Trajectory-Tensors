# Internal
from global_config.global_config import WHICH_TARGETS_PATH, WHEN_TARGETS_PATH, WHERE_TARGETS_PATH, DATA_PATH, NUM_DAYS

# External
import numpy as np
import os
import pytest


which_targets = []
when_targets = []
where_targets = []
departure_cameras = []

for day in range(1, 21):
    which_data_path = os.path.join(os.path.join(WHICH_TARGETS_PATH, "targets_day_" + str(day) + ".npy"))
    when_data_path = os.path.join(os.path.join(WHEN_TARGETS_PATH, "targets_day_" + str(day) + ".npy"))
    where_data_path = os.path.join(os.path.join(WHERE_TARGETS_PATH, "targets_day_" + str(day) + ".npy"))
    departure_camera_path = os.path.join(
        DATA_PATH, "inputs", "departure_cameras", "departure_cameras_day_" + str(day) + ".npy"
    )

    which_targets.append(np.load(which_data_path))
    when_targets.append(np.load(when_data_path))
    where_targets.append(np.load(where_data_path))
    departure_cameras.append(np.load(departure_camera_path))


def test_which_targets():
    """
    Test all inputs have the right number of targets and some have multiple targets
    """
    num_multi_targets = 0
    for which_targets_day in which_targets:
        # All inputs have a label
        assert np.all(which_targets_day.sum(axis=1) > 0)
        # No inputs have more than 3 targets
        assert np.all(which_targets_day.sum(axis=1) < 4)

        num_multi_targets += np.sum(which_targets_day.sum(axis=1) > 1)

    # Some days have multi-targets
    assert num_multi_targets > 0


def test_when_targets():
    """
    Test all inputs have the right number of when targets and some have multiple targets
    """
    num_multi_targets = 0
    for when_targets_day in when_targets:
        # All inputs have a label
        assert np.all(when_targets_day.sum(axis=1).sum(axis=1) > 0)

        num_multi_targets += np.sum((when_targets_day.sum(axis=2) > 1).sum(axis=1) > 1)

    # Some days have multi-targets
    assert num_multi_targets > 0


def test_where_targets():
    """
    Test all inputs have the right number of where targets and some have multiple targets
    """
    num_multi_targets = 0
    for where_targets_day in where_targets:
        # All inputs have a label
        assert np.all(where_targets_day.sum(axis=3).sum(axis=3).sum(axis=1).sum(axis=1) > 0)
        num_multi_targets += np.sum((where_targets_day.sum(axis=3).sum(axis=3).sum(axis=2) > 1).sum(axis=1) > 1)

    # Some days have multi-targets
    assert num_multi_targets > 0


def test_input_target_different():
    """
    Test departure and entrance cameras are different for all labels
    """
    for day in range(len(departure_cameras)):
        which_targets_day = which_targets[day]
        when_targets_day = when_targets[day]
        where_targets_day = where_targets[day]
        departure_cameras_day = departure_cameras[day]
        # Which
        for departure_camera, target in zip(departure_cameras_day, which_targets_day):
            entrance_cameras = np.argwhere(target == 1) + 1
            assert departure_camera not in entrance_cameras
        # When
        for departure_camera, when_target in zip(departure_cameras_day, when_targets_day):
            target = when_target.sum(axis=1) > 1
            entrance_cameras = np.argwhere(target == 1) + 1
            assert departure_camera not in entrance_cameras
        # Where
        for departure_camera, where_target in zip(departure_cameras_day, where_targets_day):
            target = where_target.sum(axis=3).sum(axis=2).sum(axis=1) > 1
            entrance_cameras = np.argwhere(target == 1) + 1
            assert departure_camera not in entrance_cameras


def test_all_same_target_cameras():
    """
    Checks the target cameras are the same for all 3 problem formulations
    """
    for day_num in range(1, NUM_DAYS):
        which_targets_day = which_targets[day_num]
        when_targets_day = when_targets[day_num]
        where_targets_day = where_targets[day_num]

        assert len(which_targets_day) == len(when_targets_day) == len(where_targets_day)

        for idx in range(len(which_targets_day)):
            which_target_cameras = np.argwhere(which_targets_day[idx])
            when_target_cameras = np.argwhere(when_targets_day[idx].sum(axis=1))
            where_target_cameras = np.argwhere(where_targets_day[idx].sum(axis=3).sum(axis=2).sum(axis=1))
            print(which_target_cameras, when_target_cameras, where_target_cameras)
            assert np.all(which_target_cameras == when_target_cameras)
            assert np.all(which_target_cameras == where_target_cameras)
            assert np.all(when_target_cameras == where_target_cameras)
