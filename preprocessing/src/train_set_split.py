# Internal
from global_config.global_config import (
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    COODINATE_TRAJECTORY_INPUTS_PATH,
    DEPARTURE_CAMERAS_PATH,
    NUM_DAYS,
    WHICH_TARGETS_PATH,
    WHEN_TARGETS_PATH,
    WHERE_TARGETS_PATH,
    DATA_PATH,
    NUM_CAMERAS,
    INPUT_TRAJECTORY_LENGTH,
    BASE_HEATMAP_SIZE,
    CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH,
    FUTURE_TRAJECTORY_LENGTH,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
)

# External
import numpy as np
import pandas as pd
import os
import pickle


def split_numpy_array(array, train_indexs, val_indexs, test_indexs):
    """
    Splits a numpy array into train, val, and test
    """
    train_data = array[train_indexs]
    val_data = array[val_indexs]
    test_data = array[test_indexs]
    return train_data, val_data, test_data


def save_numpy_data(save_path, fold, train_data, val_data, test_data):
    """
    Saves train val test numpy data
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "train_fold" + str(fold) + ".npy"), train_data)
    np.save(os.path.join(save_path, "val_fold" + str(fold) + ".npy"), val_data)
    np.save(os.path.join(save_path, "test_fold" + str(fold) + ".npy"), test_data)


def save_pickle_data(save_path, fold, train_data, val_data, test_data):
    """
    Saves train val test data as pickles
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "train_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(train_data, fp)
    with open(os.path.join(save_path, "val_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(val_data, fp)
    with open(os.path.join(save_path, "test_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(test_data, fp)


def get_inputs_and_which_targets():
    all_coordinate_trajectory_inputs = np.zeros((0, INPUT_TRAJECTORY_LENGTH, 4))
    all_which_targets = np.zeros((0, NUM_CAMERAS))
    all_day_numbers = np.zeros((0))
    all_departure_cameras = np.zeros((0))
    for day_num in range(1, NUM_DAYS + 1):
        day = "day_" + str(day_num)
        # Read data
        coordinate_trajectory_inputs = np.load(os.path.join(COODINATE_TRAJECTORY_INPUTS_PATH, "inputs_" + day + ".npy"))
        which_targets = np.load(os.path.join(WHICH_TARGETS_PATH, "targets_" + day + ".npy"))
        departure_cameras = np.load(os.path.join(DEPARTURE_CAMERAS_PATH, "departure_cameras_" + day + ".npy"))
        day_numbers = np.repeat(np.array([day_num]), len(which_targets))
        # Test all are same length
        assert len(day_numbers) == len(departure_cameras) == len(which_targets) == len(coordinate_trajectory_inputs)
        # Append to create complete dataset
        all_coordinate_trajectory_inputs = np.append(
            all_coordinate_trajectory_inputs, coordinate_trajectory_inputs, axis=0
        )
        all_which_targets = np.append(all_which_targets, which_targets, axis=0)
        all_departure_cameras = np.append(all_departure_cameras, departure_cameras)
        all_day_numbers = np.append(all_day_numbers, day_numbers)
    return all_coordinate_trajectory_inputs, all_departure_cameras, all_day_numbers, all_which_targets


def get_multi_target_inputs_and_targets(heatmap_scale):
    all_which_targets = []
    all_when_targets = []
    all_where_targets = []
    all_day_numbers = np.zeros((0))
    all_departure_cameras = []
    all_trajectory_tensors = []
    all_coordinate_trajectory_inputs = []
    for day in range(1, NUM_DAYS + 1):
        multi_target_tensor_data_path = os.path.join(
            os.path.join(
                DATA_PATH,
                "multi_target",
                "inputs",
                "trajectory_tensors",
                "size_" + str(BASE_HEATMAP_SIZE[0] * heatmap_scale),
                "trajectory_tensors_day_" + str(day) + ".pickle",
            )
        )
        which_target_path = os.path.join(
            DATA_PATH, "multi_target", "targets", "which", "targets_day_" + str(day) + ".pickle"
        )
        when_target_path = os.path.join(
            DATA_PATH, "multi_target", "targets", "when", "targets_day_" + str(day) + ".pickle"
        )
        where_target_path = os.path.join(
            DATA_PATH, "multi_target", "targets", "where", "targets_day_" + str(day) + ".pickle"
        )
        departure_cameras_path = os.path.join(
            DATA_PATH, "multi_target", "inputs", "departure_cameras", "departure_cameras_day_" + str(day) + ".pickle"
        )
        coordinate_trajectory_path = os.path.join(
            DATA_PATH, "multi_target", "inputs", "coordinate_trajectories", "inputs_day_" + str(day) + ".pickle"
        )

        try:
            with open(which_target_path, "rb") as fp:
                which_targets = pickle.load(fp)
                all_which_targets.append(which_targets)
            with open(when_target_path, "rb") as fp:
                when_targets = pickle.load(fp)
                all_when_targets.append(when_targets)
            with open(where_target_path, "rb") as fp:
                where_targets = pickle.load(fp)
                all_where_targets.append(where_targets)
            with open(departure_cameras_path, "rb") as fp:
                departure_cameras = pickle.load(fp)
                all_departure_cameras.append(departure_cameras)
            with open(multi_target_tensor_data_path, "rb") as fp:
                trajectory_tensors = pickle.load(fp)
                all_trajectory_tensors.append(trajectory_tensors)
            with open(coordinate_trajectory_path, "rb") as fp:
                coordinate_trajectories = pickle.load(fp)
                all_coordinate_trajectory_inputs.append(coordinate_trajectories)

            day_numbers = np.repeat(np.array([day]), len(which_targets))
            all_day_numbers = np.append(all_day_numbers, day_numbers)
            # Test all are same length
            assert (
                len(day_numbers)
                == len(departure_cameras)
                == len(which_targets)
                == len(when_targets)
                == len(where_targets)
                == len(trajectory_tensors)
                == len(coordinate_trajectories)
            )
        except ValueError:
            continue
    return (
        all_departure_cameras,
        all_day_numbers,
        all_which_targets,
        all_when_targets,
        all_where_targets,
        all_trajectory_tensors,
        all_coordinate_trajectory_inputs,
    )


def get_when_targets():
    all_when_targets = np.zeros((0, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH))
    for day_num in range(1, NUM_DAYS + 1):
        day = "day_" + str(day_num)
        when_targets = np.load(os.path.join(WHEN_TARGETS_PATH, "targets_" + day + ".npy"))
        all_when_targets = np.append(all_when_targets, when_targets, axis=0)
    return all_when_targets


def get_where_targets():
    all_where_targets = np.zeros((0, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH) + BASE_HEATMAP_SIZE, dtype=np.bool)
    for day_num in range(1, NUM_DAYS + 1):
        day = "day_" + str(day_num)
        where_targets = np.load(os.path.join(WHERE_TARGETS_PATH, "targets_" + day + ".npy"))
        all_where_targets = np.append(all_where_targets, where_targets, axis=0)
    return all_where_targets


def get_trajectory_tensor_data(heatmap_scale, multi_view):
    all_trajectory_tensors = np.zeros(
        (
            0,
            NUM_CAMERAS,
            INPUT_TRAJECTORY_LENGTH,
            BASE_HEATMAP_SIZE[0] * heatmap_scale,
            BASE_HEATMAP_SIZE[1] * heatmap_scale,
        ),
        dtype=np.bool,
    )
    for day_num in range(1, NUM_DAYS + 1):
        day = "day_" + str(day_num)
        print("Reading data for", day)
        # Read data
        if multi_view:
            trajectory_tensors = np.load(
                os.path.join(
                    DATA_PATH,
                    "inputs",
                    "trajectory_tensors",
                    "size_" + str(BASE_HEATMAP_SIZE[0] * heatmap_scale),
                    "trajectory_tensors_day_" + str(day_num) + ".npy",
                )
            )
        else:
            trajectory_tensors = np.load(
                os.path.join(
                    DATA_PATH,
                    "inputs",
                    "trajectory_tensors_single_view",
                    "size_" + str(BASE_HEATMAP_SIZE[0] * heatmap_scale),
                    "trajectory_tensors_day_" + str(day_num) + ".npy",
                )
            )
        all_trajectory_tensors = np.append(all_trajectory_tensors, trajectory_tensors, axis=0)
    return all_trajectory_tensors


def split_data_train_val_test(heatmap_scale, multi_view):
    (
        all_coordinate_trajectory_inputs,
        all_departure_cameras,
        all_day_numbers,
        all_which_targets,
    ) = get_inputs_and_which_targets()
    all_trajectory_tensors = get_trajectory_tensor_data(heatmap_scale, multi_view)
    all_when_targets = get_when_targets()
    all_where_targets = get_where_targets()

    for fold in range(1, 6):
        print("Splitting train-test fold", fold)
        # Get indexes
        train_indexs = np.argwhere(all_day_numbers % 5 != fold - 1).flatten()
        test_indexs = np.argwhere(all_day_numbers % 5 == fold - 1).flatten()
        # Must be rounded to multiple of 10 to ensure same sample with different offset
        # does not appear in both val and test
        val_indexs = test_indexs[0 : int(len(test_indexs) / 20) * 10]
        test_indexs = test_indexs[int(len(test_indexs) / 20) * 10 :]
        assert len(np.intersect1d(train_indexs, val_indexs)) == 0
        assert len(np.intersect1d(val_indexs, test_indexs)) == 0
        assert len(np.intersect1d(train_indexs, test_indexs)) == 0

        (
            train_coordinate_trajectory_inputs,
            val_coordinate_trajectory_inputs,
            test_coordinate_trajectory_inputs,
        ) = split_numpy_array(all_coordinate_trajectory_inputs, train_indexs, val_indexs, test_indexs)

        (train_trajectory_tensors, val_trajectory_tensors, test_trajectory_tensors) = split_numpy_array(
            all_trajectory_tensors, train_indexs, val_indexs, test_indexs
        )

        train_which_targets, val_which_targets, test_which_targets = split_numpy_array(
            all_which_targets, train_indexs, val_indexs, test_indexs
        )
        train_when_targets, val_when_targets, test_when_targets = split_numpy_array(
            all_when_targets, train_indexs, val_indexs, test_indexs
        )
        train_where_targets, val_where_targets, test_where_targets = split_numpy_array(
            all_where_targets, train_indexs, val_indexs, test_indexs
        )

        train_departure_cameras, val_departure_cameras, test_departure_cameras = split_numpy_array(
            all_departure_cameras.astype("uint8"), train_indexs, val_indexs, test_indexs
        )

        assert len(train_coordinate_trajectory_inputs) == len(train_which_targets) == len(train_departure_cameras)
        assert len(val_coordinate_trajectory_inputs) == len(val_which_targets) == len(val_departure_cameras)
        assert len(test_coordinate_trajectory_inputs) == len(test_which_targets) == len(test_departure_cameras)

        if multi_view:
            trajectory_tensor_save_path = CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH
        else:
            trajectory_tensor_save_path = CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH

        trajectory_tensor_save_path = os.path.join(
            trajectory_tensor_save_path, "size_" + str(BASE_HEATMAP_SIZE[0] * heatmap_scale)
        )
        if not os.path.exists(trajectory_tensor_save_path):
            os.makedirs(trajectory_tensor_save_path)

        save_numpy_data(
            trajectory_tensor_save_path, fold, train_trajectory_tensors, val_trajectory_tensors, test_trajectory_tensors
        )

        save_numpy_data(
            CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
            fold,
            train_coordinate_trajectory_inputs,
            val_coordinate_trajectory_inputs,
            test_coordinate_trajectory_inputs,
        )
        save_numpy_data(
            CROSS_VALIDATION_WHICH_TARGETS_PATH, fold, train_which_targets, val_which_targets, test_which_targets
        )
        save_numpy_data(
            CROSS_VALIDATION_WHEN_TARGETS_PATH, fold, train_when_targets, val_when_targets, test_when_targets
        )
        save_numpy_data(
            CROSS_VALIDATION_WHERE_TARGETS_PATH, fold, train_where_targets, val_where_targets, test_where_targets
        )
        save_numpy_data(
            CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
            fold,
            train_departure_cameras,
            val_departure_cameras,
            test_departure_cameras,
        )


def multi_target_split_data_train_val_test(heatmap_scale):
    (
        all_departure_cameras,
        all_day_numbers,
        all_which_targets,
        all_when_targets,
        all_where_targets,
        all_trajectory_tensors,
        all_coordinate_trajectory_inputs,
    ) = get_multi_target_inputs_and_targets(heatmap_scale)

    # Flatten
    all_which_targets = np.array([item for sublist in all_which_targets for item in sublist])
    all_when_targets = np.array([item for sublist in all_when_targets for item in sublist])
    all_where_targets = np.array([item for sublist in all_where_targets for item in sublist])
    all_departure_cameras = np.array([item for sublist in all_departure_cameras for item in sublist])
    all_trajectory_tensors = np.array([item for sublist in all_trajectory_tensors for item in sublist])
    all_coordinate_trajectory_inputs = np.array(
        [item for sublist in all_coordinate_trajectory_inputs for item in sublist]
    )

    for fold in range(1, 6):
        print("Splitting train-test fold", fold)
        # Get indexes
        train_indexs = np.argwhere(all_day_numbers % 5 != fold - 1).flatten()
        test_indexs = np.argwhere(all_day_numbers % 5 == fold - 1).flatten()
        # Must be rounded to multiple of 10 to ensure same sample with different offset
        # does not appear in both val and test
        val_indexs = test_indexs[0 : int(len(test_indexs) / 20) * 10]
        test_indexs = test_indexs[int(len(test_indexs) / 20) * 10 :]
        assert len(np.intersect1d(train_indexs, val_indexs)) == 0
        assert len(np.intersect1d(val_indexs, test_indexs)) == 0
        assert len(np.intersect1d(train_indexs, test_indexs)) == 0

        train_which_targets, val_which_targets, test_which_targets = split_numpy_array(
            all_which_targets, train_indexs, val_indexs, test_indexs
        )
        train_when_targets, val_when_targets, test_when_targets = split_numpy_array(
            all_when_targets, train_indexs, val_indexs, test_indexs
        )
        train_where_targets, val_where_targets, test_where_targets = split_numpy_array(
            all_where_targets, train_indexs, val_indexs, test_indexs
        )
        train_departure_cameras, val_departure_cameras, test_departure_cameras = split_numpy_array(
            all_departure_cameras, train_indexs, val_indexs, test_indexs
        )
        train_trajectory_tensors, val_trajectory_tensors, test_trajectory_tensors = split_numpy_array(
            all_trajectory_tensors, train_indexs, val_indexs, test_indexs
        )
        (
            train_coordinate_trajectory_inputs,
            val_coordinate_trajectory_inputs,
            test_coordinate_trajectory_inputs,
        ) = split_numpy_array(all_coordinate_trajectory_inputs, train_indexs, val_indexs, test_indexs)

        assert (
            len(train_which_targets)
            == len(train_departure_cameras)
            == len(train_trajectory_tensors)
            == len(train_when_targets)
            == len(train_where_targets)
            == len(train_coordinate_trajectory_inputs)
        )
        assert (
            len(val_which_targets)
            == len(val_departure_cameras)
            == len(val_trajectory_tensors)
            == len(val_when_targets)
            == len(val_where_targets)
            == len(val_coordinate_trajectory_inputs)
        )
        assert (
            len(test_which_targets)
            == len(test_departure_cameras)
            == len(test_trajectory_tensors)
            == len(test_when_targets)
            == len(test_where_targets)
            == len(test_coordinate_trajectory_inputs)
        )

        trajectory_tensor_save_path = os.path.join(
            DATA_PATH,
            "cross_validation",
            "multi_target",
            "inputs",
            "trajectory_tensors",
            "size_" + str(BASE_HEATMAP_SIZE[0] * heatmap_scale),
        )
        which_targets_save_path = os.path.join(
            DATA_PATH, "cross_validation", "multi_target", "targets", "which_targets"
        )
        when_targets_save_path = os.path.join(DATA_PATH, "cross_validation", "multi_target", "targets", "when_targets")
        where_targets_save_path = os.path.join(
            DATA_PATH, "cross_validation", "multi_target", "targets", "where_targets"
        )
        departure_cameras_save_path = os.path.join(
            DATA_PATH, "cross_validation", "multi_target", "inputs", "departure_cameras"
        )
        coordinate_trajectory_save_path = os.path.join(
            DATA_PATH, "cross_validation", "multi_target", "inputs", "coordinate_trajectories"
        )

        if not os.path.exists(trajectory_tensor_save_path):
            os.makedirs(trajectory_tensor_save_path)
        if not os.path.exists(which_targets_save_path):
            os.makedirs(which_targets_save_path)
        if not os.path.exists(when_targets_save_path):
            os.makedirs(when_targets_save_path)
        if not os.path.exists(where_targets_save_path):
            os.makedirs(where_targets_save_path)
        if not os.path.exists(departure_cameras_save_path):
            os.makedirs(departure_cameras_save_path)
        if not os.path.exists(coordinate_trajectory_save_path):
            os.makedirs(coordinate_trajectory_save_path)

        save_pickle_data(
            trajectory_tensor_save_path, fold, train_trajectory_tensors, val_trajectory_tensors, test_trajectory_tensors
        )
        save_pickle_data(which_targets_save_path, fold, train_which_targets, val_which_targets, test_which_targets)
        save_pickle_data(when_targets_save_path, fold, train_when_targets, val_when_targets, test_when_targets)
        save_pickle_data(where_targets_save_path, fold, train_where_targets, val_where_targets, test_where_targets)
        save_pickle_data(
            departure_cameras_save_path, fold, train_departure_cameras, val_departure_cameras, test_departure_cameras
        )
        save_pickle_data(
            coordinate_trajectory_save_path,
            fold,
            train_coordinate_trajectory_inputs,
            val_coordinate_trajectory_inputs,
            test_coordinate_trajectory_inputs,
        )
