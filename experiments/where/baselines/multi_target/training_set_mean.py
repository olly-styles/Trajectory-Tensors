# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH,
    DATA_PATH,
    FUTURE_TRAJECTORY_LENGTH,
    BASE_HEATMAP_SIZE,
)

# External
import numpy as np
import os
import pickle

for fold_num in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("Computing for fold ", fold_num)
    fold = str(fold_num)

    # Get departure cameras from complete dataset
    train_departure_cameras = np.load(
        os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "train_fold" + fold + ".npy")
    )
    # Get test departure cameras from multi-target subset
    test_departure_cameras_path = os.path.join(
        DATA_PATH, "cross_validation", "multi_target", "inputs", "departure_cameras", "test_fold" + fold + ".pickle"
    )

    with open(test_departure_cameras_path, "rb") as fp:
        test_departure_camera_list = pickle.load(fp)

    # Get targets
    train_targets = np.load(os.path.join(CROSS_VALIDATION_WHERE_TARGETS_PATH, "train_fold" + fold + ".npy"))
    test_targets_path = os.path.join(
        DATA_PATH, "cross_validation", "multi_target", "targets", "where_targets", "test_fold" + fold + ".pickle"
    )
    with open(test_targets_path, "rb") as fp:
        test_targets_list = pickle.load(fp)

    test_departure_cameras = np.empty((0))
    for sample in test_departure_camera_list:
        for target in sample:
            test_departure_cameras = np.append(target, test_departure_cameras)
    test_targets = np.empty((0, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1]))
    for sample in test_targets_list:
        for target in sample:
            target = target.reshape(
                1, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1]
            )
            test_targets = np.append(target, test_targets, axis=0)
    all_predictions = np.zeros(
        (len(test_targets), NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1]),
        dtype="float",
    )
    for camera in range(1, NUM_CAMERAS + 1):
        camera_targets = train_targets[train_departure_cameras == camera]
        # This camera has not been observed in the train set
        if len(camera_targets) == 0:
            mean_predictions = np.ones(
                (NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1])
            ) * (1 / (FUTURE_TRAJECTORY_LENGTH * BASE_HEATMAP_SIZE[0] * BASE_HEATMAP_SIZE[1]))
        else:
            mean_predictions = np.mean(camera_targets, axis=0)
        camera_indexes = np.argwhere(test_departure_cameras == camera).squeeze()
        all_predictions[camera_indexes] = np.repeat(mean_predictions[np.newaxis, :], len(camera_indexes), axis=0)

    if not os.path.exists(CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH):
        os.makedirs(CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH)
    np.save(
        os.path.join(CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH, "multi_target_test_fold" + fold + ".npy"),
        all_predictions,
    )
