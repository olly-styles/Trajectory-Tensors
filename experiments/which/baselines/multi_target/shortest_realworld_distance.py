# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CAMERA_DISTANCE_RANKING,
    CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH,
    DATA_PATH,
)
from experiments.utils import ranks_to_scores

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
    test_targets_path = os.path.join(
        DATA_PATH, "cross_validation", "multi_target", "targets", "which_targets", "test_fold" + fold + ".pickle"
    )
    with open(test_targets_path, "rb") as fp:
        test_targets_list = pickle.load(fp)

    with open(test_departure_cameras_path, "rb") as fp:
        test_departure_camera_list = pickle.load(fp)
    test_departure_cameras = np.empty((0))
    for sample in test_departure_camera_list:
        for target in sample:
            test_departure_cameras = np.append(target, test_departure_cameras)
    test_targets = np.empty((0, 15))
    for sample in test_targets_list:
        for target in sample:
            target = target.reshape(1, 15)
            test_targets = np.append(target, test_targets, axis=0)

    all_predictions = np.zeros((len(test_targets), NUM_CAMERAS), dtype="uint8")
    for camera in range(1, NUM_CAMERAS + 1):
        ranked_camera_predictions = CAMERA_DISTANCE_RANKING[camera - 1]
        ranked_camera_predictions = np.append(ranked_camera_predictions, camera)
        camera_indexes = np.argwhere(test_departure_cameras == camera).squeeze()
        if len(camera_indexes) != 0:
            all_predictions[camera_indexes] = np.repeat(
                ranked_camera_predictions[np.newaxis, :], len(camera_indexes), axis=0
            )
    scores = ranks_to_scores(all_predictions, NUM_CAMERAS)
    if not os.path.exists(CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH):
        os.makedirs(CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH)

    np.save(
        os.path.join(CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH, "multi_target_test_fold" + fold + ".npy"),
        scores,
    )
