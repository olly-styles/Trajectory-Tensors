# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH,
    INPUT_TRAJECTORY_LENGTH,
    DATA_PATH,
    FUTURE_TRAJECTORY_LENGTH,
)

# External
import numpy as np
import os
import pickle
from experiments.utils import ranks_to_scores, append_missing_cameras

for fold_num in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("Computing for fold ", fold_num)
    fold = str(fold_num)

    # Get train data from full dataset
    train_inputs = np.load(os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "train_fold" + fold + ".npy"))
    train_inputs = train_inputs.reshape(-1, 40)
    train_departure_cameras = np.load(
        os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "train_fold" + fold + ".npy")
    )

    # Get test data from multi-target subset
    test_input_path = os.path.join(
        DATA_PATH,
        "cross_validation",
        "multi_target",
        "inputs",
        "coordinate_trajectories",
        "test_fold" + str(fold_num) + ".pickle",
    )
    with open(test_input_path, "rb") as fp:
        test_inputs_list = pickle.load(fp)
    test_inputs = np.empty((0, INPUT_TRAJECTORY_LENGTH, 4))
    for sample in test_inputs_list:
        for trajectory in sample:
            trajectory = np.expand_dims(trajectory, axis=0)
            test_inputs = np.append(trajectory, test_inputs, axis=0)

    test_departure_cameras_path = os.path.join(
        DATA_PATH, "cross_validation", "multi_target", "inputs", "departure_cameras", "test_fold" + fold + ".pickle"
    )
    with open(test_departure_cameras_path, "rb") as fp:
        test_departure_camera_list = pickle.load(fp)
    test_departure_cameras = np.empty((0))
    for sample in test_departure_camera_list:
        for target in sample:
            test_departure_cameras = np.append(target, test_departure_cameras)

    # Get targets
    train_targets = np.load(os.path.join(CROSS_VALIDATION_WHEN_TARGETS_PATH, "train_fold" + fold + ".npy"))

    test_targets_path = os.path.join(
        DATA_PATH, "cross_validation", "multi_target", "targets", "when_targets", "test_fold" + fold + ".pickle"
    )
    with open(test_targets_path, "rb") as fp:
        test_targets_list = pickle.load(fp)

    test_targets = np.empty((0, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH))
    for sample in test_targets_list:
        for target in sample:
            target = target.reshape(1, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH)
            test_targets = np.append(target, test_targets, axis=0)

    all_predictions = np.zeros((len(test_targets), NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH))

    for row in range(len(test_inputs)):
        test_departure_camera = test_departure_cameras[row]
        test_trajectory = test_inputs[row]
        cam_inputs = train_inputs[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()
        cam_labels = train_targets[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()
        # This camera is not in the train set
        if len(cam_inputs) == 0:
            top_prediction = np.zeros((NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH))
        else:
            test_trajectory = test_trajectory.reshape(-1, 40)
            idx = np.argsort(np.abs(cam_inputs - test_trajectory).sum(axis=1))[0]
            top_prediction = cam_labels[idx]
        all_predictions[row] = top_prediction

    if not os.path.exists(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH):
        os.makedirs(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH)
    np.save(
        os.path.join(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH, "multi_target_test_fold" + fold + ".npy"),
        all_predictions,
    )
