# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH,
    FUTURE_TRAJECTORY_LENGTH,
)

# External
import numpy as np
import os
from experiments.utils import ranks_to_scores, append_missing_cameras

for fold_num in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("Computing for fold ", fold_num)
    fold = str(fold_num)

    # Get coordinate trajectories
    train_inputs = np.load(os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "train_fold" + fold + ".npy"))
    test_inputs = np.load(os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "test_fold" + fold + ".npy"))
    train_inputs = train_inputs.reshape(-1, 40)
    test_inputs = test_inputs.reshape(-1, 40)

    # Get departure cameras
    train_departure_cameras = np.load(
        os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "train_fold" + fold + ".npy")
    )
    test_departure_cameras = np.load(os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "test_fold" + fold + ".npy"))

    # Get targets
    train_targets = np.load(os.path.join(CROSS_VALIDATION_WHEN_TARGETS_PATH, "train_fold" + fold + ".npy"))

    all_predictions = np.zeros((len(test_inputs), NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH), dtype="uint8")

    for row in range(len(test_inputs)):
        test_departure_camera = test_departure_cameras[row]
        test_trajectory = test_inputs[row]
        cam_inputs = train_inputs[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()
        cam_labels = train_targets[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()
        # This camera is not in the train set
        if len(cam_inputs) == 0:
            top_prediction = np.zeros((NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH))
        else:
            idx = np.argsort(np.abs(cam_inputs - test_trajectory).sum(axis=1))[0]
            top_prediction = cam_labels[idx]
        all_predictions[row] = top_prediction

    if not os.path.exists(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH):
        os.makedirs(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH)
    np.save(
        os.path.join(CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH, "test_fold" + fold + ".npy"), all_predictions
    )
