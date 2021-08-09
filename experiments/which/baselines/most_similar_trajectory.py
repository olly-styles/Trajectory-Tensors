# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH,
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
    train_targets = np.load(os.path.join(CROSS_VALIDATION_WHICH_TARGETS_PATH, "train_fold" + fold + ".npy"))
    # Un one hot encode
    train_targets = np.argmax(train_targets, axis=1)
    all_predictions = np.zeros((len(test_inputs), 15), dtype="uint8")

    for row in range(len(test_inputs)):
        test_departure_camera = test_departure_cameras[row]
        test_trajectory = test_inputs[row]
        cam_inputs = train_inputs[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()
        cam_labels = train_targets[np.argwhere(train_departure_cameras == test_departure_camera)].squeeze()

        idx = np.argsort(np.abs(cam_inputs - test_trajectory).sum(axis=1))
        sorted_predictions = cam_labels[idx]

        _, idx = np.unique(sorted_predictions, return_index=True)
        ranked_predictions = sorted_predictions[np.sort(idx)] + 1
        ranked_predictions = append_missing_cameras(ranked_predictions, NUM_CAMERAS)
        all_predictions[row] = ranked_predictions
        scores = ranks_to_scores(all_predictions, NUM_CAMERAS)
    if not os.path.exists(CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH):
        os.makedirs(CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH)
    np.save(os.path.join(CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH, "test_fold" + fold + ".npy"), scores)
