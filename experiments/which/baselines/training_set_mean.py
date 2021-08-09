# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    NUM_CAMERAS,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH,
)

# External
import numpy as np
import os

for fold_num in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("Computing for fold ", fold_num)
    fold = str(fold_num)

    # Get departure cameras
    train_departure_cameras = np.load(
        os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "train_fold" + fold + ".npy")
    )
    test_departure_cameras = np.load(os.path.join(CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH, "test_fold" + fold + ".npy"))

    # Get targets
    train_targets = np.load(os.path.join(CROSS_VALIDATION_WHICH_TARGETS_PATH, "train_fold" + fold + ".npy"))
    test_targets = np.load(os.path.join(CROSS_VALIDATION_WHICH_TARGETS_PATH, "test_fold" + fold + ".npy"))

    all_predictions = np.zeros((len(test_targets), NUM_CAMERAS))

    for camera in range(1, NUM_CAMERAS + 1):
        camera_targets = train_targets[train_departure_cameras == camera]
        # This camera has not been observed in the train set
        if len(camera_targets) == 0:
            mean_predictions = np.ones(NUM_CAMERAS) * (1 / NUM_CAMERAS)
        else:
            mean_predictions = np.mean(camera_targets, axis=0)
        camera_indexes = np.argwhere(test_departure_cameras == camera).squeeze()
        all_predictions[camera_indexes] = np.repeat(mean_predictions[np.newaxis, :], len(camera_indexes), axis=0)

    if not os.path.exists(CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH):
        os.makedirs(CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH)
    np.save(os.path.join(CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH, "test_fold" + fold + ".npy"), all_predictions)
