# Internal
from global_config.global_config import (
    CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
)

# External
import numpy as np
import os


def compute_hand_crafted_features(inputs, mean=None, std=None):
    hand_crafted_features = np.zeros((inputs.shape[0], 10))
    # Velocity
    hand_crafted_features[:, 0] = (inputs[:, -1, 2] - inputs[:, -1, 0]) - (inputs[:, 0, 2] - inputs[:, 0, 0])
    hand_crafted_features[:, 1] = (inputs[:, -1, 3] - inputs[:, -1, 1]) - (inputs[:, 0, 3] - inputs[:, 0, 1])
    # Acceleration
    start_velocity_x = (inputs[:, 1, 2] - inputs[:, 1, 0]) - (inputs[:, 0, 2] - inputs[:, 0, 0])
    end_velocity_x = (inputs[:, -1, 2] - inputs[:, -1, 0]) - (inputs[:, -2, 2] - inputs[:, -2, 0])
    start_velocity_y = (inputs[:, 1, 3] - inputs[:, 1, 1]) - (inputs[:, 0, 3] - inputs[:, 0, 1])
    end_velocity_y = (inputs[:, -1, 3] - inputs[:, -1, 1]) - (inputs[:, -2, 3] - inputs[:, -2, 1])
    hand_crafted_features[:, 2] = start_velocity_x - end_velocity_x
    hand_crafted_features[:, 3] = start_velocity_y - end_velocity_y
    # Scale
    hand_crafted_features[:, 4] = inputs[:, -1, 2] - inputs[:, -1, 0]
    hand_crafted_features[:, 5] = inputs[:, -1, 3] - inputs[:, -1, 1]
    # Exit location
    hand_crafted_features[:, 6] = inputs[:, -1, 0]
    hand_crafted_features[:, 7] = inputs[:, -1, 1]
    hand_crafted_features[:, 8] = inputs[:, -1, 2]
    hand_crafted_features[:, 9] = inputs[:, -1, 3]
    # Standardize
    if mean is None and std is None:
        mean = hand_crafted_features.mean(axis=0)
        std = hand_crafted_features.std(axis=0)
    hand_crafted_features = (hand_crafted_features - mean) / std
    return hand_crafted_features, mean, std


if __name__ == "__main__":
    for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
        print("Computing features for fold", fold)
        train_inputs = np.load(
            os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "train_fold" + str(fold) + ".npy")
        )
        val_inputs = np.load(
            os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "val_fold" + str(fold) + ".npy")
        )
        test_inputs = np.load(
            os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "test_fold" + str(fold) + ".npy")
        )

        train_hand_crafted, mean, std = compute_hand_crafted_features(train_inputs)
        val_hand_crafted, _, _ = compute_hand_crafted_features(val_inputs, mean, std)
        test_hand_crafted, _, _ = compute_hand_crafted_features(test_inputs, mean, std)

        if not os.path.exists(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH):
            os.makedirs(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH)
        np.save(
            os.path.join(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH, "train_fold" + str(fold) + ".npy"),
            train_hand_crafted,
        )
        np.save(
            os.path.join(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH, "val_fold" + str(fold) + ".npy"), val_hand_crafted
        )
        np.save(
            os.path.join(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH, "test_fold" + str(fold) + ".npy"),
            test_hand_crafted,
        )
