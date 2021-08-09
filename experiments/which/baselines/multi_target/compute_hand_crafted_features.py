# Internal
from global_config.global_config import (
    CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    DATA_PATH,
)
from experiments.which.baselines.compute_hand_crafted_features import compute_hand_crafted_features

# External
import numpy as np
import os
import pickle


def compute_multi_target_hand_crafted_features(inputs, mean, std):
    hand_crafted_features = [[]] * len(inputs)
    for dataset_index, input in enumerate(inputs):
        for target in input:
            hand_crafted_feature = np.zeros((1, 10))
            # Velocity
            hand_crafted_feature[0, 0] = (target[-1, 2] - target[-1, 0]) - (target[0, 2] - target[0, 0])
            hand_crafted_feature[0, 1] = (target[-1, 3] - target[-1, 1]) - (target[0, 3] - target[0, 1])
            # Acceleration
            start_velocity_x = (target[1, 2] - target[1, 0]) - (target[0, 2] - target[0, 0])
            end_velocity_x = (target[-1, 2] - target[-1, 0]) - (target[-2, 2] - target[-2, 0])
            start_velocity_y = (target[1, 3] - target[1, 1]) - (target[0, 3] - target[0, 1])
            end_velocity_y = (target[-1, 3] - target[-1, 1]) - (target[-2, 3] - target[-2, 1])
            hand_crafted_feature[0, 2] = start_velocity_x - end_velocity_x
            hand_crafted_feature[0, 3] = start_velocity_y - end_velocity_y
            # Scale
            hand_crafted_feature[0, 4] = target[-1, 2] - target[-1, 0]
            hand_crafted_feature[0, 5] = target[-1, 3] - target[-1, 1]
            # Exit location
            hand_crafted_feature[0, 6] = target[-1, 0]
            hand_crafted_feature[0, 7] = target[-1, 1]
            hand_crafted_feature[0, 8] = target[-1, 2]
            hand_crafted_feature[0, 9] = target[-1, 3]

            hand_crafted_feature = (hand_crafted_feature - mean) / std
            hand_crafted_features[dataset_index] = hand_crafted_features[dataset_index] + [hand_crafted_feature]

    return hand_crafted_features


for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("Computing features for fold", fold)
    train_inputs = np.load(
        os.path.join(CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH, "train_fold" + str(fold) + ".npy")
    )
    test_input_path = os.path.join(
        DATA_PATH,
        "cross_validation",
        "multi_target",
        "inputs",
        "coordinate_trajectories",
        "test_fold" + str(fold) + ".pickle",
    )
    with open(test_input_path, "rb") as fp:
        test_inputs_list = pickle.load(fp)

    _, mean, std = compute_hand_crafted_features(train_inputs)
    test_hand_crafted = compute_multi_target_hand_crafted_features(test_inputs_list, mean, std)
    if not os.path.exists(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH):
        os.makedirs(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH)
    with open(
        os.path.join(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH, "multi_target", "test_fold" + str(fold) + ".pickle"),
        "wb",
    ) as fp:
        pickle.dump(test_hand_crafted, fp)
