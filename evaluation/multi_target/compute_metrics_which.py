# Internal
from global_config.global_config import (
    CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH,
    CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH,
    CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH,
    CROSS_VALIDATION_WHICH_HAND_CRAFTED_FEATURE_PATH,
    CROSS_VALIDATION_WHICH_GRU_PATH,
    CROSS_VALIDATION_WHICH_LSTM_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    CROSS_VALIDATION_WHICH_1DCNN_PATH,
    CROSS_VALIDATION_WHICH_3DCNN_PATH,
    CROSS_VALIDATION_WHICH_2D1DCNN_PATH,
    CROSS_VALIDATION_WHICH_2D1DCNN_SINGLE_VIEW_PATH,
    CROSS_VALIDATION_WHICH_3DCNN_SINGLE_VIEW_PATH,
    CROSS_VALIDATION_WHICH_CNN_GRU_PATH,
    CROSS_VALIDATION_WHICH_CNN_GRU_SINGLE_VIEW_PATH,
    DATA_PATH,
)
from evaluation.mctf_metrics import get_ap

# External
import os
import numpy as np
import pickle

models = {
    "Training set mean": CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH,
    "Shortest realworld distance": CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH,
    "Most similar trajectory": CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH,
    "Hand crafted features": CROSS_VALIDATION_WHICH_HAND_CRAFTED_FEATURE_PATH,
    "Coordinate trajectory GRU": CROSS_VALIDATION_WHICH_GRU_PATH,
    "Coordinate trajectory LSTM": CROSS_VALIDATION_WHICH_LSTM_PATH,
    "Coordinate trajectory 1D-CNN": CROSS_VALIDATION_WHICH_1DCNN_PATH,
    "Trajectory tensor 3D-CNN": CROSS_VALIDATION_WHICH_3DCNN_PATH,
    "Trajectory tensor single view 3D-CNN": CROSS_VALIDATION_WHICH_3DCNN_SINGLE_VIEW_PATH,
    "Trajectory tensor 2D-1D-CNN": CROSS_VALIDATION_WHICH_2D1DCNN_PATH,
    "Trajectory tensor single view 2D-1D-CNN": CROSS_VALIDATION_WHICH_2D1DCNN_SINGLE_VIEW_PATH,
    "Trajectory tensor CNN-GRU": CROSS_VALIDATION_WHICH_CNN_GRU_PATH,
    "Trajectory tensor single view CNN-GRU": CROSS_VALIDATION_WHICH_CNN_GRU_SINGLE_VIEW_PATH,
}

for model in models:
    model_aps = []
    try:
        for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
            test_targets_path = os.path.join(
                DATA_PATH,
                "cross_validation",
                "multi_target",
                "targets",
                "which_targets",
                "test_fold" + str(fold) + ".pickle",
            )
            with open(test_targets_path, "rb") as fp:
                test_targets_list = pickle.load(fp)
            targets = np.empty((0, 15))
            for sample in test_targets_list:
                for target in sample:
                    target = target.reshape(1, 15)
                    targets = np.append(target, targets, axis=0)
            predictions = np.load(os.path.join(models[model], "multi_target_test_fold" + str(fold) + ".npy"))
            model_aps.append(get_ap(targets, predictions))
        mean_ap = np.round(np.mean(model_aps) * 100, 1)
        print(model, "AP:", mean_ap)
    except FileNotFoundError:
        print(model, ": predictions not found")
