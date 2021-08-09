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
)
from evaluation.mctf_metrics import get_ap

# External
import os
import numpy as np

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
            targets = np.load(os.path.join(CROSS_VALIDATION_WHICH_TARGETS_PATH, "test_fold" + str(fold) + ".npy"))
            predictions = np.load(os.path.join(models[model], "test_fold" + str(fold) + ".npy"))
            model_aps.append(get_ap(targets, predictions))
        mean_ap = np.round(np.mean(model_aps) * 100, 1)
        print(model, "AP:", mean_ap)
    except FileNotFoundError:
        print(model, ": predictions not found")
