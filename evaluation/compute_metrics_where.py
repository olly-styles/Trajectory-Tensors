# Internal
from global_config.global_config import (
    CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH,
    CROSS_VALIDATION_WHERE_MOST_SIMILAR_TRAJECTORY_PATH,
    CROSS_VALIDATION_WHERE_HAND_CRAFTED_FEATURE_PATH,
    CROSS_VALIDATION_WHERE_GRU_PATH,
    CROSS_VALIDATION_WHERE_LSTM_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
    CROSS_VALIDATION_WHERE_1DCNN_PATH,
    CROSS_VALIDATION_WHERE_3D_CNN_PATH,
    CROSS_VALIDATION_WHERE_2D_1D_CNN_PATH,
    CROSS_VALIDATION_WHERE_2D_1D_CNN_SINGLE_VIEW_PATH,
    CROSS_VALIDATION_WHERE_3D_CNN_SINGLE_VIEW_PATH,
    CROSS_VALIDATION_WHERE_CNN_GRU_PATH,
    CROSS_VALIDATION_WHERE_CNN_GRU_SINGLE_VIEW_PATH,
    GRID_CELL_SIZE,
)
from evaluation.mctf_metrics import get_ap, get_dataset_siou_where, get_dataset_tensor_fde, get_dataset_tensor_ade

# External
import os
import numpy as np

models = {
    "Training set mean": CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH,
    "Most similar trajectory": CROSS_VALIDATION_WHERE_MOST_SIMILAR_TRAJECTORY_PATH,
    "Hand-crafted features": CROSS_VALIDATION_WHERE_HAND_CRAFTED_FEATURE_PATH,
    "Coordinate trajectory GRU": CROSS_VALIDATION_WHERE_GRU_PATH,
    "Coordinate trajectory LSTM": CROSS_VALIDATION_WHERE_LSTM_PATH,
    "Coordinate trajectory 1D-CNN": CROSS_VALIDATION_WHERE_1DCNN_PATH,
    "Trajectory tensor 3D-CNN": CROSS_VALIDATION_WHERE_3D_CNN_PATH,
    "Trajectory tensor single view 3D-CNN": CROSS_VALIDATION_WHERE_3D_CNN_SINGLE_VIEW_PATH,
    "Trajectory tensor 2D-1D-CNN": CROSS_VALIDATION_WHERE_2D_1D_CNN_PATH,
    "Trajectory tensor single view 2D-1D-CNN": CROSS_VALIDATION_WHERE_2D_1D_CNN_SINGLE_VIEW_PATH,
    "Trajectory tensor CNN-GRU": CROSS_VALIDATION_WHERE_CNN_GRU_PATH,
    "Trajectory tensor single view CNN-GRU": CROSS_VALIDATION_WHERE_CNN_GRU_SINGLE_VIEW_PATH,
}

for model in models:
    model_aps = []
    model_sious = []
    model_tensor_ades = []
    model_tensor_fdes = []
    try:
        for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
            targets = np.load(os.path.join(CROSS_VALIDATION_WHERE_TARGETS_PATH, "test_fold" + str(fold) + ".npy"))
            predictions = np.load(os.path.join(models[model], "test_fold" + str(fold) + ".npy"))
            model_tensor_ades.append(get_dataset_tensor_ade(targets, predictions, GRID_CELL_SIZE))
            model_tensor_fdes.append(get_dataset_tensor_fde(targets, predictions, GRID_CELL_SIZE))
            model_aps.append(get_ap(targets, predictions))
            model_sious.append(get_dataset_siou_where(targets, predictions))
        mean_ap = np.round(np.mean(model_aps) * 100, 1)
        mean_siou = np.round(np.mean(model_sious) * 100, 1)
        ade = np.round(np.mean(model_tensor_ades), 1)
        fde = np.round(np.mean(model_tensor_fdes), 1)
        print(model, "AP:", mean_ap, "SIOU:", mean_siou, "ADE:", ade, "FDE:", fde)
    except FileNotFoundError:
        print(model, ": predictions not found")
