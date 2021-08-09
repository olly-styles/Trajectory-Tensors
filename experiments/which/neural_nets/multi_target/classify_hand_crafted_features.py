# Internal
from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_model, test_model
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
    CROSS_VALIDATION_WHICH_HAND_CRAFTED_FEATURE_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHICH,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    DATA_PATH,
)
from experiments.models import FullyConnectedClassifier

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os
import numpy as np


# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cuda")
INPUT_SIZE = 10
OUTPUT_SIZE = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_WORKERS = 4
WEIGHT_DECAY = 0
NUM_EPOCHS = 30
DEBUG_MODE = False
MODEL_SAVE_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHICH, "hand_crafted_features")
TARGETS_PATH = os.path.join("WNMF-dataset", "cross_validation", "multi_target", "targets", "which_targets")
DEPARTURE_CAMERA_PATH = os.path.join(DATA_PATH, "cross_validation", "multi_target", "inputs", "departure_cameras")
MODEL_LOAD_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHICH, "hand_crafted_features")

network_args = {"device": DEVICE, "num_cameras": NUM_CAMERAS, "input_size": INPUT_SIZE, "output_size": OUTPUT_SIZE}
all_aps = []
for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

    # ########## SET UP DATASET ########## #
    test_dataset_args = {
        "inputs_path": os.path.join(CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH, "multi_target"),
        "departure_cameras_path": DEPARTURE_CAMERA_PATH,
        "targets_path": TARGETS_PATH,
        "fold": fold,
        "phase": "test",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": True,
        "multi_target": True,
    }

    test_loader = get_coordinate_trajectory_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    model = FullyConnectedClassifier(**network_args).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "fold_" + str(fold) + ".weights")))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.BCELoss()

    # ########## EVALUATE ########## #

    test_args = {
        "model": model,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": CROSS_VALIDATION_WHICH_HAND_CRAFTED_FEATURE_PATH,
        "variable_batch_size": True,
    }

    loss, ap = test_model(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
    all_aps.append(ap)
print("MEAN AP:", np.round(np.mean(all_aps), 3) * 100)
