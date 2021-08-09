# Internal
from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_model, test_model
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
    CROSS_VALIDATION_WHEN_HAND_CRAFTED_FEATURE_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHEN,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    FUTURE_TRAJECTORY_LENGTH,
)
from experiments.models import FullyConnectedClassifier

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os

# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cpu")
INPUT_SIZE = 10
OUTPUT_SIZE = NUM_CAMERAS * FUTURE_TRAJECTORY_LENGTH
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
WEIGHT_DECAY = 0
NUM_EPOCHS = 25
DEBUG_MODE = False
MODEL_SAVE_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHEN, "hand_crafted_features")
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

network_args = {"device": DEVICE, "num_cameras": NUM_CAMERAS, "input_size": INPUT_SIZE, "output_size": OUTPUT_SIZE}

for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
    # ########## SET UP DATASET ########## #
    train_dataset_args = {
        "inputs_path": CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "train",
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": True,
        "flatten_targets": True,
    }

    val_dataset_args = {
        "inputs_path": CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "val",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": True,
        "flatten_targets": True,
    }

    test_dataset_args = {
        "inputs_path": CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "test",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": True,
        "flatten_targets": True,
    }
    train_loader = get_coordinate_trajectory_dataset(**train_dataset_args)
    val_loader = get_coordinate_trajectory_dataset(**val_dataset_args)
    test_loader = get_coordinate_trajectory_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    model = FullyConnectedClassifier(**network_args).to(DEVICE)
    print(fold, model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.BCELoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_ap = 0

    for epoch in range(NUM_EPOCHS):
        print("----------- EPOCH " + str(epoch) + " -----------")

        trainer_args = {
            "model": model,
            "device": DEVICE,
            "train_loader": train_loader,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
        }

        loss, ap = train_model(**trainer_args)
        print("Train loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

        val_args = {
            "model": model,
            "device": DEVICE,
            "test_loader": val_loader,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
        }

        loss, ap = test_model(**val_args)
        print("Validation loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

        if ap > best_ap:
            best_ap = ap
            best_model = copy.deepcopy(model)
            model_save_name = "fold_" + str(fold) + ".weights"
            torch.save(best_model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_save_name))

    test_args = {
        "model": best_model,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": CROSS_VALIDATION_WHEN_HAND_CRAFTED_FEATURE_PATH,
        "predictions_save_shape": (15, 60),
    }

    loss, ap = test_model(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
