# Internal
from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_encoder_decoder, test_encoder_decoder
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHEN_1DCNN_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHEN,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    DATA_PATH,
)
from experiments.models import ConvolutionalDecoder, ConvolutionalEncoder

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os
import numpy as np


# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cuda")
INPUT_SIZE = 4
OUTPUT_SIZE = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_HIDDEN_UNITS = 128
WEIGHT_DECAY = 0
NUM_EPOCHS = 30
DEBUG_MODE = False
MODEL_SAVE_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHEN, "1dcnn")

TARGETS_PATH = os.path.join("WNMF-dataset", "cross_validation", "multi_target", "targets", "when_targets")
DEPARTURE_CAMERA_PATH = os.path.join(DATA_PATH, "cross_validation", "multi_target", "inputs", "departure_cameras")
MODEL_LOAD_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHEN, "1dcnn")
INPUTS_PATH = os.path.join(DATA_PATH, "cross_validation", "multi_target", "inputs", "coordinate_trajectories")


encoder_args = {"device": DEVICE, "num_cameras": NUM_CAMERAS, "input_size": INPUT_SIZE, "output_size": NUM_HIDDEN_UNITS}

decoder_args = {
    "device": DEVICE,
    "num_cameras": NUM_CAMERAS,
    "input_size": NUM_HIDDEN_UNITS,
    "output_size": OUTPUT_SIZE,
}
all_aps = []
for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

    # ########## SET UP DATASET ########## #
    test_dataset_args = {
        "inputs_path": INPUTS_PATH,
        "departure_cameras_path": DEPARTURE_CAMERA_PATH,
        "targets_path": TARGETS_PATH,
        "fold": fold,
        "phase": "test",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": False,
        "multi_target": True,
    }

    test_loader = get_coordinate_trajectory_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    encoder = ConvolutionalEncoder(**encoder_args).to(DEVICE)
    decoder = ConvolutionalDecoder(**decoder_args).to(DEVICE)
    encoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "encoder_fold_" + str(fold) + ".weights")))
    decoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "decoder_fold_" + str(fold) + ".weights")))

    loss_function = nn.BCELoss()

    # ########## EVALUATE ########## #

    test_args = {
        "encoder": encoder,
        "decoder": decoder,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": None,
        "variable_batch_size": True,
    }

    loss, ap = test_encoder_decoder(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
    all_aps.append(ap)
print("MEAN AP:", np.round(np.mean(all_aps), 3) * 100)
