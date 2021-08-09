# Internal
from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_encoder_decoder, test_encoder_decoder
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHEN_LSTM_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHEN,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHEN_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    FUTURE_TRAJECTORY_LENGTH,
)
from experiments.models import RecurrentDecoder, RecurrentEncoder

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os

# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cuda")
INPUT_SIZE = 4
OUTPUT_SIZE = NUM_CAMERAS * FUTURE_TRAJECTORY_LENGTH
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_HIDDEN_UNITS = 128
WEIGHT_DECAY = 0
NUM_EPOCHS = 20
DEBUG_MODE = False
MODEL_SAVE_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHEN, "lstm")

encoder_args = {
    "device": DEVICE,
    "num_cameras": NUM_CAMERAS,
    "input_size": INPUT_SIZE,
    "num_hidden_units": NUM_HIDDEN_UNITS,
    "recurrence_type": "lstm",
}

decoder_args = {
    "device": DEVICE,
    "num_cameras": NUM_CAMERAS,
    "input_size": NUM_HIDDEN_UNITS,
    "recurrence_type": "lstm",
    "output_size": NUM_CAMERAS,
    "num_timesteps": FUTURE_TRAJECTORY_LENGTH,
}


for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

    # ########## SET UP DATASET ########## #
    train_dataset_args = {
        "inputs_path": CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "train",
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": False,
        "flatten_targets": False,
    }

    val_dataset_args = {
        "inputs_path": CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "val",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": False,
        "flatten_targets": False,
    }

    test_dataset_args = {
        "inputs_path": CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHEN_TARGETS_PATH,
        "fold": fold,
        "phase": "test",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "flatten_inputs": False,
        "flatten_targets": False,
    }

    train_loader = get_coordinate_trajectory_dataset(**train_dataset_args)
    val_loader = get_coordinate_trajectory_dataset(**val_dataset_args)
    test_loader = get_coordinate_trajectory_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    encoder = RecurrentEncoder(**encoder_args).to(DEVICE)
    decoder = RecurrentDecoder(**decoder_args).to(DEVICE)
    params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.BCELoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_ap = 0

    for epoch in range(NUM_EPOCHS):
        print("----------- EPOCH " + str(epoch) + " -----------")

        trainer_args = {
            "encoder": encoder,
            "decoder": decoder,
            "device": DEVICE,
            "train_loader": train_loader,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
        }

        loss, ap = train_encoder_decoder(**trainer_args)
        print("Train loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

        val_args = {
            "encoder": encoder,
            "decoder": decoder,
            "device": DEVICE,
            "test_loader": val_loader,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
        }

        loss, ap = test_encoder_decoder(**val_args)
        print("Validation loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

        if ap > best_ap:
            best_ap = ap
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            encoder_save_name = "encoder_fold_" + str(fold) + ".weights"
            decoder_save_name = "decoder_fold_" + str(fold) + ".weights"
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            torch.save(best_encoder.state_dict(), os.path.join(MODEL_SAVE_PATH, encoder_save_name))
            torch.save(best_decoder.state_dict(), os.path.join(MODEL_SAVE_PATH, decoder_save_name))

    test_args = {
        "encoder": best_encoder,
        "decoder": best_decoder,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": CROSS_VALIDATION_WHEN_LSTM_PATH,
        "predictions_save_shape": (15, 60),
    }

    loss, ap = test_encoder_decoder(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
