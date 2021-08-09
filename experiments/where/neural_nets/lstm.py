# Internal
from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_with_spatial_upsample, test_with_spatial_upsample
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHERE_LSTM_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHERE,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    FUTURE_TRAJECTORY_LENGTH,
    BASE_HEATMAP_SIZE,
)
from experiments.models import RecurrentEncoder, RecurrentTemporalDecoder, CNN_2D_Decoder

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os

# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cuda")
INPUT_SIZE = 4
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_HIDDEN_UNITS = 128
WEIGHT_DECAY = 0
NUM_EPOCHS = 56
DEBUG_MODE = False
MODEL_SAVE_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHERE, "lstm")

encoder_args = {
    "device": DEVICE,
    "num_cameras": NUM_CAMERAS,
    "input_size": INPUT_SIZE,
    "num_hidden_units": NUM_HIDDEN_UNITS,
    "recurrence_type": "lstm",
}

temporal_decoder_args = {
    "device": DEVICE,
    "num_cameras": NUM_CAMERAS,
    "input_size": NUM_HIDDEN_UNITS,
    "recurrence_type": "lstm",
    "num_timesteps": FUTURE_TRAJECTORY_LENGTH,
}

spatial_decoder_args = {"device": DEVICE, "feature_size": NUM_HIDDEN_UNITS, "num_cameras": NUM_CAMERAS}

for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

    # ########## SET UP DATASET ########## #
    train_dataset_args = {
        "inputs_path": CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
        "departure_cameras_path": CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
        "targets_path": CROSS_VALIDATION_WHERE_TARGETS_PATH,
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
        "targets_path": CROSS_VALIDATION_WHERE_TARGETS_PATH,
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
        "targets_path": CROSS_VALIDATION_WHERE_TARGETS_PATH,
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
    temporal_decoder = RecurrentTemporalDecoder(**temporal_decoder_args).to(DEVICE)
    spatial_decoder = CNN_2D_Decoder(**spatial_decoder_args).to(DEVICE)

    params = list(encoder.parameters()) + list(temporal_decoder.parameters()) + list(spatial_decoder.parameters())

    print(encoder, temporal_decoder, spatial_decoder)

    optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.BCELoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_ap = 0

    for epoch in range(NUM_EPOCHS):
        print("----------- EPOCH " + str(epoch) + " -----------")

        trainer_args = {
            "encoder": encoder,
            "temporal_decoder": temporal_decoder,
            "spatial_decoder": spatial_decoder,
            "device": DEVICE,
            "train_loader": train_loader,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
            "compute_ap": False,
        }

        loss, ap = train_with_spatial_upsample(**trainer_args)
        print("Train loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

        val_args = {
            "encoder": encoder,
            "temporal_decoder": temporal_decoder,
            "spatial_decoder": spatial_decoder,
            "device": DEVICE,
            "test_loader": val_loader,
            "loss_function": loss_function,
            "debug_mode": DEBUG_MODE,
        }
        if epoch % 5 == 0 and epoch != 0:
            loss, ap = test_with_spatial_upsample(**val_args)
            print("Validation loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
            if ap >= best_ap:
                best_ap = ap
                best_encoder = copy.deepcopy(encoder)
                best_temporal_decoder = copy.deepcopy(temporal_decoder)
                best_spatial_decoder = copy.deepcopy(spatial_decoder)
                encoder_save_name = "encoder_fold_" + str(fold) + ".weights"
                temporal_decoder_save_name = "temporal_decoder_fold_" + str(fold) + ".weights"
                spatial_decoder_save_name = "spatial_decoder_fold_" + str(fold) + ".weights"
                if not os.path.exists(MODEL_SAVE_PATH):
                    os.makedirs(MODEL_SAVE_PATH)
                torch.save(best_encoder.state_dict(), os.path.join(MODEL_SAVE_PATH, encoder_save_name))
                torch.save(
                    best_temporal_decoder.state_dict(), os.path.join(MODEL_SAVE_PATH, temporal_decoder_save_name)
                )
                torch.save(best_spatial_decoder.state_dict(), os.path.join(MODEL_SAVE_PATH, spatial_decoder_save_name))

    test_args = {
        "encoder": best_encoder,
        "temporal_decoder": best_temporal_decoder,
        "spatial_decoder": best_spatial_decoder,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": CROSS_VALIDATION_WHERE_LSTM_PATH,
        "predictions_save_shape": (15, 60, 9, 16),
    }

    loss, ap = test_with_spatial_upsample(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
