# Internal
from experiments.datasets import get_trajectory_tensor_dataset
from experiments.trainer import train_embedder_encoder_decoder, test_embedder_encoder_decoder
from global_config.global_config import (
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHICH_CNN_GRU_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHICH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH,
    CROSS_VALIDATION_MODELS_PATH_AUTOENCODER,
    DATA_PATH,
)
from experiments.models import FullyConnectedTrajectoryTensorClassifier, CNN_2D_Encoder, TrajectoryTensorGRU

# External
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os
import pandas as pd
import gc
import numpy as np


# ########## NETWORK CONFIG ########## #
DEVICE = torch.device("cuda")
OUTPUT_SIZE = 15
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_HIDDEN_UNITS = 128
WEIGHT_DECAY = 0
NUM_EPOCHS = 25
MODEL_LOAD_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHICH, "cnn_gru")
DEBUG_MODE = False
heatmap_scale = 2
heatmap_smoothing_sigma = 3
heatmap_size = (9 * heatmap_scale, 16 * heatmap_scale)


trajectory_tensor_path = os.path.join(DATA_PATH, "cross_validation", "multi_target", "inputs", "trajectory_tensors")

which_targets_path = os.path.join(DATA_PATH, "cross_validation", "multi_target", "targets", "which_targets")

embedder_args = {"device": DEVICE, "heatmap_size": heatmap_size, "output_size": NUM_HIDDEN_UNITS}

encoder_args = {"device": DEVICE, "num_hidden_units": NUM_HIDDEN_UNITS, "input_size": NUM_HIDDEN_UNITS}

decoder_args = {"device": DEVICE, "input_size": NUM_HIDDEN_UNITS, "output_size": OUTPUT_SIZE}

all_aps = []
for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

    test_dataset_args = {
        "inputs_path": trajectory_tensor_path,
        "targets_path": which_targets_path,
        "fold": fold,
        "heatmap_size": heatmap_size,
        "heatmap_smoothing_sigma": heatmap_smoothing_sigma,
        "phase": "test",
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "multi_target": True,
    }

    test_loader = get_trajectory_tensor_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    embedder = CNN_2D_Encoder(**embedder_args).to(DEVICE)
    encoder = TrajectoryTensorGRU(**encoder_args).to(DEVICE)
    decoder = FullyConnectedTrajectoryTensorClassifier(**decoder_args).to(DEVICE)

    encoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "encoder_fold_" + str(fold) + ".weights")))
    decoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "decoder_fold_" + str(fold) + ".weights")))

    embedder_path = os.path.join(
        CROSS_VALIDATION_MODELS_PATH_AUTOENCODER,
        "multi_view",
        "size_" + str(heatmap_size[0]),
        "sigma_" + str(heatmap_smoothing_sigma),
        "encoder_fold_" + str(fold) + ".weights",
    )
    embedder.load_state_dict(torch.load(embedder_path))

    params = list(embedder.parameters()) + list(encoder.parameters()) + list(decoder.parameters())

    loss_function = nn.BCELoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_ap = 0

    test_args = {
        "embedder": embedder,
        "encoder": encoder,
        "decoder": decoder,
        "device": DEVICE,
        "test_loader": test_loader,
        "loss_function": loss_function,
        "debug_mode": False,
        "fold_num": fold,
        "predictions_save_path": None,
        "coordinate_trajectory_inputs": False,
        "variable_batch_size": True,
    }

    loss, ap = test_embedder_encoder_decoder(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
    gc.collect()
    all_aps.append(ap)
print("MEAN AP:", np.round(np.mean(all_aps), 3) * 100)
