# Internal
from experiments.datasets import get_trajectory_tensor_dataset
from experiments.trainer import train_encoder_decoder, test_encoder_decoder
from global_config.global_config import (
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHERE,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_WHERE_GRID_SEARCH_PATH,
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH,
    DATA_PATH,
)
from experiments.models import CNN_3D_Decoder, CNN_3D_Encoder

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
NUM_HIDDEN_UNITS = 512
WEIGHT_DECAY = 0
NUM_EPOCHS = 25
DEBUG_MODE = False
MODEL_LOAD_PATH = os.path.join(CROSS_VALIDATION_MODELS_PATH_WHERE, "3dcnn")
heatmap_scale = 2
heatmap_smoothing_sigma = 4
heatmap_size = (9 * heatmap_scale, 16 * heatmap_scale)

results = pd.DataFrame()

heatmap_size = (9 * heatmap_scale, 16 * heatmap_scale)

trajectory_tensor_path = os.path.join(DATA_PATH, "cross_validation", "multi_target", "inputs", "trajectory_tensors")

where_targets_path = os.path.join(DATA_PATH, "cross_validation", "multi_target", "targets", "where_targets")

encoder_args = {"device": DEVICE, "output_size": NUM_HIDDEN_UNITS, "heatmap_size": heatmap_size}

decoder_args = {"device": DEVICE, "input_size": NUM_HIDDEN_UNITS}
all_aps = []
for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):
    print("fold", fold)

    # ########## SET UP DATASET ########## #
    test_dataset_args = {
        "inputs_path": trajectory_tensor_path,
        "targets_path": where_targets_path,
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
    encoder = CNN_3D_Encoder(**encoder_args).to(DEVICE)
    decoder = CNN_3D_Decoder(**decoder_args).to(DEVICE)
    encoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "encoder_fold_" + str(fold) + ".weights")))
    decoder.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, "decoder_fold_" + str(fold) + ".weights")))

    params = list(encoder.parameters()) + list(decoder.parameters())

    loss_function = nn.BCELoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_ap = 0

    test_args = {
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

    loss, ap = test_encoder_decoder(**test_args)
    print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))
    all_aps.append(ap)
    gc.collect()

print("MEAN AP:", np.round(np.mean(all_aps), 3) * 100)
