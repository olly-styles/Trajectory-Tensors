# Internal
from experiments.datasets import get_trajectory_tensor_dataset
from experiments.trainer import train_encoder_decoder, test_encoder_decoder
from global_config.global_config import (
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHICH_3DCNN_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHICH,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHICH_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH,
    CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH,
)
from experiments.models import FullyConnectedTrajectoryTensorClassifier, CNN_3D_Encoder

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
DEVICE = torch.device("cpu")
OUTPUT_SIZE = 15
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_HIDDEN_UNITS = 512
WEIGHT_DECAY = 0
NUM_EPOCHS = 20
DEBUG_MODE = False
MODEL_SAVE_PATH = None

results = pd.DataFrame()

for multi_view_tensors in [True]:
    for heatmap_scale in [1, 2, 3]:
        for heatmap_smoothing_sigma in [4, 3, 2, 1, 0]:
            heatmap_size = (9 * heatmap_scale, 16 * heatmap_scale)
            if multi_view_tensors:
                trajectory_tensor_path = CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH
            else:
                trajectory_tensor_path = CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH

            encoder_args = {"device": DEVICE, "output_size": NUM_HIDDEN_UNITS, "heatmap_size": heatmap_size}

            decoder_args = {"device": DEVICE, "input_size": NUM_HIDDEN_UNITS, "output_size": OUTPUT_SIZE}

            for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

                # ########## SET UP DATASET ########## #
                train_dataset_args = {
                    "inputs_path": trajectory_tensor_path,
                    "targets_path": CROSS_VALIDATION_WHICH_TARGETS_PATH,
                    "fold": fold,
                    "heatmap_size": heatmap_size,
                    "heatmap_smoothing_sigma": heatmap_smoothing_sigma,
                    "phase": "train",
                    "batch_size": BATCH_SIZE,
                    "shuffle": True,
                    "num_workers": NUM_WORKERS,
                }

                val_dataset_args = {
                    "inputs_path": trajectory_tensor_path,
                    "targets_path": CROSS_VALIDATION_WHICH_TARGETS_PATH,
                    "fold": fold,
                    "heatmap_size": heatmap_size,
                    "heatmap_smoothing_sigma": heatmap_smoothing_sigma,
                    "phase": "val",
                    "batch_size": BATCH_SIZE,
                    "shuffle": False,
                    "num_workers": NUM_WORKERS,
                }

                test_dataset_args = {
                    "inputs_path": trajectory_tensor_path,
                    "targets_path": CROSS_VALIDATION_WHICH_TARGETS_PATH,
                    "fold": fold,
                    "heatmap_size": heatmap_size,
                    "heatmap_smoothing_sigma": heatmap_smoothing_sigma,
                    "phase": "test",
                    "batch_size": BATCH_SIZE,
                    "shuffle": False,
                    "num_workers": NUM_WORKERS,
                }

                train_loader = get_trajectory_tensor_dataset(**train_dataset_args)
                val_loader = get_trajectory_tensor_dataset(**val_dataset_args)
                test_loader = get_trajectory_tensor_dataset(**test_dataset_args)

                # ########## SET UP MODEL ########## #
                encoder = CNN_3D_Encoder(**encoder_args).to(DEVICE)
                decoder = FullyConnectedTrajectoryTensorClassifier(**decoder_args).to(DEVICE)
                params = list(encoder.parameters()) + list(decoder.parameters())

                optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                loss_function = nn.BCELoss()

                # ########## TRAIN AND EVALUATE ########## #
                best_loss = np.inf

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
                        "coordinate_trajectory_inputs": False,
                        "compute_ap": False,
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
                        "coordinate_trajectory_inputs": False,
                        "compute_ap": False,
                    }

                    loss, ap = test_encoder_decoder(**val_args)
                    print("Validation loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

                    if loss <= best_loss:
                        best_loss = loss
                        best_encoder = copy.deepcopy(encoder)
                        best_decoder = copy.deepcopy(decoder)
                        encoder_save_name = "encoder_fold_" + str(fold) + ".weights"
                        decoder_save_name = "decoder_fold_" + str(fold) + ".weights"
                        if not MODEL_SAVE_PATH is None:
                            torch.save(best_encoder.state_dict(), os.path.join(MODEL_SAVE_PATH, encoder_save_name))
                            torch.save(best_decoder.state_dict(), os.path.join(MODEL_SAVE_PATH, decoder_save_name))

                # Get val AP for best model
                best_val_args = {
                    "encoder": best_encoder,
                    "decoder": best_decoder,
                    "device": DEVICE,
                    "test_loader": val_loader,
                    "loss_function": loss_function,
                    "debug_mode": DEBUG_MODE,
                    "coordinate_trajectory_inputs": False,
                    "compute_ap": True,
                }
                loss, val_ap = test_encoder_decoder(**best_val_args)
                print("Val loss: {0:.5f} AP: {1:.4f}".format(loss, val_ap))

                test_args = {
                    "encoder": best_encoder,
                    "decoder": best_decoder,
                    "device": DEVICE,
                    "test_loader": test_loader,
                    "loss_function": loss_function,
                    "debug_mode": False,
                    "fold_num": fold,
                    "predictions_save_path": None,
                    "coordinate_trajectory_inputs": False,
                }

                loss, ap = test_encoder_decoder(**test_args)
                print("Test loss: {0:.5f} AP: {1:.4f}".format(loss, ap))

                result = {
                    "Fold": fold,
                    "heatmap_smoothing_sigma": heatmap_smoothing_sigma,
                    "multi_view_tensors": multi_view_tensors,
                    "heatmap_scale": heatmap_scale,
                    "val_ap": val_ap,
                    "test_ap": ap,
                }
                results = results.append(result, ignore_index=True)
                save_path = os.path.join(CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH, "3dcnn")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                results.to_csv(os.path.join(save_path, "results.csv"), index=False)
                gc.collect()
