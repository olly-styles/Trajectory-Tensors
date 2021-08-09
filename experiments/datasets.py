# Internal
from experiments.utils import normalize_trajectory_tensor, smooth_trajectory_tensor

# External
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle


class CoordinateTrajectoryDataset(Dataset):
    def __init__(self, inputs_path, departure_cameras_path, targets_path, flatten_inputs, flatten_targets=False):
        self.inputs = np.load(inputs_path)
        self.departure_cameras = np.load(departure_cameras_path)
        self.targets = np.load(targets_path)
        self.flatten_inputs = flatten_inputs
        self.flatten_targets = flatten_targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        batch_inputs = self.inputs[idx]
        batch_departure_cameras = self.departure_cameras[idx]
        batch_targets = self.targets[idx]

        # Index from 0
        batch_departure_cameras -= 1

        if self.flatten_inputs:
            batch_inputs = batch_inputs.flatten()
        if self.flatten_targets:
            batch_targets = batch_targets.flatten()

        return {"inputs": batch_inputs, "departure_cameras": batch_departure_cameras, "targets": batch_targets}


class MultiTargetCoordinateTrajectoryDataset(Dataset):
    def __init__(self, inputs_path, departure_cameras_path, targets_path, flatten_inputs, flatten_targets=False):
        with open(inputs_path, "rb") as fp:
            self.inputs = pickle.load(fp)
        with open(targets_path, "rb") as fp:
            self.targets = pickle.load(fp)
        with open(departure_cameras_path, "rb") as fp:
            self.departure_cameras = pickle.load(fp)
        self.flatten_inputs = flatten_inputs
        self.flatten_targets = flatten_targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        batch_inputs = np.stack(self.inputs[idx], axis=0).astype(np.float32)
        batch_targets = np.stack(self.targets[idx], axis=0)
        batch_departure_cameras = np.stack(self.departure_cameras[idx], axis=0)

        # Index from 0
        batch_departure_cameras -= 1

        if self.flatten_inputs:
            batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)
        if self.flatten_targets:
            batch_targets = batch_targets.reshape(batch_targets.shape[0], -1)

        return {"inputs": batch_inputs, "departure_cameras": batch_departure_cameras, "targets": batch_targets}


class TrajectoryTensorDataset(Dataset):
    def __init__(self, inputs_path, targets_path, heatmap_smoothing_sigma):
        self.inputs = np.load(inputs_path)
        self.targets = np.load(targets_path)
        self.heatmap_smoothing_sigma = heatmap_smoothing_sigma

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        this_input = self.inputs[idx].astype(np.float32)
        this_target = self.targets[idx]

        if self.heatmap_smoothing_sigma != 0:
            this_input = smooth_trajectory_tensor(this_input, self.heatmap_smoothing_sigma)
            this_input = normalize_trajectory_tensor(this_input)

        return {"inputs": this_input, "targets": this_target}


class MultiTargetTrajectoryTensorDataset(Dataset):
    def __init__(self, inputs_path, targets_path, heatmap_smoothing_sigma):
        with open(inputs_path, "rb") as fp:
            self.inputs = pickle.load(fp)
        with open(targets_path, "rb") as fp:
            self.targets = pickle.load(fp)
        self.heatmap_smoothing_sigma = heatmap_smoothing_sigma

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        this_input = np.stack(self.inputs[idx], axis=0).astype(np.float32)
        this_target = np.stack(self.targets[idx], axis=0)

        if self.heatmap_smoothing_sigma != 0:
            for target_num in range(this_input.shape[0]):
                this_input[target_num] = smooth_trajectory_tensor(this_input[target_num], self.heatmap_smoothing_sigma)
                this_input[target_num] = normalize_trajectory_tensor(this_input[target_num])

        return {"inputs": this_input, "targets": this_target}


def get_coordinate_trajectory_dataset(
    inputs_path,
    departure_cameras_path,
    targets_path,
    phase,
    fold,
    batch_size,
    shuffle,
    num_workers,
    flatten_inputs,
    flatten_targets=False,
    multi_target=False,
):
    if multi_target:
        filename = phase + "_fold" + str(fold) + ".pickle"
    else:
        filename = phase + "_fold" + str(fold) + ".npy"
    inputs_path = os.path.join(inputs_path, filename)
    departure_cameras_path = os.path.join(departure_cameras_path, filename)
    targets_path = os.path.join(targets_path, filename)

    if multi_target:
        dataset = MultiTargetCoordinateTrajectoryDataset(
            inputs_path, departure_cameras_path, targets_path, flatten_inputs, flatten_targets
        )
    else:
        dataset = CoordinateTrajectoryDataset(
            inputs_path, departure_cameras_path, targets_path, flatten_inputs, flatten_targets
        )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return dataset_loader


def get_trajectory_tensor_dataset(
    inputs_path,
    targets_path,
    phase,
    fold,
    batch_size,
    shuffle,
    num_workers,
    heatmap_size,
    heatmap_smoothing_sigma,
    multi_target=False,
):
    if multi_target:
        filename = phase + "_fold" + str(fold) + ".pickle"
    else:
        filename = phase + "_fold" + str(fold) + ".npy"
    inputs_path = os.path.join(inputs_path, "size_" + str(heatmap_size[0]), filename)
    targets_path = os.path.join(targets_path, filename)

    if multi_target:
        dataset = MultiTargetTrajectoryTensorDataset(inputs_path, targets_path, heatmap_smoothing_sigma)
    else:
        dataset = TrajectoryTensorDataset(inputs_path, targets_path, heatmap_smoothing_sigma)

    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return dataset_loader
