import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import os


def train_model(model, device, train_loader, optimizer, loss_function, debug_mode, compute_ap=True):
    model.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)
        departure_cameras = batch["departure_cameras"].to(device)
        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        # Forward
        outputs = model(inputs, departure_cameras)
        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0

    total_loss /= len(train_loader)

    return total_loss, ap


def test_model(
    model,
    device,
    test_loader,
    loss_function,
    debug_mode,
    fold_num=None,
    predictions_save_path=None,
    predictions_save_shape=None,
    compute_ap=True,
    variable_batch_size=False,
):
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            departure_cameras = batch["departure_cameras"].to(device)
            targets = batch["targets"].to(device)
            inputs = torch.squeeze(inputs).float()
            targets = torch.squeeze(targets).float()
            departure_cameras = torch.squeeze(departure_cameras)

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            # Forward
            outputs = model(inputs, departure_cameras)
            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)

        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        np.save(os.path.join(predictions_save_path, "test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ap


def train_encoder_decoder(
    encoder,
    decoder,
    device,
    train_loader,
    optimizer,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    compute_ap=True,
):
    encoder.train()
    decoder.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)
        if coordinate_trajectory_inputs:
            departure_cameras = batch["departure_cameras"].to(device)
        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        if coordinate_trajectory_inputs:
            # Forward
            features = encoder(inputs, departure_cameras)
            outputs = decoder(features, departure_cameras)
        else:
            # Forward
            features = encoder(inputs)
            outputs = decoder(features)

        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0

    total_loss /= len(train_loader)

    return total_loss, ap


def test_encoder_decoder(
    encoder,
    decoder,
    device,
    test_loader,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    fold_num=None,
    predictions_save_path=None,
    predictions_save_shape=None,
    compute_ap=True,
    variable_batch_size=False,
):
    encoder.eval()
    decoder.eval()
    loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            if coordinate_trajectory_inputs:
                departure_cameras = batch["departure_cameras"].to(device)
            targets = batch["targets"].to(device)
            inputs = torch.squeeze(inputs).float()
            targets = torch.squeeze(targets).float()
            departure_cameras = torch.squeeze(departure_cameras)

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            if coordinate_trajectory_inputs:
                # Forward
                features = encoder(inputs, departure_cameras)
                outputs = decoder(features, departure_cameras)
            else:
                features = encoder(inputs)
                outputs = decoder(features)

            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)
        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0

        if predictions_save_path is not None:
            if not os.path.exists(predictions_save_path):
                os.makedirs(predictions_save_path)
            if predictions_save_shape is not None:
                all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
            np.save(os.path.join(predictions_save_path, "test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ap


def train_embedder_encoder_decoder(
    embedder,
    encoder,
    decoder,
    device,
    train_loader,
    optimizer,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    compute_ap=True,
):
    embedder.train()
    encoder.train()
    decoder.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)
        if coordinate_trajectory_inputs:
            departure_cameras = batch["departure_cameras"].to(device)
        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        if coordinate_trajectory_inputs:
            # Forward
            features = encoder(inputs, departure_cameras)
            outputs = decoder(features, departure_cameras)
        else:
            # Forward
            embedding = embedder(inputs)
            features = encoder(embedding)
            outputs = decoder(features)

        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0

    total_loss /= len(train_loader)

    return total_loss, ap


def test_embedder_encoder_decoder(
    embedder,
    encoder,
    decoder,
    device,
    test_loader,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    fold_num=None,
    predictions_save_path=None,
    predictions_save_shape=None,
    compute_ap=True,
    variable_batch_size=False,
):
    embedder.eval()
    encoder.eval()
    decoder.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            if coordinate_trajectory_inputs:
                departure_cameras = batch["departure_cameras"].to(device)
            targets = batch["targets"].to(device)
            inputs = torch.squeeze(inputs).float()
            targets = torch.squeeze(targets).float()

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            if coordinate_trajectory_inputs:
                # Forward
                features = encoder(inputs, departure_cameras)
                outputs = decoder(features, departure_cameras)
            else:
                # Forward
                embedding = embedder(inputs)
                features = encoder(embedding)
                outputs = decoder(features)

            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        np.save(os.path.join(predictions_save_path, "test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ap


def train_with_spatial_upsample(
    encoder,
    temporal_decoder,
    spatial_decoder,
    device,
    train_loader,
    optimizer,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    compute_ap=True,
):
    encoder.train()
    temporal_decoder.train()
    spatial_decoder.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)
        if coordinate_trajectory_inputs:
            departure_cameras = batch["departure_cameras"].to(device)
        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        if coordinate_trajectory_inputs:
            # Forward
            features = encoder(inputs, departure_cameras)
            temporal_upsample = temporal_decoder(features, departure_cameras)
            outputs = spatial_decoder(temporal_upsample, departure_cameras)
        else:
            # Forward
            features = encoder(inputs)
            temporal_upsample = temporal_decoder(features)
            outputs = spatial_decoder(temporal_upsample)

        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0

    total_loss /= len(train_loader)

    return total_loss, ap


def test_with_spatial_upsample(
    encoder,
    temporal_decoder,
    spatial_decoder,
    device,
    test_loader,
    loss_function,
    debug_mode,
    coordinate_trajectory_inputs=True,
    fold_num=None,
    predictions_save_path=None,
    predictions_save_shape=None,
    compute_ap=True,
    variable_batch_size=False,
):
    encoder.eval()
    temporal_decoder.eval()
    spatial_decoder.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            if coordinate_trajectory_inputs:
                departure_cameras = batch["departure_cameras"].to(device)
            targets = batch["targets"].to(device)
            inputs = torch.squeeze(inputs).float()
            targets = torch.squeeze(targets).float()
            departure_cameras = torch.squeeze(departure_cameras)

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            if coordinate_trajectory_inputs:
                # Forward
                features = encoder(inputs, departure_cameras)
                temporal_upsample = temporal_decoder(features, departure_cameras)
                outputs = spatial_decoder(temporal_upsample, departure_cameras)
            else:
                # Forward
                features = encoder(inputs)
                temporal_upsample = temporal_decoder(features)
                outputs = spatial_decoder(temporal_upsample)

            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)

        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        print("saving to ", predictions_save_path)
        np.save(os.path.join(predictions_save_path, "test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ap


def train_with_embedder_and_spatial_upsample(
    embedder,
    encoder,
    temporal_decoder,
    spatial_decoder,
    device,
    train_loader,
    optimizer,
    loss_function,
    debug_mode,
    compute_ap=True,
):
    embedder.train()
    encoder.train()
    temporal_decoder.train()
    spatial_decoder.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)

        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        # Forward
        embedded_inputs = embedder(inputs)
        features = encoder(embedded_inputs)
        temporal_upsample = temporal_decoder(features)
        outputs = spatial_decoder(temporal_upsample)
        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0

    total_loss /= len(train_loader)

    return total_loss, ap


def test_with_embedder_and_spatial_upsample(
    embedder,
    encoder,
    temporal_decoder,
    spatial_decoder,
    device,
    test_loader,
    loss_function,
    debug_mode,
    fold_num=None,
    predictions_save_path=None,
    predictions_save_shape=None,
    compute_ap=True,
    variable_batch_size=False,
):
    embedder.eval()
    encoder.eval()
    temporal_decoder.eval()
    spatial_decoder.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            inputs = torch.squeeze(inputs).float()
            targets = torch.squeeze(targets).float()

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            # Forward
            embedded_inputs = embedder(inputs)
            features = encoder(embedded_inputs)
            temporal_upsample = temporal_decoder(features)
            outputs = spatial_decoder(temporal_upsample)

            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx : (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)
        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        print("saving to ", predictions_save_path)
        np.save(os.path.join(predictions_save_path, "test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ap
