import pandas as pd
import numpy as np
import os
import pickle
from global_config.global_config import (
    LABELED_TRACK_PATH,
    ALL_BOUNDING_BOXES_PATH,
    OFFSET_LEN,
    INPUT_TRAJECTORY_LENGTH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    DATA_PATH,
    NUM_CAMERAS,
    INPUT_TRAJECTORY_LENGTH,
    WHICH_TARGETS_PATH,
    MAX_FRAMES_BETWEEN_MULTI_TARGET_TRAJECTORIES,
    UNTRIMMED_VIDEO_LENGTH_FRAMES,
    FUTURE_TRAJECTORY_LENGTH,
    BASE_HEATMAP_SIZE,
)
from preprocessing.utils.preprocessing_utils import normalize_bounding_box, bounding_box_to_heatmap


def get_all_multi_target_inputs(heatmap_size):
    total_data_samples = 0
    total_trajectories = 0
    max_num_trajectories = 0
    for day_num in range(1, 21):
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))
        labeled_bounding_boxes = labeled_bounding_boxes.drop_duplicates(subset=["departure_index"]).reset_index()
        labeled_bounding_boxes = labeled_bounding_boxes.drop_duplicates(subset="entrance_index")
        exit_frame_bins = np.arange(0, UNTRIMMED_VIDEO_LENGTH_FRAMES, MAX_FRAMES_BETWEEN_MULTI_TARGET_TRAJECTORIES)
        labeled_bounding_boxes["frame_num_bin"] = pd.cut(labeled_bounding_boxes.frame_num, exit_frame_bins)
        min_track_length = OFFSET_LEN + MAX_FRAMES_BETWEEN_MULTI_TARGET_TRAJECTORIES + INPUT_TRAJECTORY_LENGTH
        labeled_bounding_boxes = labeled_bounding_boxes[labeled_bounding_boxes["track_length"] >= min_track_length]
        multi_target_indexes = (
            labeled_bounding_boxes.groupby(["frame_num_bin", "hour"])["entrance_index"]
            .filter(lambda x: x.nunique() > 1)
            .index
        )
        multi_target_boxes = labeled_bounding_boxes.loc[multi_target_indexes].drop_duplicates(subset="entrance_index")
        if len(multi_target_boxes) > 0:
            max_num_trajectories_for_day = int(multi_target_boxes.groupby(["frame_num_bin", "hour"]).count().max()[0])
        else:
            max_num_trajectories_for_day = 0
        if max_num_trajectories_for_day > max_num_trajectories:
            max_num_trajectories = max_num_trajectories_for_day
        total_trajectories += len(multi_target_boxes)
        multi_target_boxes["bin_hour_index"] = (
            multi_target_boxes["frame_num_bin"].astype(str) + "-" + multi_target_boxes["hour"].astype(str)
        )
        bin_hour_indexs = multi_target_boxes["bin_hour_index"].unique()
        total_data_samples_day = len(bin_hour_indexs)
        total_data_samples += total_data_samples_day
        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        departure_cameras = [[]] * total_data_samples_day * OFFSET_LEN
        which_targets = [[]] * total_data_samples_day * OFFSET_LEN
        when_targets = [[]] * total_data_samples_day * OFFSET_LEN
        where_targets = [[]] * total_data_samples_day * OFFSET_LEN
        trajectory_tensors = [[]] * total_data_samples_day * OFFSET_LEN
        coordinate_trajectory_inputs = [[]] * total_data_samples_day * OFFSET_LEN
        for ix, bin_hour_index in enumerate(bin_hour_indexs):
            rows = multi_target_boxes[multi_target_boxes["bin_hour_index"] == bin_hour_index]
            departure_indexes = rows["departure_index"]
            departure_rows = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"].isin(departure_indexes)]
            departure_rows = departure_rows.reset_index()
            for departure_ix, departure_row in departure_rows.iterrows():
                next_cam = departure_row["next_cam"] - 1
                transition_time = departure_row["transition_time"]
                reference_track = departure_row["track"]
                track_length = all_bounding_boxes.iloc[int(departure_row["entrance_index"])]["max_track_len"]

                # Test transition times
                assert (transition_time >= 1) and (transition_time < FUTURE_TRAJECTORY_LENGTH)
                when_target = np.zeros((NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH), dtype=np.bool)
                where_target = np.zeros(
                    (NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1]), dtype=np.bool
                )

                for t in range(transition_time - 1, transition_time + track_length):
                    if t >= FUTURE_TRAJECTORY_LENGTH:
                        break
                    if t == transition_time - 1:
                        first_entrance_row = all_bounding_boxes.iloc[
                            int(departure_row["entrance_index"] - transition_time + t + 1)
                        ]
                    entrance_row = all_bounding_boxes.iloc[
                        int(departure_row["entrance_index"] - transition_time + t + 1)
                    ]
                    # Check if track is fragmented
                    if entrance_row["track"] == first_entrance_row["track"]:
                        # When target
                        when_target[next_cam, int(t)] = 1
                        # Where target
                        entrance_row_box = entrance_row[["x1", "y1", "x2", "y2"]].values
                        normalized_box = normalize_bounding_box(entrance_row_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                        heatmap = bounding_box_to_heatmap(normalized_box, BASE_HEATMAP_SIZE)
                        where_target[next_cam, int(t)] = heatmap

                when_targets[(ix * OFFSET_LEN)] = when_targets[(ix * OFFSET_LEN)] + [when_target]
                where_targets[(ix * OFFSET_LEN)] = where_targets[(ix * OFFSET_LEN)] + [where_target]

                for offset in range(0, OFFSET_LEN):
                    dataset_index = (ix * OFFSET_LEN) + offset
                    if offset > 0:
                        multi_target_when_target = []
                        multi_target_where_target = []
                        for target in range(len(when_targets[(ix * 10)])):
                            when_target = np.zeros((NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH), dtype=np.bool)
                            when_target[:, offset:] = when_targets[(ix * 10)][target][:, :-offset].copy()
                            multi_target_when_target.append(when_target)

                            where_target = np.zeros(
                                (NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH, BASE_HEATMAP_SIZE[0], BASE_HEATMAP_SIZE[1]),
                                dtype=np.bool,
                            )
                            where_target[:, offset:] = where_targets[(ix * 10)][target][:, :-offset].copy()
                            multi_target_where_target.append(where_target)

                        when_targets[dataset_index] = multi_target_when_target
                        where_targets[dataset_index] = multi_target_where_target

                    # Which labels
                    target = np.zeros((NUM_CAMERAS))
                    coordinate_trajectory = np.zeros((INPUT_TRAJECTORY_LENGTH, 4))

                    target[next_cam] = 1

                    which_targets[dataset_index] = which_targets[dataset_index] + [target]

                    for i in range(len(when_targets[dataset_index])):
                        where_cameras = (
                            np.argwhere(where_targets[dataset_index][i].sum(axis=3).sum(axis=2).sum(axis=1) > 1) + 1
                        )
                        when_cameras = np.argwhere(when_targets[dataset_index][i].sum(axis=1) > 1) + 1
                        which_cameras = np.argwhere(which_targets[dataset_index][i]) + 1
                        assert which_cameras == when_cameras == where_cameras

                    # Departure cameras
                    departure_camera = departure_row["camera"]
                    departure_cameras[dataset_index] = departure_cameras[dataset_index] + [departure_camera]

                    # Trajectory tensors
                    trajectory_tensor = np.zeros(((NUM_CAMERAS, INPUT_TRAJECTORY_LENGTH) + heatmap_size), dtype=np.bool)
                    for t in range(0, INPUT_TRAJECTORY_LENGTH):
                        this_input_track = int(
                            all_bounding_boxes.iloc[int(departure_row["departure_index"] - OFFSET_LEN - offset + t)][
                                "track"
                            ]
                        )
                        entrance_row = all_bounding_boxes.iloc[
                            int(departure_row["entrance_index"] - OFFSET_LEN - offset - transition_time + t)
                        ]

                        bounding_box = all_bounding_boxes.iloc[
                            int(departure_row["departure_index"] - OFFSET_LEN - offset + t)
                        ][["x1", "y1", "x2", "y2"]].astype(int)
                        normalized_bounding_box = normalize_bounding_box(bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                        heatmap = bounding_box_to_heatmap(normalized_bounding_box, heatmap_size)
                        trajectory_tensor[departure_camera - 1, t, :] = heatmap
                        coordinate_trajectory[t, :] = normalized_bounding_box
                    trajectory_tensors[dataset_index] = trajectory_tensors[dataset_index] + [trajectory_tensor]
                    coordinate_trajectory_inputs[dataset_index] = coordinate_trajectory_inputs[dataset_index] + [
                        coordinate_trajectory
                    ]

                # Alternative view heatmaps
                frame_num = departure_row["frame_num"]
                hour = departure_row["hour"]
                camera = departure_row["camera"]
                candidate_boxes = all_bounding_boxes[all_bounding_boxes["hour"] == hour]
                candidate_boxes = candidate_boxes[
                    (candidate_boxes["frame_num"] <= frame_num) & (candidate_boxes["frame_num"] > frame_num - 20)
                ]
                candidate_boxes = candidate_boxes[candidate_boxes["camera"] != camera]
                candidate_boxes["cameratrack"] = (
                    candidate_boxes["camera"].astype(str) + "-" + candidate_boxes["track"].astype(str)
                )
                if departure_row["alternate_camtracks"] == 0:
                    continue
                for camtrack in departure_row["alternate_camtracks"][0]:
                    camtrack_boxes = candidate_boxes[candidate_boxes["cameratrack"] == camtrack]
                    camera = camtrack_boxes["camera"].values[0]
                    # Alternetive view cannot be one of the target cameras
                    target = np.argwhere(which_targets[ix * OFFSET_LEN][departure_ix])[0]
                    if (camera - 1) in target:
                        continue
                    for offset in range(0, OFFSET_LEN):
                        for t in range(0, INPUT_TRAJECTORY_LENGTH):
                            frame = frame_num - OFFSET_LEN - offset + t
                            frame_row = camtrack_boxes[camtrack_boxes["frame_num"] == frame]
                            if len(frame_row) == 1:
                                bounding_box = frame_row[["x1", "y1", "x2", "y2"]].values[0]
                                input_num = (ix * OFFSET_LEN) + offset
                                normalized_bounding_box = normalize_bounding_box(
                                    bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT
                                )
                                heatmap = bounding_box_to_heatmap(normalized_bounding_box, heatmap_size)
                                trajectory_tensors[dataset_index][departure_ix][camera - 1, t, :] = heatmap
                                coordinate_trajectory_inputs[dataset_index][departure_ix][
                                    t, :
                                ] = normalized_bounding_box

        save_path_trajectory_tensor_inputs = os.path.join(
            DATA_PATH, "multi_target", "inputs", "trajectory_tensors", "size_" + str(heatmap_size[0])
        )
        save_path_which_targets = os.path.join(DATA_PATH, "multi_target", "targets", "which")
        save_path_when_targets = os.path.join(DATA_PATH, "multi_target", "targets", "when")
        save_path_where_targets = os.path.join(DATA_PATH, "multi_target", "targets", "where")
        save_path_departure_cameras = os.path.join(DATA_PATH, "multi_target", "inputs", "departure_cameras")
        save_path_coordinate_trajectory_inputs = os.path.join(
            DATA_PATH, "multi_target", "inputs", "coordinate_trajectories"
        )

        if not os.path.exists(save_path_trajectory_tensor_inputs):
            os.makedirs(save_path_trajectory_tensor_inputs)
        if not os.path.exists(save_path_which_targets):
            os.makedirs(save_path_which_targets)
        if not os.path.exists(save_path_when_targets):
            os.makedirs(save_path_when_targets)
        if not os.path.exists(save_path_where_targets):
            os.makedirs(save_path_where_targets)
        if not os.path.exists(save_path_departure_cameras):
            os.makedirs(save_path_departure_cameras)
        if not os.path.exists(save_path_coordinate_trajectory_inputs):
            os.makedirs(save_path_coordinate_trajectory_inputs)

        with open(
            os.path.join(save_path_trajectory_tensor_inputs, "trajectory_tensors_day_" + str(day_num) + ".pickle"), "wb"
        ) as fp:
            pickle.dump(trajectory_tensors, fp)
        with open(os.path.join(save_path_which_targets, "targets_day_" + str(day_num) + ".pickle"), "wb") as fp:
            pickle.dump(which_targets, fp)
        with open(os.path.join(save_path_when_targets, "targets_day_" + str(day_num) + ".pickle"), "wb") as fp:
            pickle.dump(when_targets, fp)
        with open(os.path.join(save_path_where_targets, "targets_day_" + str(day_num) + ".pickle"), "wb") as fp:
            pickle.dump(where_targets, fp)
        with open(
            os.path.join(save_path_departure_cameras, "departure_cameras_day_" + str(day_num) + ".pickle"), "wb"
        ) as fp:
            pickle.dump(departure_cameras, fp)
        with open(
            os.path.join(save_path_coordinate_trajectory_inputs, "inputs_day_" + str(day_num) + ".pickle"), "wb"
        ) as fp:
            pickle.dump(coordinate_trajectory_inputs, fp)

    print(
        f"Multi-target dataset stats. \
        Total data samples: {total_data_samples} \
        Total trajectories: {total_trajectories} \
        Max trajectories for a single sample: {max_num_trajectories}"
    )
