# Internal
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
)
from preprocessing.utils.preprocessing_utils import normalize_bounding_box, bounding_box_to_heatmap

# External
import pandas as pd
import numpy as np
import os


def get_coordinate_trajectory_inputs():
    for day_num in range(1, 21):
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))

        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        inputs = np.zeros((len(departure_indexs) * OFFSET_LEN, INPUT_TRAJECTORY_LENGTH, 4), dtype="float")
        departure_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")

        for ix, departure_index in enumerate(departure_indexs):
            print("Day", day_num, ix, " of ", len(departure_indexs))
            row = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index].head()
            reference_track = row["track"].values[0]

            for offset in range(0, OFFSET_LEN):
                departure_camera = row["camera"].values[0]
                departure_cameras[(ix * OFFSET_LEN) + offset] = departure_camera
                for t in range(0, INPUT_TRAJECTORY_LENGTH):
                    this_input_track = int(
                        all_bounding_boxes.iloc[int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)][
                            "track"
                        ]
                    )
                    assert reference_track == this_input_track

                    bounding_box = all_bounding_boxes.iloc[
                        int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)
                    ][["x1", "y1", "x2", "y2"]].astype(int)

                    normalized_bounding_box = normalize_bounding_box(bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT)

                    inputs[(ix * OFFSET_LEN) + offset, t, :] = normalized_bounding_box

        save_path_coordinate_trajectory_inputs = os.path.join(DATA_PATH, "inputs", "input_coordinate_trajectories")
        save_path_departure_cameras = os.path.join(DATA_PATH, "inputs", "departure_cameras")

        if not os.path.exists(save_path_coordinate_trajectory_inputs):
            os.makedirs(save_path_coordinate_trajectory_inputs)
        if not os.path.exists(save_path_departure_cameras):
            os.makedirs(save_path_departure_cameras)

        np.save(os.path.join(save_path_coordinate_trajectory_inputs, "inputs_day_" + str(day_num) + ".npy"), inputs)
        np.save(
            os.path.join(save_path_departure_cameras, "departure_cameras_day_" + str(day_num) + ".npy"),
            departure_cameras,
        )


def get_trajectory_tensor_inputs(heatmap_size):
    for day_num in range(1, 21):
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))
        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        trajectory_tensors = np.zeros(
            ((len(departure_indexs) * OFFSET_LEN, NUM_CAMERAS, INPUT_TRAJECTORY_LENGTH) + heatmap_size), dtype=np.bool
        )
        departure_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")
        targets = np.load(os.path.join(WHICH_TARGETS_PATH, "targets_day_" + str(day_num) + ".npy"))

        for ix, departure_index in enumerate(departure_indexs):
            if ix % 20 == 0:
                print("Day", day_num, ix, " of ", len(departure_indexs))
            row = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index].head()
            reference_track = row["track"].values[0]

            for offset in range(0, OFFSET_LEN):
                departure_camera = row["camera"].values[0]
                departure_cameras[(ix * OFFSET_LEN) + offset] = departure_camera
                for t in range(0, INPUT_TRAJECTORY_LENGTH):
                    this_input_track = int(
                        all_bounding_boxes.iloc[int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)][
                            "track"
                        ]
                    )
                    assert reference_track == this_input_track
                    bounding_box = all_bounding_boxes.iloc[
                        int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)
                    ][["x1", "y1", "x2", "y2"]].astype(int)

                    normalized_bounding_box = normalize_bounding_box(bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                    heatmap = bounding_box_to_heatmap(normalized_bounding_box, heatmap_size)
                    input_num = (ix * OFFSET_LEN) + offset
                    trajectory_tensors[input_num, departure_camera - 1, t, :] = heatmap

            reference_track = row["track"].values[0]
            frame_num = row["frame_num"].values[0]
            next_camera = row["next_cam"].values[0]
            hour = row["hour"].values[0]
            camera = row["camera"].values[0]
            candidate_boxes = all_bounding_boxes[all_bounding_boxes["hour"] == hour]
            candidate_boxes = candidate_boxes[
                (candidate_boxes["frame_num"] <= frame_num) & (candidate_boxes["frame_num"] > frame_num - 20)
            ]
            candidate_boxes = candidate_boxes[candidate_boxes["camera"] != camera]
            candidate_boxes["cameratrack"] = (
                candidate_boxes["camera"].astype(str) + "-" + candidate_boxes["track"].astype(str)
            )
            # Alternative view heatmaps
            if row["alternate_camtracks"].values[0] == 0:
                continue
            for camtrack in row["alternate_camtracks"].values[0][0]:
                camtrack_boxes = candidate_boxes[candidate_boxes["cameratrack"] == camtrack]
                camera = camtrack_boxes["camera"].values[0]
                # Alternetive view cannot be one of the target cameras
                target = np.argwhere(targets[ix * OFFSET_LEN])
                if (camera - 1) in target:
                    continue
                for offset in range(0, OFFSET_LEN):
                    for t in range(0, INPUT_TRAJECTORY_LENGTH):
                        frame = frame_num - OFFSET_LEN - offset + t
                        frame_row = camtrack_boxes[camtrack_boxes["frame_num"] == frame]
                        if len(frame_row) == 1:
                            bounding_box = frame_row[["x1", "y1", "x2", "y2"]].values[0]
                            input_num = (ix * OFFSET_LEN) + offset
                            normalized_bounding_box = normalize_bounding_box(bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                            heatmap = bounding_box_to_heatmap(normalized_bounding_box, heatmap_size)
                            trajectory_tensors[input_num, camera - 1, t, :] = heatmap
        save_path_trajectory_tensor_inputs = os.path.join(
            DATA_PATH, "inputs", "trajectory_tensors", "size_" + str(heatmap_size[0])
        )

        if not os.path.exists(save_path_trajectory_tensor_inputs):
            os.makedirs(save_path_trajectory_tensor_inputs)

        np.save(
            os.path.join(save_path_trajectory_tensor_inputs, "trajectory_tensors_day_" + str(day_num) + ".npy"),
            trajectory_tensors,
        )


def get_trajectory_tensor_single_view_inputs(heatmap_size):
    for day_num in range(1, 21):
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))
        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        trajectory_tensors = np.zeros(
            ((len(departure_indexs) * OFFSET_LEN, NUM_CAMERAS, INPUT_TRAJECTORY_LENGTH) + heatmap_size), dtype=np.bool
        )
        departure_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")

        for ix, departure_index in enumerate(departure_indexs):
            print("Day", day_num, ix, " of ", len(departure_indexs))
            row = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index].head()
            reference_track = row["track"].values[0]

            for offset in range(0, OFFSET_LEN):
                departure_camera = row["camera"].values[0]
                departure_cameras[(ix * OFFSET_LEN) + offset] = departure_camera
                for t in range(0, INPUT_TRAJECTORY_LENGTH):
                    this_input_track = int(
                        all_bounding_boxes.iloc[int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)][
                            "track"
                        ]
                    )
                    assert reference_track == this_input_track

                    bounding_box = all_bounding_boxes.iloc[
                        int(row["departure_index"].values[0] - OFFSET_LEN - offset + t)
                    ][["x1", "y1", "x2", "y2"]].astype(int)

                    normalized_bounding_box = normalize_bounding_box(bounding_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                    heatmap = bounding_box_to_heatmap(normalized_bounding_box, heatmap_size)
                    input_num = (ix * OFFSET_LEN) + offset
                    trajectory_tensors[input_num, departure_camera - 1, t, :] = heatmap

        save_path_trajectory_tensor_inputs = os.path.join(
            DATA_PATH, "inputs", "trajectory_tensors_single_view", "size_" + str(heatmap_size[0])
        )

        if not os.path.exists(save_path_trajectory_tensor_inputs):
            os.makedirs(save_path_trajectory_tensor_inputs)

        np.save(
            os.path.join(save_path_trajectory_tensor_inputs, "trajectory_tensors_day_" + str(day_num) + ".npy"),
            trajectory_tensors,
        )
