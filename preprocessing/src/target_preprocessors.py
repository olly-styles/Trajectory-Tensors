# Internal
from global_config.global_config import (
    LABELED_TRACK_PATH,
    ALL_BOUNDING_BOXES_PATH,
    OFFSET_LEN,
    INPUT_TRAJECTORY_LENGTH,
    WHICH_TARGETS_PATH,
    FUTURE_TRAJECTORY_LENGTH,
    NUM_CAMERAS,
    WHEN_TARGETS_PATH,
    BASE_HEATMAP_SIZE,
    WHERE_TARGETS_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from preprocessing.utils.preprocessing_utils import bounding_box_to_heatmap, normalize_bounding_box

# External
import pandas as pd
import os
import numpy as np


def get_which_targets():
    for day_num in range(1, 21):
        print("Day", day_num, "of 20")

        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))

        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        inputs = np.zeros((len(departure_indexs) * OFFSET_LEN, INPUT_TRAJECTORY_LENGTH, 4), dtype="float")
        input_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")
        targets = np.zeros((len(departure_indexs) * OFFSET_LEN, 15), dtype="bool")

        for ix, departure_index in enumerate(departure_indexs):
            print("Day", day_num, ix, " of ", len(departure_indexs))
            rows = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index]
            for _, row in rows.iterrows():
                for offset in range(0, OFFSET_LEN):
                    targets[(ix * OFFSET_LEN) + offset, row["next_cam"] - 1] = 1

        save_name = "targets_day_" + str(day_num) + ".npy"
        if not os.path.exists(WHICH_TARGETS_PATH):
            os.makedirs(WHICH_TARGETS_PATH)
        np.save(os.path.join(WHICH_TARGETS_PATH, save_name), targets)


def get_when_targets():
    for day_num in range(1, 21):
        print("Day", day_num, "of 20")
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))

        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        inputs = np.zeros((len(departure_indexs) * OFFSET_LEN, INPUT_TRAJECTORY_LENGTH, 4), dtype="float")
        input_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")
        when_targets = np.zeros(
            (len(departure_indexs) * OFFSET_LEN, NUM_CAMERAS, FUTURE_TRAJECTORY_LENGTH), dtype="bool"
        )

        for ix, departure_index in enumerate(departure_indexs):
            rows = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index]
            for _, row in rows.iterrows():
                reference_track = row["track"]
                transition_time = row["transition_time"]
                next_cam = row["next_cam"]
                track_length = all_bounding_boxes.iloc[int(row["entrance_index"])]["max_track_len"]

                # Test transition times
                assert (transition_time >= 1) and (transition_time < FUTURE_TRAJECTORY_LENGTH)

                for t in range(transition_time - 1, transition_time + track_length):
                    if t >= FUTURE_TRAJECTORY_LENGTH:
                        break
                    if t == transition_time - 1:
                        first_entrance_row = all_bounding_boxes.iloc[
                            int(row["entrance_index"] - transition_time + t + 1)
                        ]
                    entrance_row = all_bounding_boxes.iloc[int(row["entrance_index"] - transition_time + t + 1)]
                    # Check if track is fragmented
                    if entrance_row["track"] == first_entrance_row["track"]:
                        when_targets[(ix * 10), next_cam - 1, int(t)] = 1
                for offset in range(1, OFFSET_LEN):
                    when_targets[(ix * 10) + offset, :, offset:] = when_targets[ix * 10, :, :-offset].copy()

        save_name = "targets_day_" + str(day_num) + ".npy"
        if not os.path.exists(WHEN_TARGETS_PATH):
            os.makedirs(WHEN_TARGETS_PATH)
        np.save(os.path.join(WHEN_TARGETS_PATH, save_name), when_targets)


def get_where_targets():
    for day_num in range(1, 21):
        print("Day", day_num, "of 20")
        labeled_bounding_boxes = pd.read_json(os.path.join(LABELED_TRACK_PATH, "day_" + str(day_num) + ".json"))

        departure_indexs = labeled_bounding_boxes["departure_index"].unique()

        all_bounding_boxes = pd.read_csv(
            os.path.join(ALL_BOUNDING_BOXES_PATH, "all_bounding_boxes_day_" + str(day_num) + ".csv")
        )

        inputs = np.zeros((len(departure_indexs) * OFFSET_LEN, INPUT_TRAJECTORY_LENGTH, 4), dtype="float")
        input_cameras = np.zeros((len(departure_indexs) * OFFSET_LEN), dtype="int")
        where_targets = np.zeros(
            (
                len(departure_indexs) * OFFSET_LEN,
                NUM_CAMERAS,
                FUTURE_TRAJECTORY_LENGTH,
                BASE_HEATMAP_SIZE[0],
                BASE_HEATMAP_SIZE[1],
            ),
            dtype="bool",
        )

        for ix, departure_index in enumerate(departure_indexs):
            rows = labeled_bounding_boxes[labeled_bounding_boxes["departure_index"] == departure_index]
            for _, row in rows.iterrows():
                reference_track = row["track"]
                transition_time = row["transition_time"]
                next_cam = row["next_cam"]
                track_length = all_bounding_boxes.iloc[int(row["entrance_index"])]["max_track_len"]
                # Test transition times
                assert (transition_time >= 1) and (transition_time < FUTURE_TRAJECTORY_LENGTH)
                for t in range(transition_time - 1, transition_time + track_length):
                    if t >= FUTURE_TRAJECTORY_LENGTH:
                        break
                    if t == transition_time - 1:
                        first_entrance_row = all_bounding_boxes.iloc[
                            int(row["entrance_index"] - transition_time + t + 1)
                        ]
                    entrance_row = all_bounding_boxes.iloc[int(row["entrance_index"] - transition_time + t + 1)]
                    if entrance_row["track"] == first_entrance_row["track"]:
                        assert entrance_row["camera"] == next_cam
                        entrance_row_box = entrance_row[["x1", "y1", "x2", "y2"]].values
                        normalized_box = normalize_bounding_box(entrance_row_box, IMAGE_WIDTH, IMAGE_HEIGHT)
                        heatmap = bounding_box_to_heatmap(normalized_box, BASE_HEATMAP_SIZE)
                        where_targets[(ix * 10), next_cam - 1, int(t)] = heatmap
                for offset in range(1, OFFSET_LEN):
                    where_targets[(ix * 10) + offset, :, offset:] = where_targets[ix * 10, :, :-offset].copy()
        save_name = "targets_day_" + str(day_num) + ".npy"
        if not os.path.exists(WHERE_TARGETS_PATH):
            os.makedirs(WHERE_TARGETS_PATH)
        np.save(os.path.join(WHERE_TARGETS_PATH, save_name), where_targets)
