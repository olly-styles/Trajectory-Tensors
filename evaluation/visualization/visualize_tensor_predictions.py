import numpy as np
import argparse
import os
import math
from global_config.global_config import NUM_CAMERAS, DATA_PATH, OFFSET_LEN
import evaluation.visualization.utils as utils
import cv2
import cmapy


parser = argparse.ArgumentParser()
parser.add_argument("--match_number", default=0)
parser.add_argument("--day", default=1)
parser.add_argument("--model", default="ground_truth")
args = parser.parse_args()

fold, index = utils.get_fold_and_index(args.day, args.match_number)
print(fold, index)

if args.model == "ground_truth":
    tensor_data_path = os.path.join(
        os.path.join(DATA_PATH, "cross_validation", "targets", "where_targets", "test_fold" + str(fold) + ".npy")
    )
    tensor = np.load(tensor_data_path)[index]
else:
    tensor_data_path = os.path.join(
        os.path.join(
            DATA_PATH, "cross_validation", "predictions", "where", args.model, "test_fold" + str(fold) + ".npy"
        )
    )
    tensor = np.load(tensor_data_path)[index]

output_video = np.zeros((60, 1080, 1920, 3), dtype="uint8")
for camera_num in range(1, NUM_CAMERAS + 1):
    print(camera_num)
    video_path = os.path.join(
        DATA_PATH,
        "videos",
        "day_" + str(args.day),
        "entrances",
        "camera_" + str(camera_num),
        "entrance_" + str(args.match_number).zfill(3) + ".mp4",
    )
    video = utils.read_video(video_path)
    video = utils.resize_video(video, (480, 270))

    tensor = tensor.astype(float)
    tensor = tensor / tensor.max()
    for frame in range(60):
        this_heatmap = tensor[camera_num - 1, frame]
        this_heatmap = cv2.resize(this_heatmap, dsize=(480, 270), interpolation=cv2.INTER_NEAREST)
        this_heatmap = 255 - (this_heatmap * 255)
        this_heatmap = this_heatmap.astype("uint8")
        this_heatmap = cv2.applyColorMap(this_heatmap, cmapy.cmap("coolwarm"))
        video[frame] = cv2.addWeighted(this_heatmap, 0.6, video[frame], 0.4, 0)
        video[frame] = utils.draw_grid(video[frame], (9, 16), 1)

    start_x = int(math.floor((camera_num - 1) / 4) * 270)
    start_y = int(((camera_num - 1) % 4) * 480)
    output_video[:, start_x : start_x + 270, start_y : start_y + 480, :] = video

utils.write_image(
    "trajectory_tensor_prediction_example_" + args.day + "_" + args.match_number + ".jpg", output_video[10]
)

utils.write_video(
    "trajectory_tensor_prediction_example"
    + args.day
    + "_"
    + args.match_number
    + "_"
    + args.model.split("/")[-1]
    + ".mp4",
    output_video,
    fps=5,
)
