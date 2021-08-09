import numpy as np
import argparse
import os
import math
from global_config.global_config import NUM_CAMERAS, DATA_PATH, OFFSET_LEN
import evaluation.visualization.utils as utils
import cv2
import cmapy
from experiments.utils import smooth_trajectory_tensor, normalize_trajectory_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--match_number", default=0)
parser.add_argument("--day", default=1)
args = parser.parse_args()

multi_view_tensor_data_path = os.path.join(
    os.path.join(
        DATA_PATH, "inputs", "trajectory_tensors", "size_27", "trajectory_tensors_day_" + str(args.day) + ".npy"
    )
)
input_tensor = np.load(multi_view_tensor_data_path)[int(args.match_number) * OFFSET_LEN]
print(args.day)
output_video = np.zeros((20, 1080, 1920, 3), dtype="uint8")
for camera_num in range(1, NUM_CAMERAS + 1):
    print(camera_num)
    video_path = os.path.join(
        DATA_PATH,
        "videos",
        "day_" + str(args.day),
        "departures",
        "camera_" + str(camera_num),
        "departure_" + str(args.match_number).zfill(3) + ".mp4",
    )
    video = utils.read_video(video_path)
    video = utils.resize_video(video, (480, 270))

    input_tensor = input_tensor.astype(float)
    input_tensor = smooth_trajectory_tensor(input_tensor, 1)
    input_tensor = normalize_trajectory_tensor(input_tensor)
    for frame in range(10):
        this_heatmap = input_tensor[camera_num - 1, frame]
        this_heatmap = cv2.resize(this_heatmap, dsize=(480, 270), interpolation=cv2.INTER_NEAREST)
        this_heatmap = 255 - (this_heatmap * 255)
        this_heatmap = this_heatmap.astype("uint8")
        this_heatmap = cv2.applyColorMap(this_heatmap, cmapy.cmap("coolwarm"))
        video[frame + 10] = cv2.addWeighted(this_heatmap, 0.6, video[frame + 10], 0.4, 0)
        video[frame + 10] = utils.draw_grid(video[frame + 10], (27, 48), 1)

    start_x = int(math.floor((camera_num - 1) / 4) * 270)
    start_y = int(((camera_num - 1) % 4) * 480)
    output_video[:, start_x : start_x + 270, start_y : start_y + 480, :] = video

utils.write_image("trajectory_tensor_input_example_" + args.day + "_" + args.match_number + ".jpg", output_video[10])

utils.write_video("trajectory_tensor_input_example.mp4", output_video, fps=5)
