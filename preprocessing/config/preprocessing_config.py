import os
from global_config.global_config import DATA_PATH

# Dataset details
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FRAME_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CAMERAS = 15

# Trajectory config
BASE_HEATMAP_SIZE = (9, 16)
OFFSET_LEN = 10
INPUT_TRAJECTORY_LENGTH = 10
FUTURE_TRAJECTORY_LENGTH = 60

# Data paths
LABELED_TRACK_PATH = os.path.join(DATA_PATH, "verified_cross_camera_matches")
ENTRANCES_DEPARTURES_PATH = os.path.join(DATA_PATH, "entrances_and_departures")
ALL_BOUNDING_BOXES_PATH = os.path.join(DATA_PATH, "all_bounding_boxes")
WHICH_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "which")
WHEN_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "when")
WHERE_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "where")
