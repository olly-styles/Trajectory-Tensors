import os
import numpy as np

# Dataset details
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FRAME_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CAMERAS = 15
NUM_CROSS_VAL_FOLDS = 5
NUM_DAYS = 20
UNTRIMMED_VIDEO_LENGTH_FRAMES = 6000

# Trajectory config
BASE_HEATMAP_SIZE = (9, 16)
OFFSET_LEN = 10
INPUT_TRAJECTORY_LENGTH = 10
FUTURE_TRAJECTORY_LENGTH = 60
MAX_FRAMES_BETWEEN_MULTI_TARGET_TRAJECTORIES = 10

GRID_CELL_SIZE = FRAME_SIZE[0] / BASE_HEATMAP_SIZE[0]


# Camera distances ranked for shortest_realworld_distance baseline
CAMERA_DISTANCE_RANKING = np.array(
    [
        [15, 2, 3, 4, 14, 13, 12, 5, 6, 7, 11, 10, 9, 8],
        [3, 4, 13, 14, 12, 1, 5, 6, 7, 15, 11, 10, 9, 8],
        [2, 4, 13, 14, 12, 1, 5, 6, 7, 15, 11, 10, 9, 8],
        [3, 2, 13, 12, 14, 5, 1, 6, 7, 15, 11, 10, 9, 8],
        [6, 7, 10, 11, 9, 4, 3, 2, 12, 13, 14, 8, 1, 15],
        [5, 7, 10, 9, 11, 4, 3, 2, 12, 13, 14, 1, 15, 8],
        [6, 5, 10, 9, 11, 8, 4, 3, 2, 12, 13, 14, 1, 15],
        [9, 10, 11, 7, 6, 5, 12, 13, 14, 4, 3, 2, 15, 1],
        [10, 11, 6, 7, 5, 8, 12, 13, 14, 4, 3, 2, 15, 1],
        [9, 11, 6, 7, 5, 8, 12, 13, 14, 4, 3, 2, 15, 1],
        [10, 9, 6, 5, 7, 12, 13, 14, 8, 4, 3, 2, 15, 1],
        [13, 14, 3, 4, 2, 11, 10, 9, 5, 6, 7, 8, 15, 1],
        [12, 14, 3, 2, 4, 11, 10, 9, 15, 5, 6, 7, 1, 8],
        [13, 12, 3, 2, 4, 15, 11, 10, 9, 1, 5, 6, 7, 8],
        [1, 14, 13, 12, 2, 3, 4, 11, 10, 9, 5, 6, 7, 8],
    ]
)


# Data paths
DATA_PATH = os.path.join("WNMF-dataset")
LABELED_TRACK_PATH = os.path.join(DATA_PATH, "verified_cross_camera_matches")
ENTRANCES_DEPARTURES_PATH = os.path.join(DATA_PATH, "entrances_and_departures")
ALL_BOUNDING_BOXES_PATH = os.path.join(DATA_PATH, "all_bounding_boxes")
WHICH_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "which")
WHEN_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "when")
WHERE_TARGETS_PATH = os.path.join(DATA_PATH, "targets", "where")
COODINATE_TRAJECTORY_INPUTS_PATH = os.path.join(DATA_PATH, "inputs", "input_coordinate_trajectories")
DEPARTURE_CAMERAS_PATH = os.path.join(DATA_PATH, "inputs", "departure_cameras")

# Cross validation data paths
CROSS_VALIDATION_PATH = os.path.join(DATA_PATH, "cross_validation")
CROSS_VALIDATION_INPUTS_PATH = os.path.join(CROSS_VALIDATION_PATH, "inputs")
CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH = os.path.join(CROSS_VALIDATION_INPUTS_PATH, "coordinate_trajectories")
CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH = os.path.join(CROSS_VALIDATION_INPUTS_PATH, "departure_cameras")
CROSS_VALIDATION_MULTI_VIEW_TRAJECTORY_TENSORS_PATH = os.path.join(CROSS_VALIDATION_INPUTS_PATH, "trajectory_tensors")
CROSS_VALIDATION_SINGLE_VIEW_TRAJECTORY_TENSORS_PATH = os.path.join(
    CROSS_VALIDATION_INPUTS_PATH, "trajectory_tensors_single_view"
)
CROSS_VALIDATION_HAND_CRAFTED_FEATURES_PATH = os.path.join(CROSS_VALIDATION_INPUTS_PATH, "hand_crafted_features")


CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH = os.path.join(CROSS_VALIDATION_PATH, "grid_search_results", "which")
CROSS_VALIDATION_WHEN_GRID_SEARCH_PATH = os.path.join(CROSS_VALIDATION_PATH, "grid_search_results", "when")
CROSS_VALIDATION_WHERE_GRID_SEARCH_PATH = os.path.join(CROSS_VALIDATION_PATH, "grid_search_results", "where")


CROSS_VALIDATION_WHICH_TARGETS_PATH = os.path.join(CROSS_VALIDATION_PATH, "targets", "which_targets")
CROSS_VALIDATION_WHEN_TARGETS_PATH = os.path.join(CROSS_VALIDATION_PATH, "targets", "when_targets")
CROSS_VALIDATION_WHERE_TARGETS_PATH = os.path.join(CROSS_VALIDATION_PATH, "targets", "where_targets")
CROSS_VALIDATION_PREDICTIONS_PATH = os.path.join(CROSS_VALIDATION_PATH, "predictions")
CROSS_VALIDATION_MODELS_PATH_WHICH = os.path.join(CROSS_VALIDATION_PATH, "models", "which")
CROSS_VALIDATION_MODELS_PATH_WHEN = os.path.join(CROSS_VALIDATION_PATH, "models", "when")
CROSS_VALIDATION_MODELS_PATH_WHERE = os.path.join(CROSS_VALIDATION_PATH, "models", "where")
CROSS_VALIDATION_MODELS_PATH_AUTOENCODER = os.path.join(CROSS_VALIDATION_PATH, "models", "autoencoder")

# WHICH predictions path
CROSS_VALIDATION_WHICH_PREDICTIONS_PATH = os.path.join(CROSS_VALIDATION_PREDICTIONS_PATH, "which")
CROSS_VALIDATION_WHICH_TRAINING_SET_MEAN_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "baselines", "training_set_mean"
)
CROSS_VALIDATION_WHICH_SHORTEST_REALWORLD_DISTANCE_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "baselines", "shortest_realworld_distance"
)
CROSS_VALIDATION_WHICH_MOST_SIMILAR_TRAJECTORY_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "baselines", "most_similar_trajectory"
)
CROSS_VALIDATION_WHICH_HAND_CRAFTED_FEATURE_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "baselines", "hand_crafted_features"
)
CROSS_VALIDATION_WHICH_GRU_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "coordinate_trajectories", "gru"
)
CROSS_VALIDATION_WHICH_LSTM_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "coordinate_trajectories", "lstm"
)
CROSS_VALIDATION_WHICH_1DCNN_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "coordinate_trajectories", "1dcnn"
)

CROSS_VALIDATION_WHICH_3DCNN_PATH = os.path.join(CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn")
CROSS_VALIDATION_WHICH_2D1DCNN_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn"
)
CROSS_VALIDATION_WHICH_CNN_GRU_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru"
)

CROSS_VALIDATION_WHICH_3DCNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn_single_view"
)
CROSS_VALIDATION_WHICH_2D1DCNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn_single_view"
)
CROSS_VALIDATION_WHICH_CNN_GRU_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHICH_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru_single_view"
)

# WHEN predictions path
CROSS_VALIDATION_WHEN_PREDICTIONS_PATH = os.path.join(CROSS_VALIDATION_PREDICTIONS_PATH, "when")
CROSS_VALIDATION_WHEN_MOST_SIMILAR_TRAJECTORY_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "baselines", "most_similar_trajectory"
)
CROSS_VALIDATION_WHEN_TRAINING_SET_MEAN_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "baselines", "training_set_mean"
)
CROSS_VALIDATION_WHEN_HAND_CRAFTED_FEATURE_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "baselines", "hand_crafted_features"
)
CROSS_VALIDATION_WHEN_1DCNN_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "coordinate_trajectories", "1dcnn"
)
CROSS_VALIDATION_WHEN_LSTM_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "coordinate_trajectories", "lstm"
)
CROSS_VALIDATION_WHEN_GRU_PATH = os.path.join(CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "coordinate_trajectories", "gru")
CROSS_VALIDATION_WHEN_3D_CNN_PATH = os.path.join(CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn")
CROSS_VALIDATION_WHEN_2D_1D_CNN_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn"
)
CROSS_VALIDATION_WHEN_CNN_GRU_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru"
)

CROSS_VALIDATION_WHEN_3D_CNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn_single_view"
)
CROSS_VALIDATION_WHEN_2D_1D_CNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn_single_view"
)
CROSS_VALIDATION_WHEN_CNN_GRU_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHEN_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru_single_view"
)

# WHERE predictions path
CROSS_VALIDATION_WHERE_PREDICTIONS_PATH = os.path.join(CROSS_VALIDATION_PREDICTIONS_PATH, "where")
CROSS_VALIDATION_WHERE_MOST_SIMILAR_TRAJECTORY_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "baselines", "most_similar_trajectory"
)
CROSS_VALIDATION_WHERE_TRAINING_SET_MEAN_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "baselines", "training_set_mean"
)
CROSS_VALIDATION_WHERE_HAND_CRAFTED_FEATURE_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "baselines", "hand_crafted_features"
)
CROSS_VALIDATION_WHERE_1DCNN_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "coordinate_trajectories", "1dcnn"
)
CROSS_VALIDATION_WHERE_LSTM_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "coordinate_trajectories", "lstm"
)
CROSS_VALIDATION_WHERE_GRU_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "coordinate_trajectories", "gru"
)
CROSS_VALIDATION_WHERE_3D_CNN_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn"
)
CROSS_VALIDATION_WHERE_2D_1D_CNN_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn"
)
CROSS_VALIDATION_WHERE_CNN_GRU_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru"
)
CROSS_VALIDATION_WHERE_3D_CNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "3dcnn_single_view"
)
CROSS_VALIDATION_WHERE_2D_1D_CNN_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "2d1dcnn_single_view"
)
CROSS_VALIDATION_WHERE_CNN_GRU_SINGLE_VIEW_PATH = os.path.join(
    CROSS_VALIDATION_WHERE_PREDICTIONS_PATH, "trajectory_tensors", "cnn_gru_single_view"
)
