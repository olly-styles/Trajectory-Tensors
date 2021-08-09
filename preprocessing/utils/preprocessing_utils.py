# External
import numpy as np


def normalize_bounding_box(box, frame_width, frame_height):
    """
    Converts bounding box in format [x1, y1, x2, y2] with range
    0 to frame_height, 0 to frame_width to 0 to 1
    """
    normalized_box = box.copy().astype(float)
    normalized_box[0] = box[0] / float(frame_width)
    normalized_box[1] = box[1] / float(frame_height)
    normalized_box[2] = box[2] / float(frame_width)
    normalized_box[3] = box[3] / float(frame_height)
    return normalized_box


def bounding_box_to_heatmap(box, heatmap_size):
    """
    Converts a normalized bounding box to a heatmap
    """
    heatmap = np.zeros(heatmap_size, dtype=np.uint8)
    x1_heatmap = np.floor(box[0] * heatmap_size[1]).astype(int)
    y1_heatmap = np.floor(box[1] * heatmap_size[0]).astype(int)
    x2_heatmap = np.ceil(box[2] * heatmap_size[1]).astype(int)
    y2_heatmap = np.ceil(box[3] * heatmap_size[0]).astype(int)
    heatmap[y1_heatmap:y2_heatmap, x1_heatmap:x2_heatmap] = 1
    return heatmap
