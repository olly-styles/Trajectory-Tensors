import imageio
import numpy as np
import cv2
import cmapy
from scipy.ndimage import gaussian_filter
from preprocessing.src.train_set_split import get_inputs_and_which_targets
from global_config.global_config import OFFSET_LEN


def read_video(path, start_frame=0, num_frames=None):
    """
    Given a video at the specified path, reads num_frames starting from
    start_frame using the imageio libary.
    Parameters
    ----------
    path: str
        File path to video file
    start_frame: int
        Video frame to start reading from
    num_frames: int
        num of video frames to read. If None, read all frames in the video.
    Returns
    -------
    Numpy array
        type: uint8
        shape: (num_frames, image_width, image_height, 3)
        The RGB video with range 0 - 255
    """
    video_reader = imageio.get_reader(path, "ffmpeg")
    metadata = video_reader.get_meta_data()
    resolution = metadata["source_size"]
    if num_frames is None:
        num_frames = int(metadata["fps"] * metadata["duration"]) + 1
    end_frame = num_frames + start_frame
    video_shape = (num_frames, resolution[1], resolution[0], 3)
    video = np.zeros(video_shape, dtype="uint8")
    for frame_num, im in enumerate(video_reader):
        if frame_num < start_frame:
            continue
        if frame_num == end_frame:
            break
        video[int(frame_num - start_frame)] = im
    video = video[: int(frame_num - start_frame) + 1]
    return video


def write_video(path, video, fps):
    """
    Writes a video represented by a numpy array to the file at path using
    the imageio libary.
    Parameters
    ----------
    path: str
        File path to write video file
    video: np array
        type: uint8
        shape: (num_frames, image_width, image_height, 3)
        An RGB video with range 0 - 255
    fps: int
        Frames per second to encode video
    """
    writer = imageio.get_writer(path, fps=fps, macro_block_size=8)
    for im in video:
        writer.append_data(im)
    writer.close()


def draw_grid(image, grid_size, grid_thickness):
    grid_interval_y = int(image.shape[0] / grid_size[0])
    grid_interval_x = int(image.shape[1] / grid_size[1])
    for vertical_line in range(0, image.shape[0], grid_interval_x):
        image = cv2.line(image, (0, vertical_line), (image.shape[1], vertical_line), (0, 0, 0), grid_thickness, 1)
    for horizontal_line in range(0, image.shape[1], grid_interval_y):
        image = cv2.line(image, (horizontal_line, 0), (horizontal_line, image.shape[0]), (0, 0, 0), grid_thickness, 1)
    return image


def overlay_heatmap(image, heatmap, heatmap_weight):
    heatmap = cv2.resize(heatmap, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap = 255 - (heatmap * 255)
    heatmap = heatmap.astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cmapy.cmap("coolwarm"))
    overlayed = cv2.addWeighted(heatmap, heatmap_weight, image, 1 - heatmap_weight, 0)
    return overlayed


def resize_video(video, new_size):
    """
    Resizes a video represented by a numpy array.
    Parameters
    ----------
    video: np array
        type: uint8
        shape: (num_frames, image_width, image_height, 3)
        An RGB video with range 0 - 255
    new_size: touple
        New video resolution
    """
    num_frames = len(video)
    new_video = np.zeros((num_frames, new_size[1], new_size[0], 3), dtype="uint8")
    for frame_num in range(0, len(video)):
        new_video[frame_num] = cv2.resize(video[frame_num], dsize=new_size, interpolation=cv2.INTER_CUBIC)
    return new_video


def write_image(path, image):
    """
    Writes an image represented by a numpy array to the file at path using
    the imageio libary.
    Parameters
    ----------
    path: str
        File path to write video file
    image: np array
        type: uint8
        shape: (image_width, image_height, 3)
        An RGB image with range 0 - 255
    """
    imageio.imwrite(path, image)


def get_fold_and_index(day, match_num):
    _, _, all_day_numbers, _ = get_inputs_and_which_targets()
    get_fold = lambda x: (x % 5) + 1
    all_folds = get_fold(all_day_numbers)
    folds = all_folds[np.argwhere(all_day_numbers == float(day))]
    assert np.all(folds[0] == folds)
    fold = int(folds[0])

    test_indexs = np.argwhere(all_day_numbers % 5 == fold - 1).flatten()
    # Must be rounded to multiple of OFFSET_LEN to ensure same sample with different offset
    # does not appear in both val and test
    val_indexs = test_indexs[0 : int(len(test_indexs) / 20) * OFFSET_LEN]
    test_indexs = test_indexs[int(len(test_indexs) / 20) * OFFSET_LEN :]

    if float(day) not in all_day_numbers[test_indexs]:
        print("day not in test data")
        exit(1)

    index = (int(match_num) * OFFSET_LEN) + (np.where(all_day_numbers[test_indexs] == float(day))[0][0])

    return fold, index
