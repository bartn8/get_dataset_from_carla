from termcolor import colored
import numpy as np

import data_generator.config as config


def color_error_string(string):
    return colored(string, "red", attrs=["bold"])  # , "blink"


def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])


def color_info_success(string):
    return colored(string, "green", attrs=["bold"])


def get_a_title(string, color):
    line = "#" * (len(string) + 2)
    final_string = line + "\n#" + string + "#\n" + line
    return colored(final_string, color, attrs=["bold"])


class NutException(Exception):
    """
    Exception Raised during Carla's StartUp!
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :return: (2, H, W) numpy, LiDAR as sparse image
    """
    MAX_HIST_POINTS = 5

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_IMAGE_W + 1)
        ybins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_IMAGE_H + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > MAX_HIST_POINTS] = MAX_HIST_POINTS
        overhead_splat = hist / MAX_HIST_POINTS
        # The transpose here is an efficient axis swap.
        # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
        # (x height channel, y width channel)
        return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < -2.5 + config.MAXIMUM_LIDAR_HEIGHT]
    lidar = lidar[lidar[..., 2] > -2.5 + config.MINIMUM_LIDAR_HEIGHT]
    features = splat_points(lidar)
    features = np.stack([features], axis=-1)
    features = np.transpose(features, (2, 0, 1))
    features *= 255
    features = features.astype(np.uint8)
    return features

