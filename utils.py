import numpy as np


def get_ball_mask(arr, x, y, z, r):
    """returns a mask for a ball of radius r, around x,y,z, in 3d array arr."""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    z_min = max(0, z - r)
    z_max = min(arr.shape[2], z + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max, z_min:z_max]
    xx, yy, zz = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1], :sub_arr.shape[2]]
    return sub_arr[((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2 + (zz - (z - z_min)) ** 2) <= r ** 2]


def get_circle_mask(arr, x, y, r):
    """return as mask for a circle of radius r around x,y in 2d array ayy"""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max]
    xx, yy = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1]]
    return sub_arr[((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2) <= r ** 2]


def get_ball_mean(arr, x, y, z, r):
    return np.mean(get_ball_mask(arr, x, y, z, r))


def get_ball_median(arr, x, y, z, r):
    return np.median(get_ball_mask(arr, x, y, z, r))


def get_circle_median(arr, x, y, r):
    return np.median(get_circle_mask(arr, x, y, r))


def is_in_circle(x, y, r, center_x, center_y):
    return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) < r


def median_threshold(density_maps, r, frac):
    """Calculates a threshold for densities using a fraction of the median in a ball of radius r around the center."""
    x = int(density_maps[0].shape[0] / 2)
    y = int(density_maps[0].shape[1] / 2)
    z = int(density_maps[0].shape[2] / 2)
    vals = []
    for density_map in density_maps:
        arr = get_ball_mask(density_map, x, y, z, r)
        for val in arr:
            vals.append(val)
    return np.median(vals) * frac  # todo this is usually 0.
