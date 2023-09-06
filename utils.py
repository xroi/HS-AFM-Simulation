import numpy as np


def get_ball_mask(arr, x, y, z, r):
    """returns a maks for a ball of radius r, around x,y,z, in 3d array arr."""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    z_min = max(0, z - r)
    z_max = min(arr.shape[2], z + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max, z_min:z_max]
    xx, yy, zz = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1], :sub_arr.shape[2]]
    mask = ((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2 + (zz - (z - z_min)) ** 2) <= r ** 2
    return mask


def get_ball_mean(arr, x, y, z, r):
    return np.mean(get_ball_mask(arr, x, y, z, r))


def get_ball_median(arr, x, y, z, r):
    return np.median(get_ball_mask(arr, x, y, z, r))


def median_threshold(height_maps, r, frac):
    """Calculates a threshold for densities using a fraction of the median in a ball of radius r around the center."""
    x = height_maps.shape[0] / 2
    y = height_maps.shape[1] / 2
    z = height_maps.shape[2] / 2
    return get_ball_median(height_maps, x, y, z, r) * frac
