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


def get_torus_top_z(x, y, center_x, center_y, center_z, r, p):
    if is_in_circle(x, y, r - p, center_x, center_y):
        return 0
    if not is_in_circle(x, y, r, center_x, center_y):
        return center_z + p
    point_r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return int(np.sqrt(p ** 2 - (r - point_r) ** 2) + center_z)


def torus_inner_radius(r, p, z):
    """
    :param r: radius from center of tube to center of torus ring
    :param p: radius of tube
    :param z: height where z=0 is center of torus ring
    """
    if z < -p or z > p:
        return r
    return r - np.sqrt(p ** 2 - z ** 2)


def get_coordinate_list(cylinder_layers, fg_per_layer, r, p, bounding_box_size_z):
    """for anchoring fgs on torus."""
    coordinates = []
    for i in np.arange(0.0, r + 0.01, r / (cylinder_layers - 1)):
        inner_r = torus_inner_radius(r, p, p - i)
        for j in range(int(fg_per_layer)):
            theta = 2.0 * np.pi * j / fg_per_layer
            x = inner_r * np.cos(theta)
            y = inner_r * np.sin(theta)
            z = i + (bounding_box_size_z / 2) - p
            coordinates.append([x, y, z])
    return coordinates
