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
    """return a mask for a circle of radius r around x,y in 2d array ayy"""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max]
    xx, yy = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1]]
    return sub_arr[((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2) <= r ** 2]


def get_ring_mask(arr, x, y, r):
    """return a mask for the outline of a circle of radius r around x,y in 2d array ayy"""
    y_indices, x_indices = np.indices(arr.shape)
    distance = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
    mask = (distance >= r) & (distance < r + 1)
    return arr[mask]


def get_ring_means_array(arr, x, y, max_r):
    arr = np.mean(np.dstack(arr), axis=2)
    vals = []
    for r in range(max_r):
        vals.append(get_ring_mean(arr, x, y, r))
    return np.array(vals)


def get_max_r(shape, x, y):
    return int(min([x, shape[0] - x - 1, y, shape[1] - y - 1]))


def get_ring_mean(arr, x, y, r):
    return np.mean(get_ring_mask(arr, x, y, r))


def get_ball_mean(arr, x, y, z, r):
    return np.mean(get_ball_mask(arr, x, y, z, r))


def get_ball_median(arr, x, y, z, r):
    return np.median(get_ball_mask(arr, x, y, z, r))


def get_circle_median(arr, x, y, r):
    return np.median(get_circle_mask(arr, x, y, r))


def get_circle_mean(arr, x, y, r):
    return np.mean(get_circle_mask(arr, x, y, r))


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


def get_torus_top_z(x, y, centers, r, p):
    if is_in_circle(x, y, r - p, centers[0], centers[1]):
        return 0
    if not is_in_circle(x, y, r, centers[0], centers[1]):
        return centers[2] + p
    point_r = np.sqrt((x - centers[0]) ** 2 + (y - centers[1]) ** 2)
    return int(np.sqrt(p ** 2 - (r - point_r) ** 2) + centers[2])


def torus_inner_radius(r, p, z):
    """
    :param r: radius from center of tube to center of torus ring
    :param p: radius of tube
    :param z: height where z=0 is center of torus ring
    """
    if z < -p or z > p:
        return r
    return r - np.sqrt(p ** 2 - z ** 2)


def get_coordinate_list(cylinder_layers, fg_per_layer, r, p):
    """for anchoring fgs on torus."""
    coordinates = []
    for i in range(cylinder_layers):
        theta1 = np.pi * i / (cylinder_layers - 1) - (np.pi / 2)
        z = p * np.sin(theta1)
        inner_r = torus_inner_radius(r, p, z)
        for j in range(int(fg_per_layer)):
            theta2 = 2.0 * np.pi * j / fg_per_layer
            x = inner_r * np.cos(theta2)
            y = inner_r * np.sin(theta2)
            coordinates.append([x, y, z])
    return coordinates