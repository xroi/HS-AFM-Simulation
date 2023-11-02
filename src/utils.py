import numpy as np

AVOGADRO = 6.0221408e+23


def get_ball_vals(arr: np.ndarray, x: int, y: int, z: int, r: int):
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


def get_ball_mask(arr: np.ndarray, x: int, y: int, z: int, r: int):
    xx, yy, zz = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    dist_from_center = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
    return dist_from_center < r


def get_top_of_ball_mask(arr: np.ndarray, x: int, y: int, z: int, r: int):
    ball = get_ball_mask(arr, x, y, z, r)
    smaller_ball = get_ball_mask(arr, x, y, z, r - 1)
    mask = np.logical_xor(ball, smaller_ball)
    mask[:, :, :z] = 0
    return mask


def get_circle_vals(arr: np.ndarray, x: int, y: int, r: int):
    """return a mask for a circle of radius r around x,y in 2d array ayy"""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max]
    xx, yy = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1]]
    return sub_arr[((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2) <= r ** 2]


def get_anti_circle_vals(arr: np.ndarray, x: int, y: int, r: int):
    """return a mask for a circle of radius r around x,y in 2d array ayy"""
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max]
    xx, yy = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1]]
    return sub_arr[((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2) > r ** 2]


def get_ring_vals(arr: np.ndarray, x: int, y: int, r: int):
    """return a mask for the outline of a circle of radius r around x,y in 2d array ayy"""
    y_indices, x_indices = np.indices(arr.shape)
    distance = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
    mask = (distance >= r) & (distance < r + 1)
    return arr[mask]


def get_ring_means_array(arr: np.ndarray, x: int, y: int):
    """calculates the means by radial distance around x,y in 2d array arr."""
    max_r = get_max_r(arr.shape, x, y)
    vals = []
    for r in range(max_r):
        vals.append(get_ring_mean(arr, x, y, r))
    return np.array(vals)


def get_max_r(shape: tuple[int, int], x: int, y: int):
    return int(min([x, shape[0] - x - 1, y, shape[1] - y - 1]))


def get_ring_mean(arr: np.ndarray, x: int, y: int, r: int):
    return np.mean(get_ring_vals(arr, x, y, r))


def get_ball_mean(arr: np.ndarray, x: int, y: int, z: int, r: int):
    return np.mean(get_ball_vals(arr, x, y, z, r))


def get_ball_median(arr: np.ndarray, x: int, y: int, z: int, r: int):
    return np.median(get_ball_vals(arr, x, y, z, r))


def get_circle_median(arr: np.ndarray, x: int, y: int, r: int):
    return np.median(get_circle_vals(arr, x, y, r))


def get_circle_mean(arr: np.ndarray, x: int, y: int, r: int):
    return np.mean(get_circle_vals(arr, x, y, r))


def get_circle_max(arr: np.ndarray, x: int, y: int, r: int):
    return np.max(get_circle_vals(arr, x, y, r))


def is_in_circle(x: int, y: int, r: int, center_x: int, center_y: int):
    return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) < r


def get_torus_top_z(x: int, y: int, centers: tuple[int, int, int], r: int, p: int):
    if is_in_circle(x, y, r - p, centers[0], centers[1]):
        return 0
    if not is_in_circle(x, y, r, centers[0], centers[1]):
        return centers[2] + p
    point_r = np.sqrt((x - centers[0]) ** 2 + (y - centers[1]) ** 2)
    return int(np.sqrt(p ** 2 - (r - point_r) ** 2) + centers[2])


def torus_inner_radius(r: float, p: float, z: int):
    """
    :param r: radius from center of tube to center of torus ring
    :param p: radius of tube
    :param z: height where z=0 is center of torus ring
    """
    if z < -p or z > p:
        return r
    return r - np.sqrt(p ** 2 - z ** 2)


def get_coordinate_list(cylinder_layers: int, fg_per_layer: int, r: float, p: float):
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


def concentration_to_amount(molar: float, box_side_a: float):
    volume = np.power(box_side_a, 3)
    return (molar * AVOGADRO * volume) / 1e+27


def amount_to_concentration(amount: int, box_side_a: float):
    volume = np.power(box_side_a, 3)
    return amount / (AVOGADRO * volume * 1e-27)


def calculate_z_distribution(maps: np.ndarray, inner_r: int):
    """axes: x,y,z,fg_i,t"""
    centers = (int(maps.shape[0] / 2), int(maps.shape[1] / 2))
    maps = np.sum(maps, axis=(3, 4))
    inner_distribution = []
    outer_distribution = []
    for z in range(maps.shape[2]):
        inner_val = 0
        outer_val = 0
        inner_val += np.sum(get_circle_vals(maps[:, :, z], centers[0], centers[1], inner_r))
        outer_val += np.sum(get_anti_circle_vals(maps[:, :, z], centers[0], centers[1], inner_r))
        inner_distribution.append(inner_val)
        outer_distribution.append(outer_val)
    inner_distribution = np.array(inner_distribution)
    outer_distribution = np.array(outer_distribution)
    inner_distribution = inner_distribution / np.linalg.norm(inner_distribution)
    outer_distribution = outer_distribution / np.linalg.norm(outer_distribution)
    return inner_distribution, outer_distribution
