import numpy as np


def ball_average(x, y, z, arr, r):
    # function written by AI
    x_min = max(0, x - r)
    x_max = min(arr.shape[0], x + r + 1)
    y_min = max(0, y - r)
    y_max = min(arr.shape[1], y + r + 1)
    z_min = max(0, z - r)
    z_max = min(arr.shape[2], z + r + 1)
    sub_arr = arr[x_min:x_max, y_min:y_max, z_min:z_max]
    xx, yy, zz = np.mgrid[:sub_arr.shape[0], :sub_arr.shape[1], :sub_arr.shape[2]]
    mask = ((xx - (x - x_min)) ** 2 + (yy - (y - y_min)) ** 2 + (zz - (z - z_min)) ** 2) <= r ** 2
    return np.mean(sub_arr[mask])


def z_top(x, y, combined_density_map, args):
    for z in range(combined_density_map.shape[2] - args["needle_radius_px"], -1, -1):
        if ball_average(x, y, z, combined_density_map, args["needle_radius_px"]) > args["needle_threshold"]:
            return z  # / combined_density_map.shape[2]
    return 0


def z_sum(x, y, combined_density_map, args):
    value = np.sum(combined_density_map[x, y, :])
    if value > args["needle_threshold"]:
        return value
    return 0


def z_fraction(x, y, combined_density_map, args):
    fraction_threshold = np.sum(combined_density_map[x, y, :]) * args["needle_fraction"]
    cur = 0
    for z in range(combined_density_map.shape[2]):
        cur += combined_density_map[x, y, z]
        if cur > args["needle_threshold"] and cur > fraction_threshold:
            return z
    return 0


def z_test(x, y, combined_density_map, args):
    z = combined_density_map.shape[2] - 1
    cur = 0
    for z in range(combined_density_map.shape[2] - 1, -1, -1):
        cur += combined_density_map[x, y, z]
        if cur > args["needle_threshold"]:
            return z
    return 0


def get_height_func(name):
    match name:
        case "z_top":
            return z_top
        case "z_sum":
            return z_sum
        case "z_fraction":
            return z_fraction
        case "z_test":
            return z_test
