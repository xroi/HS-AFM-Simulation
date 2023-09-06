import numpy as np
import utils


def z_top(x, y, combined_density_map, args):
    for z in range(combined_density_map.shape[2] - args["needle_radius_px"], -1, -1):
        if utils.get_ball_mean(combined_density_map, x, y, z, args["needle_radius_px"]) > args["needle_threshold"]:
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
