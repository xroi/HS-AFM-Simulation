import numpy as np
import utils


# def z_top(x, y, density_map, needle_threshold, orig_shape, args):
#     for z in range(density_map.shape[2] - args["needle_radius_px"], -1, -1):
#         if utils.get_ball_mean(density_map, x, y, z, args["needle_radius_px"]) > args["needle_threshold"]:
#             return z  # / combined_density_map.shape[2]
#     return 0
#
#
# def z_sum(x, y, density_map, needle_threshold, orig_shape, args):
#     value = np.sum(density_map[x, y, :])
#     if value > needle_threshold:
#         return value
#     return 0
#
#
# def z_fraction(x, y, density_map, needle_threshold, orig_shape, args):
#     fraction_threshold = np.sum(density_map[x, y, :]) * args["needle_fraction"]
#     cur = 0
#     for z in range(density_map.shape[2]):
#         cur += density_map[x, y, z]
#         if cur > needle_threshold and cur > fraction_threshold:
#             return z
#     return 0


def z_test(x, y, density_map, needle_threshold, slab_top_z, is_in_tunnel):
    density_sum = 0
    for z in range(density_map.shape[2] - 1, -1, -1):
        density_sum += density_map[x, y, z]
        # density_sum += utils.get_circle_median(density_map[:, :, z], x, y, 1)
        if not is_in_tunnel and z < slab_top_z:
            density_sum += np.inf
        if density_sum > needle_threshold:
            return z
    return 0


def get_height_func(name):
    match name:
        # case "z_top":
        #     return z_top
        # case "z_sum":
        #     return z_sum
        # case "z_fraction":
        #     return z_fraction
        case "z_test":
            return z_test
