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


def z_test(x, y, density_map, needle_threshold, slab_top_z, orig_center_x, orig_center_y, args):
    z = density_map.shape[2] - 1
    cur = 0
    for z in range(density_map.shape[2] - 1, -1, -1):
        cur += density_map[x, y, z]
        if utils.is_in_circle(x + args["min_x_coord"],
                              y + args["min_y_coord"],
                              args["tunnel_radius_a"] / args["voxel_size_a"],
                              orig_center_x,
                              orig_center_y):
            if cur > needle_threshold:
                return z
        else:
            if cur > needle_threshold:
                return min(max(z, slab_top_z), args["max_z_coord"])
            else:
                return min(slab_top_z, args["max_z_coord"])
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
