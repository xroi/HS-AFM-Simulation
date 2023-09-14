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


def z_test(x, y, summed_counts_map, needle_threshold, slab_top_z, is_in_tunnel):
    counts_sum = 0
    for z in range(summed_counts_map.shape[2] - 1, -1, -1):
        counts_sum += summed_counts_map[x, y, z]
        if (counts_sum > needle_threshold) or (not is_in_tunnel and z < slab_top_z):
            return z
    return 0


def z_test2(x, y, density_map, needle_threshold, slab_top_z, is_in_tunnel):
    # for z in range(density_map.shape[2] - 1, -1, -1):
    #     count = 0
    #     for i in range(density_map.shape[3]):
    #         if np.random.binomial(1, density_map[x, y, z, i]):
    #             count += 1
    #     if (count > needle_threshold) or (not is_in_tunnel and z < slab_top_z):
    #         return z
    # return 0
    temp = density_map[x, y, :, :]
    temp = 1 - temp
    mults = np.prod(temp, axis=1)
    for z in range(density_map.shape[2] - 1, -1, -1):
        count = 0
        for i in range(density_map.shape[3]):
            if np.random.binomial(1, 1 - mults[z]):
                count += 1
        if (count > needle_threshold) or (not is_in_tunnel and z < slab_top_z):
            return z
    return 0


def height_func_wrapper(func_name, x, y, counts_maps, summed_counts_map, density_map, needle_threshold, slab_top_z,
                        is_in_tunnel, args):
    match func_name:
        # case "z_top":
        #     return z_top
        # case "z_sum":
        #     return z_sum
        # case "z_fraction":
        #     return z_fraction
        case "z_test":
            return z_test(x, y, summed_counts_map, needle_threshold, slab_top_z, is_in_tunnel)
        case "z_test2":
            return z_test2(x, y, density_map, needle_threshold, slab_top_z, is_in_tunnel)
