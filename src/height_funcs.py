import numpy as np
from itertools import product

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


def z_test(counts_map, needle_threshold, centers, args):
    """x, y, slab_top_z, z_center, z_min, should be in px"""
    height_map = np.zeros(shape=counts_map.shape[:2])
    summed_counts_map = np.sum(counts_map, axis=3)
    for x, y in product(range(counts_map.shape[0]), range(counts_map.shape[1])):
        slab_top_z = get_slab_top_z(x, y, centers, args)
        counts_sum = 0
        for z in range(summed_counts_map.shape[2] - 1, -1, -1):
            counts_sum += (summed_counts_map[x, y, z] * get_weight(np.abs(centers[2] - (z + args["min_z_coord"])),
                                                                   summed_counts_map.shape[2]))
            if (counts_sum > needle_threshold) or z < slab_top_z:
                height_map[x, y] = z
                break
    return 0


def get_fg_orientation_weight(single_fg_counts_map):
    # Previous PCA approach
    # coords = np.array(np.where(single_fg_counts_map != 0.0)).T
    # coords_mean = coords.mean(axis=0)
    # uu, dd, vv = np.linalg.svd(coords - coords_mean)
    # fit_vec = vv[0] / np.linalg.norm(vv[0])
    # return 1 / np.abs(fit_vec[2])
    dist = (np.max(np.argmax(single_fg_counts_map, axis=2)) - np.min(np.argmin(single_fg_counts_map, axis=2)))
    if dist == 0:
        return 1
    return 1 / dist


def z_test2(counts_map, needle_threshold, centers, args):
    height_map = np.zeros(shape=counts_map.shape[:2])
    fg_weights = []
    for fg_i in range(counts_map.shape[3]):
        fg_weights.append(get_fg_orientation_weight(counts_map[:, :, :, fg_i]))
    for x, y in product(range(counts_map.shape[0]), range(counts_map.shape[1])):
        slab_top_z = get_slab_top_z(x, y, centers, args)
        counts_sum = 0
        for z in range(counts_map.shape[2] - 1, -1, -1):
            for fg_i in np.unique(np.array(np.where(counts_map[x, y, z, :] != 0.0))[0, :]):
                # todo how to get mean for pixels around rim? (fake empty space)
                counts_sum += utils.get_circle_mean(counts_map[:, :, z, fg_i], x, y, args["needle_radius_px"]) * \
                              fg_weights[fg_i]
                # counts_sum += counts_map[x, y, z, fg_i] * fg_weights[fg_i]
            if counts_sum > 0:
                pass
            if (counts_sum > needle_threshold) or z < slab_top_z:
                height_map[x, y] = z
                break
    return height_map


def get_slab_top_z(x, y, centers, args):
    if args["torus_slab"]:
        slab_top_z = utils.get_torus_top_z(x,
                                           y,
                                           centers,
                                           args["tunnel_radius_a"] / args["voxel_size_a"],
                                           (args["slab_thickness_a"] / args["voxel_size_a"]) / 2)
    else:
        slab_top_z = -1 if utils.is_in_circle(x, y, args["tunnel_radius_a"] / args["voxel_size_a"], centers[0],
                                              centers[1]) else centers[2] * (args["slab_thickness_a"] / 2) / args[
            "voxel_size_a"]
    return slab_top_z


def get_weight(dist_from_cent, max_z):
    return ((max_z / 2) - dist_from_cent) / (max_z / 2)
