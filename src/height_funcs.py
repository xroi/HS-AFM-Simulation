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


def get_fg_weights_by_vector(counts_map):
    fg_weights = []
    # PCA
    for fg_i in range(counts_map.shape[3]):
        coords = np.array(np.where(counts_map[:, :, :, fg_i] != 0.0)).T
        coords_mean = coords.mean(axis=0)
        uu, dd, vv = np.linalg.svd(coords - coords_mean)
        fit_vec = vv[0] / np.linalg.norm(vv[0])
        fg_weights.append(1 / np.abs(fit_vec[2]))
    return fg_weights[0]


def get_fg_weights_by_distance(counts_map):
    dist = (np.max(np.argmax(counts_map, axis=2), axis=(0, 1)) - np.min(np.argmin(counts_map, axis=2), axis=(0, 1)))
    dist[dist == 0] = 1
    return 1 / dist


def z_test2(fgs_counts_map, floaters_counts_map, needle_threshold, centers, args):
    height_map = np.ones(shape=fgs_counts_map.shape[:2]) * args["min_z_coord"]
    fg_weights = get_fg_weights_by_distance(fgs_counts_map)
    for x, y in product(range(fgs_counts_map.shape[0]), range(fgs_counts_map.shape[1])):
        slab_top_z = get_slab_top_z(x + args["min_x_coord"], y + args["min_y_coord"], centers, args) - args[
            "min_z_coord"]
        counts_sum = 0
        for z in range(fgs_counts_map.shape[2] - 1, -1, -1):
            for fg_i in np.unique(np.array(np.where(fgs_counts_map[x, y, z, :] != 0.0))):
                # todo how to get mean for pixels around rim? (fake empty space)
                counts_sum += utils.get_circle_mean(fgs_counts_map[:, :, z, fg_i], x, y, args["needle_radius_px"]) * \
                              fg_weights[fg_i]
                # counts_sum += counts_map[x, y, z, fg_i] * fg_weights[fg_i]
            for floater_i in np.unique(np.array(np.where(floaters_counts_map[x, y, z, :] != 0.0))):
                counts_sum += utils.get_circle_mean(floaters_counts_map[:, :, z, floater_i], x, y,
                                                    args["needle_radius_px"]) * 0.1  # todo change
            if counts_sum > 0:
                pass
            if (counts_sum > needle_threshold) or z < slab_top_z:
                height_map[x, y] = z + args["min_z_coord"]
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
                                              centers[1]) else centers[2] + (
                (args["slab_thickness_a"] / 2) / args["voxel_size_a"])
    return slab_top_z


def get_weight(dist_from_cent, max_z):
    return ((max_z / 2) - dist_from_cent) / (max_z / 2)
