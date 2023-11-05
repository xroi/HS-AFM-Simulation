import numpy as np
from itertools import product
import scipy.stats
import DATA_distributions
import utils


def get_fg_weights_by_distance(counts_map: np.ndarray, fgs_pdf: dict[int, float]) -> np.ndarray:
    """
    For each fg chain: calculates the distance between the heighest z that has a count greater than 0, and the lowest
    z that has a count greater than 0, and plugs those values into the FG pdf.
    :param counts_map:
    :param fgs_pdf: holds precalculated values of the pdf of normal distribution.
    :return: A 1 dimensional numpy array where entry #i is the weight of FG #i.
    """
    dists = (np.max(np.argmax(counts_map, axis=2), axis=(0, 1)) - np.min(np.argmin(counts_map, axis=2), axis=(0, 1)))
    return np.vectorize(fgs_pdf.get)(dists)


def calculate_height_map(fgs_counts_map: np.ndarray, floaters_counts_map: np.ndarray, tip_threshold: float,
                         centers: tuple[int, int, int], pdfs: tuple[dict[int, float], dict[int, float]],
                         floater_sizes: list[float], args: dict[str, any]) -> np.ndarray:
    """
    Calculates an AFM height map according to the count maps.
    :param fgs_counts_map: ndarray of dimensionality 4, where the entries signify the count and:
    axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: fg chain.
    :param floaters_counts_map: ndarray of dimensionality 4, where the entries signify the count and:
    axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: floater type.
    :param tip_threshold: Threshold which, when reached, signifies that the tip has stopped, and returns the z for a
    specific pixel.
    :param centers: The actual coordinates of the centers of the simulation, (according to system of coordinates of
    the bounding box - not the limits given to the AFM simulation).
    :param pdfs: A tuple with two elements: each is a dict which hold precalculated values of the pdf of normal
    distribution, one used to weigh the FGs, and the other used to weigh the floaters.
    :param floater_sizes: A list of floater sizes according to the order they appear in axis 4 in floaters_counts_map.
    :param args: User arguments.
    :return: A 2d numpy array where each entry represents the Z at which the simulated AFM tip has stopped. This is
    in the bounding box system of coordinates (todo: might be changed).
    """
    height_map = np.ones(shape=fgs_counts_map.shape[:2]) * args["min_z_coord"]
    # Calculate the weights for each fg. this is done once per time step.
    fg_weights = get_fg_weights_by_distance(fgs_counts_map, pdfs[0])
    # Normalize count maps
    norm_factor = args["interval_ns"] / args["statistics_interval_ns"]
    fgs_counts_map = fgs_counts_map / norm_factor
    for x, y in product(range(fgs_counts_map.shape[0]), range(fgs_counts_map.shape[1])):
        slab_top_z = get_slab_top_z(x + args["min_x_coord"], y + args["min_y_coord"], centers, args) - args[
            "min_z_coord"]
        counts_sum = 0.0
        for z in range(fgs_counts_map.shape[2] - 1, -1, -1):
            for fg_i in np.nonzero(fgs_counts_map[x, y, z, :])[0]:
                counts_sum += fgs_counts_map[x, y, z, fg_i] * fg_weights[fg_i]
            for floater_i in np.nonzero(floaters_counts_map[x, y, z, :])[0]:
                # floater_weight = pdfs[1][z + args["min_z_coord"]] * (floater_sizes[floater_i] ** 3) * args[
                #     "floater_general_factor"]
                floater_weight = pdfs[1][z + args["min_z_coord"]] * args[
                    "floater_general_factor"]  # todo no size weighing
                counts_sum += floaters_counts_map[x, y, z, floater_i] * floater_weight
            if (counts_sum > tip_threshold) or (z < slab_top_z):
                height_map[x, y] = z + args["min_z_coord"]
                break
    return height_map


def get_slab_top_z(x: int, y: int, centers: tuple[int, int, int], args: dict[str, any]) -> int:
    """
    Gets the slab top z (i.e. the bottom limit for the tip), at a specific point.
    :param x: X of point to check.
    :param y: Y of point to check.
    :param centers: The actual coordinates of the centers of the simulation, (according to system of coordinates of
    the bounding box - not the limits given to the AFM simulation).
    :param args: User Arguments.
    :return: The slab top z (i.e. the bottom limit for the tip), at a specific point. This is in the bounding box
    system of coordinates (todo: might be changed).
    """
    if args["torus_slab"]:
        slab_top_z = utils.get_torus_top_z(x,
                                           y,
                                           centers,
                                           args["tunnel_radius_a"] / args["voxel_size_a"],
                                           (args["slab_thickness_a"] / args["voxel_size_a"]) / 2, inside=centers[2])
    else:
        slab_top_z = centers[2] if utils.is_in_circle(x, y,
                                                      args["tunnel_radius_a"] / args["voxel_size_a"],
                                                      centers[0],
                                                      centers[1]) else centers[2] + (
                (args["slab_thickness_a"] / 2) / args["voxel_size_a"])
    return slab_top_z


###########
# Unused: #
###########

def get_weight(dist_from_cent, max_z):
    return ((max_z / 2) - dist_from_cent) / (max_z / 2)


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
