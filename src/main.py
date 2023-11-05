import h5py
import numpy as np
import scipy.stats
import height_funcs
import args as arguments
import utils
import output
import raster
import auto_corr
from multiprocessing import Pool
from functools import partial
import tqdm


def main() -> None:
    args: dict[str, any] = arguments.parse_arguments()
    print(args)

    real_time_maps = get_real_time_maps(args)
    rasterized_maps = raster.get_rasterized_maps(real_time_maps, args)
    if args["output_pickle"]:
        output.save_pickle(real_time_maps, rasterized_maps, args, f"{args['output_path_prefix']}.pickle")
    original_shape = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}{args['input_suffix']}")
    center_z = int(original_shape[0] / 2)
    if args["output_real_time_gif"]:
        output.output_gif(args, np.array(real_time_maps),
                          f"{args['output_path_prefix']}_real_time.gif", center_z,
                          args["min_z_coord"] + args["tip_bottom_z_dist"],
                          args["max_z_coord"])
    if args["output_raster_gif"]:
        if len(rasterized_maps) > 0:
            output.output_gif(args, rasterized_maps,
                              f"{args['output_path_prefix']}_rasterized.gif", center_z,
                              args["min_z_coord"] + args["tip_bottom_z_dist"],
                              args["max_z_coord"])
    if args["output_post"]:
        post_analysis(args, real_time_maps, rasterized_maps, original_shape=get_hdf5_size(
            f"{args['input_path']}/{args['simulation_start_time_ns']}{args['input_suffix']}")
                      )


def post_analysis(args: dict[str, any], real_time_maps: list[np.ndarray], needle_maps: list[np.ndarray],
                  original_shape: tuple[int, int, int]) -> None:
    real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 1)
    taus = auto_corr.calculate_taus(real_time_acorrs)
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    center_z = int(original_shape[2] / 2)
    output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
                          args["max_y_coord"], center_x, center_y, 10,
                          f"{args['output_path_prefix']}_taus_real_time.png")
    tau_ring_means = utils.get_ring_means_array(taus,
                                                int(real_time_maps[0].shape[0] / 2),
                                                int(real_time_maps[0].shape[1] / 2))
    output.visualize_tau_by_radial_distance(tau_ring_means, f"{args['output_path_prefix']}_tau_radial_real_time.png")

    # real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 3)
    # taus = auto_corr.calculate_taus(real_time_acorrs)
    # output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
    #                       args["max_y_coord"], center_x, center_y, 10)

    ring_means = utils.get_ring_means_array(np.mean(np.dstack(real_time_maps), axis=2),
                                            int(real_time_maps[0].shape[0] / 2),
                                            int(real_time_maps[0].shape[1] / 2))
    ring_means = (ring_means - center_z)
    output.visualize_height_by_radial_distance(ring_means,
                                               f"{args['output_path_prefix']}_height_radial_real_time.png",
                                               sym=True,
                                               yrange=[0, 14])
    output.visualize_tcf_samples(real_time_acorrs, taus, 5, 5, f"{args['output_path_prefix']}_tcf_samples.png")


def calculate_normal_pdf(min_z: int, max_z: int, mu: float, sigma: float) -> dict[int, float]:
    """
    Calculate the pdf of normal distribution with parameters (mu,sigma) within the range of (min_z, max_z).
    :param min_z: Minimum z to calculate the pdf for.
    :param max_z: Maximum z to calculate the pdf for.
    :param mu: Mean of the normal distribution.
    :param sigma: Standard deviation of normal distribution.
    :return: A dictionary with keys between min_z and max_z and values being the respective pdf at that z.
    """
    vals = {}
    norm = scipy.stats.norm(mu, sigma)
    for i in range(min_z, max_z):
        vals[i] = norm.pdf(i)
    return vals


def get_real_time_maps(args: dict[str, any]) -> list[np.ndarray]:
    """
    Sequentially loads each hdf5 file, and calculates the height map for it.
    :param args: User arguments.
    :return: List of all height maps, for each point of time.
    """
    tip_threshold = args["tip_custom_threshold"]  # todo not using get_needle_threshold
    original_shape = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}{args['input_suffix']}")
    centers = (int(original_shape[0] / 2), int(original_shape[1] / 2), int(original_shape[2] / 2))
    fg_pdfs = calculate_normal_pdf(0, args["max_z_coord"] - args["min_z_coord"] + 1, 0,
                                   args["fgs_sigma_a"] / args["voxel_size_a"])
    floater_pdfs = calculate_normal_pdf(0, centers[2] * 2, centers[2],
                                        args["floaters_sigma_a"] / args["voxel_size_a"])
    pdfs = (fg_pdfs, floater_pdfs)
    stages_total = int(args["simulation_end_time_ns"] / args["interval_ns"] - args["simulation_start_time_ns"] /
                       args["interval_ns"])

    get_single = partial(get_single_real_time_map, args=args, centers=centers, pdfs=pdfs,
                         tip_threshold=tip_threshold)
    real_time_maps = get_real_time_maps_helper_parallel(args, get_single, stages_total)
    return real_time_maps


def get_single_real_time_map(time: int, args: dict[str, any], centers: tuple[int, int, int],
                             pdfs: tuple[dict[int, float], dict[int, float]], tip_threshold: float) -> np.array:
    """loads and calculates the height map for a single file/point of time. """
    # Load the file
    fgs_counts_map, floaters_counts_map, floater_sizes = load_individual_counts_maps(time, args)
    # Enlarge the floaters data to better represent their actual size.
    if args["enlarge_floaters"]:
        floaters_counts_map = enlarge_floater_size(floaters_counts_map, floater_sizes)
    # enlarge the maps sideways to simulate needle size:
    fgs_counts_map = enlarge_sideways(fgs_counts_map, args["tip_radius_px"])
    if len(floater_sizes) != 0:
        floaters_counts_map = enlarge_sideways(floaters_counts_map, args["tip_radius_px"])
    # Perform the height calculation
    height_map = height_funcs.calculate_height_map(fgs_counts_map, floaters_counts_map, tip_threshold,
                                                   centers, pdfs, floater_sizes, args)
    if not args["progress_bar"]:
        print(time, flush=True)

    return height_map


def get_real_time_maps_helper_parallel(args, get_single, stages_total):
    """Parallel operation of get_real_time_maps"""
    p = Pool(args["n_cores"]) if args["n_cores"] else Pool()
    if args["progress_bar"]:
        real_time_maps = list(tqdm.tqdm(p.imap(
            get_single,
            range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"])),
            total=stages_total, colour='WHITE'))
    else:
        real_time_maps = list(p.imap(
            get_single,
            range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"])))
    p.close()
    p.join()
    return real_time_maps


def enlarge_floater_size(floater_individual_counts_maps: np.ndarray, floater_sizes: list[float]) -> np.ndarray:
    """
    Enlarges every non-zero coordinate in the counts map to span across the top of a ball with the actual floater
    radius, instead of a single point.
    :param floater_individual_counts_maps: ndarray of dimensionality 4, where the entries signify the count and:
        axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: floater type.
    :param floater_sizes: list of floater sizes in angstrom, corresponding with axis 3 in
        floater_individual_counts_maps.
    :return: ndarray of dimensionality 4, where the entries signify the enlarged count and:
        axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: floater type.
    """
    shape = floater_individual_counts_maps.shape
    new_maps = np.zeros(shape=shape)
    mid_x = int(shape[0] / 2)
    mid_y = int(shape[1] / 2)
    mid_z = int(shape[2] / 2)
    # Calculating only len(floater_sized) masks and simply shifting them instead of calculating a mask for every point,
    mid_masks = [utils.get_top_of_ball_mask(floater_individual_counts_maps[:, :, :, 0], mid_x, mid_y, mid_z,
                                            floater_sizes[i]) for i in range(len(floater_sizes))]
    # For each non zero coordinate, move the respective mask into position, and set the values in the area of the
    # mask to be the same as the coordinate. todo : set the coordinate to 0
    for (x, y, z, i) in zip(*np.nonzero(floater_individual_counts_maps)):
        shift_x = int(x - mid_x)
        shift_y = int(y - mid_y)
        shift_z = int(z - mid_z)
        mask = np.roll(mid_masks[i], (shift_x, shift_y, shift_z), axis=(0, 1, 2))
        if shift_x >= 0:
            mask[:shift_x, :, :] = 0
        else:
            mask[shift_x:, :, :] = 0
        if shift_y >= 0:
            mask[:, :shift_y, :] = 0
        else:
            mask[:, shift_y:, :] = 0
        if shift_z >= 0:
            mask[:, :, :shift_z] = 0
        else:
            mask[:, :, shift_z:] = 0
        new_maps[:, :, :, i][mask] += floater_individual_counts_maps[x, y, z, i]
    return new_maps


def enlarge_sideways(maps, r):
    shape = maps.shape
    new_maps = np.zeros(shape=shape)
    mid_x = int(shape[0] / 2)
    mid_y = int(shape[1] / 2)
    mid_z = int(shape[2] / 2)
    mask = utils.get_circle_mask_3d(maps[:, :, :, 0], mid_x, mid_y, mid_z, r)
    for x, y, z, i in zip(*np.nonzero(maps)):
        shift_x = int(x - mid_x)
        shift_y = int(y - mid_y)
        shift_z = int(z - mid_z)
        new_mask = np.roll(mask, (shift_x, shift_y, shift_z), axis=(0, 1, 2))
        if shift_x >= 0:
            new_mask[:shift_x, :, :] = 0
        else:
            new_mask[shift_x:, :, :] = 0
        if shift_y >= 0:
            new_mask[:, :shift_y, :] = 0
        else:
            new_mask[:, shift_y:, :] = 0
        if shift_z >= 0:
            new_mask[:, :, :shift_z] = 0
        else:
            new_mask[:, :, shift_z:] = 0
        new_maps[:, :, :, i][new_mask] += maps[x, y, z, i]
    return new_maps


def load_individual_counts_maps(time: int, args: dict[str, any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Loads hdf5 file and returns the seperated count maps of FGs, and floaters for that file.
    :param time: the time from which to load, i.e. file would be named "<time>.hdf5"
    :param args: User arguments.
    :return: A tuple with these entries:
    [0]: ndarray of dimensionality 4, where the entries signify the count and:
        axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: fg chain.
    [1]: ndarray of dimensionality 4, where the entries signify the count and:
        axis 0: x-axis. axis 1: y-axis. axis 2: z-axis. axis 3: floater type.
    [2]: list of floater sizes in angstrom, corresponding with axis 3 in [1].
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]

    with h5py.File(f"{args['input_path']}/{time}{args['input_suffix']}", "r") as f:
        fg_data = f["fg_xyz_hist"]
        if args["separate_n_c"]:
            fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, int(len(fg_data.keys()) / 2)))
        else:
            fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(fg_data.keys())))
        for i, key in enumerate(fg_data.keys()):
            if args["separate_n_c"]:
                if i % 2 == 0:
                    fg_individual_counts_maps[:, :, :, int(i / 2)] += np.array(
                        fg_data[key][args["min_x_coord"]:args["max_x_coord"],
                        args["min_y_coord"]:args["max_y_coord"],
                        args["min_z_coord"]:args["max_z_coord"]])
                else:
                    fg_individual_counts_maps[:, :, :, int((i - 1) / 2)] += np.array(
                        fg_data[key][args["min_x_coord"]:args["max_x_coord"],
                        args["min_y_coord"]:args["max_y_coord"],
                        args["min_z_coord"]:args["max_z_coord"]])
            else:
                fg_individual_counts_maps[:, :, :, i] = np.array(fg_data[key][args["min_x_coord"]:args["max_x_coord"],
                                                                 args["min_y_coord"]:args["max_y_coord"],
                                                                 args["min_z_coord"]:args["max_z_coord"]])
        floater_data = f["floater_xyz_hist"]
        floater_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(floater_data.keys())))
        floater_sizes = []
        if args["floaters_resistance"]:
            for i, key in enumerate(floater_data.keys()):
                floater_individual_counts_maps[:, :, :, i] = np.array(
                    floater_data[key][args["min_x_coord"]:args["max_x_coord"],
                    args["min_y_coord"]:args["max_y_coord"],
                    args["min_z_coord"]:args["max_z_coord"]])
                # Get the size of the floater according to its IMP type name.
                size = int(float(''.join(map(str, list(filter(str.isdigit, key))))) / args["voxel_size_a"])
                floater_sizes.append(size)
    return fg_individual_counts_maps, floater_individual_counts_maps, floater_sizes


def get_hdf5_size(filename: str) -> tuple[int, int, int]:
    with h5py.File(filename, "r") as f:
        data = f["fg_xyz_hist"]
        arr = np.array(data[list(data.keys())[0]])
        return arr.shape


if __name__ == "__main__":
    main()
    # print((utils.get_coordinate_list(4, 8, 185.0, 75.0)))
    # output.make_bw_legend(70)
    # output.make_matplot_legend(0, 80, 'gist_rainbow')

    # pickle_dict = output.load_pickle("Outputs/12-10-2023-NTR-BATCH/0.pickle")
    # post_analysis(pickle_dict["args"], pickle_dict["real_time_maps"], pickle_dict["needle_maps"])

    # args = arguments.parse_arguments()
    # pickle_dict = output.load_pickle("multi-passive.pickle")
    # args = pickle_dict["args"]
    # real_time_maps = pickle_dict["real_time_maps"]
    # rasterized_maps = pickle_dict["rasterized_maps"]
    # original_shape = (80, 80, 80)
    # center_z = int(original_shape[0] / 2)
    # output.output_gif(args, rasterized_maps,
    #                   f"{args['output_path_prefix']}_rasterized_maps.gif", center_z, args["min_z_coord"],
    #                   args["max_z_coord"])
    # post_analysis(args, real_time_maps, needle_maps=rasterized_maps, original_shape=original_shape)

    # print(utils.concentration_to_amount(200e-6, 1500.0))
    # print(utils.amount_to_concentration(100.0, 1500.0))

    # arr = np.zeros(shape=(100, 100, 100))
    # for _ in range(5000):
    #     utils.get_ball_mask(arr, 50, 50, 50, 1)

    # args = arguments.parse_arguments()
    # print(args)
    # fg_maps = []
    # for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
    #     fgs_counts_map, floaters_counts_map, floater_sizes = get_individual_counts_maps(i, args)
    #     fg_maps.append(floaters_counts_map)
    #     print(i)
    # inner, outer = utils.calculate_z_distribution(np.stack(fg_maps, axis=-1),
    #                                               int((args["tunnel_radius_a"] - args["slab_thickness_a"] / 2) / args[
    #                                                   "voxel_size_a"]))
    # print(inner.tolist())
    # print(outer.tolist())

    # output.visualize_energy_plot(
    #     [-487.1813466285061, -769.1351780149706, -1133.930491931361, -1209.5056373480468, -1263.1785601103497,
    #      -1186.7903747206635, -1205.9046124873373, -1140.012596728273, -1154.5651876646546, -1094.1982466023708,
    #      -1143.1312865679906, -1152.084457782134, -1226.8488343372987, -1299.1298467482216, -1239.7789673942002,
    #      -1163.3445511454859, -1292.5000727242186, -1147.3718651812883, -1169.1973118226733, -1131.865399012099,
    #      -1107.0605916621223, -1255.6150727376846, -1353.4630679567576, -1416.6883595228971, -1495.7820859061874,
    #      -1524.5199267186877, -1460.8948215553316, -1414.2131414895334, -1375.4931270172203, -1487.8541176975089,
    #      -1357.7532280620137, -1280.6090141287377, -1259.2699877612088, -1285.400544531966, -1279.7246970938022,
    #      -1220.9682692414885, -1195.103044133666], "t.png")


###########
# Unused: #
###########

def get_needle_threshold(args: dict[str, any], density_maps: list[np.ndarray]) -> float:
    if args["calc_needle_threshold"] is True:
        threshold = median_threshold(density_maps, args["calc_threshold_r_px"], args["calc_threshold_frac"])
        print(f"Calculated needle threshold is: {threshold}")
        return threshold
    else:
        return args["needle_custom_threshold"]


def median_threshold(density_maps: list[np.ndarray], r: int, frac: float):
    """Calculates a threshold for densities using a fraction of the median in a ball of radius r around the center."""
    x = int(density_maps[0].shape[0] / 2)
    y = int(density_maps[0].shape[1] / 2)
    z = int(density_maps[0].shape[2] / 2)
    vals = []
    for density_map in density_maps:
        arr = utils.get_ball_vals(density_map, x, y, z, r)
        for val in arr:
            vals.append(val)
    return np.median(vals) * frac  # todo this is usually 0.


def get_real_time_maps_helper_sequential(args, get_single):
    """Sequential operation of get_real_time_maps"""
    real_time_maps = []
    for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
        real_time_maps.append(get_single(i))
        if not args["progress_bar"]:
            print(f"Finished {i}ns.", flush=True)
    return real_time_maps
