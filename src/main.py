import gzip
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.stats
import tqdm

import args as arguments
import auto_corr
import height_funcs
import raster
import utils
from output import output


def main() -> None:
    args: dict[str, any] = arguments.parse_arguments() # User supplied arguments
    print(args)
    real_time_maps  = get_real_time_maps(args)
    rasterized_maps = raster.get_rasterized_maps(real_time_maps, args)
    sample_path     = f"{args['input_path']}/{args['simulation_start_time_ns']}{args['input_suffix']}"
    original_shape  = get_hdf5_size(sample_path, args["read_from_gzip"])
    center_z = int(original_shape[0] / 2)
    
    if args["output_pickle"]:
        output.save_pickle(real_time_maps, rasterized_maps, args, f"{args['output_path_prefix']}.pickle")
        
    if args["output_non_raster_gif"]:
        out_path = f"{args['output_path_prefix']}_non_raster.gif"
        output.output_gif(args, np.array(real_time_maps), out_path, center_z, center_z, args["max_z_coord"], add_legend=True, timestamp_step=0.001, add_scale=True)
        
    if args["output_raster_gif"]:
        if len(rasterized_maps) > 0:
            out_path = f"{args['output_path_prefix']}_rasterized.gif"
            output.output_gif(args, rasterized_maps, out_path, center_z, center_z, args["max_z_coord"], add_legend=True)
            
    if args["output_post"]:
        post_analysis(args, real_time_maps, rasterized_maps, original_shape=get_hdf5_size(sample_path, args["read_from_gzip"]))


def post_analysis(args: dict[str, any], real_time_maps: list[np.ndarray], needle_maps: list[np.ndarray],
                  original_shape: tuple[int, int, int]) -> None:
    real_time_acorrs          = auto_corr.temporal_auto_correlate(real_time_maps, 1)
    taus                      = auto_corr.calculate_taus(real_time_acorrs)
    original_centers          = (int(original_shape[0] / 2), int(original_shape[1] / 2), int(original_shape[2] / 2))
    center_x                  = int(original_shape[0] / 2)
    center_y                  = int(original_shape[1] / 2)
    center_z                  = int(original_shape[2] / 2)
    tau_output_path           = f"{args['output_path_prefix']}_taus_real_time.png"
    tau_radial_output_path    = f"{args['output_path_prefix']}_tau_radial_real_time.png"
    real_time_center_x        = int(real_time_maps[0].shape[0] / 2)
    real_time_center_y        = int(real_time_maps[0].shape[1] / 2)
    tau_ring_means            = utils.get_ring_means_array(taus, real_time_center_x, real_time_center_y)
    ring_means                = utils.get_ring_means_array(np.mean(np.dstack(real_time_maps), axis=2), real_time_center_x, real_time_center_y)
    ring_means                = (ring_means - center_z)
    args2                     = args
    args2["tip_radius_px"]    = 0
    radial_height_output_path = f"{args['output_path_prefix']}_height_radial_real_time.png"
    envelope_heights          = [height_funcs.get_slab_top_z(x, center_y, original_centers, args2) - original_centers[2] for x in range(center_x, center_x - len(ring_means), -1)]
    envelope_heights          = np.array(envelope_heights)
    
    output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"], args["max_y_coord"], center_x, center_y, 10, tau_output_path)
    output.visualize_tau_by_radial_distance(tau_ring_means, tau_radial_output_path)
    output.visualize_height_by_radial_distance(ring_means, envelope_heights, radial_height_output_path, sym=True, yrange=[0, 10])
    
    # output.visualize_tcf_samples(real_time_acorrs, taus, 5, 5, f"{args['output_path_prefix']}_tcf_samples.png")
        # real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 3)
    # taus = auto_corr.calculate_taus(real_time_acorrs)
    # output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
    #                       args["max_y_coord"], center_x, center_y, 10)


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
    size_x              = args["max_x_coord"] - args["min_x_coord"]
    size_y              = args["max_y_coord"] - args["min_y_coord"]
    tip_threshold       = args["tip_custom_threshold"]  # todo not using get_needle_threshold
    original_shape      = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}{args['input_suffix']}", args["read_from_gzip"])
    centers             = (int(original_shape[0] / 2), int(original_shape[1] / 2), int(original_shape[2] / 2))
    fg_pdfs             = calculate_normal_pdf(0, args["max_z_coord"] - args["min_z_coord"] + 1, 0, args["fgs_sigma_a"] / args["voxel_size_a"])
    floater_z_pdfs      = calculate_normal_pdf(0, centers[2] * 2, centers[2], args["floaters_sigma_a"] / args["voxel_size_a"])
    floater_radial_pdfs = calculate_normal_pdf(0, int(np.sqrt((size_x / 2) ** 2 + (size_y / 2) ** 2) + 1), 0, ((args["tunnel_radius_a"] - 1 * (args["slab_thickness_a"] / 4)) / args["voxel_size_a"]))
    pdfs                = (fg_pdfs, floater_z_pdfs, floater_radial_pdfs)
    stages_total        = int(args["simulation_end_time_ns"] / args["interval_ns"] - args["simulation_start_time_ns"] / args["interval_ns"])
    get_single          = partial(get_single_real_time_map, args=args, centers=centers, pdfs=pdfs, tip_threshold=tip_threshold)
    real_time_maps      = get_real_time_maps_helper_parallel(args, get_single, stages_total)
    return real_time_maps


def get_single_real_time_map(time: int, args: dict[str, any], centers: tuple[int, int, int],
                             pdfs: tuple[dict[int, float], dict[int, float], dict[int, float]],
                             tip_threshold: float) -> np.array:
    """loads and calculates the height map for a single file/point of time. """
    # Load the file
    fgs_counts_map, floaters_counts_map, floater_sizes = load_individual_counts_maps(time, args)
    
    # Enlarge the floaters data to better represent their actual size.
    if args["enlarge_floaters"]:
        floaters_counts_map = enlarge_floater_size(floaters_counts_map, floater_sizes)
    # enlarge the maps sideways to simulate needle size: todo this is slow (gaussian 25% is slower)

    # fgs_counts_map = enlarge_sideways(fgs_counts_map, args["tip_radius_px"])
    # if len(floater_sizes) != 0:
    #     floaters_counts_map = enlarge_sideways(floaters_counts_map, args["tip_radius_px"])

    fgs_counts_map = scipy.ndimage.gaussian_filter(fgs_counts_map, sigma=args["tip_radius_px"], radius=args["tip_radius_px"], axes=(0, 1))
    if len(floater_sizes) != 0:
        floaters_counts_map = scipy.ndimage.gaussian_filter(floaters_counts_map, sigma=args["tip_radius_px"], radius=args["tip_radius_px"], axes=(0, 1))

    # Perform the height calculation
    height_map = height_funcs.calculate_height_map(fgs_counts_map, floaters_counts_map, tip_threshold, centers, pdfs, floater_sizes, args)
    if not args["progress_bar"]:
        print(time, flush=True)

    return height_map


def get_real_time_maps_helper_parallel(args, get_single, stages_total):
    """Parallel operation of get_real_time_maps"""
    map_range = range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"])
    if args["n_cores"] > 1:
        p = Pool(args["n_cores"]) if args["n_cores"] else Pool()
        if args["progress_bar"]:
            real_time_maps = list(tqdm.tqdm(p.imap(get_single, map_range),total=stages_total, colour='WHITE'))
        else:
            real_time_maps = list(p.imap(get_single, map_range))
        p.close()
        p.join()
    else:
        if args["progress_bar"]:
            real_time_maps = list(tqdm.tqdm(map(get_single, map_range), total=stages_total, colour='WHITE'))
        else:
            real_time_maps = list(map(get_single, map_range))
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
    shape     = floater_individual_counts_maps.shape
    new_maps  = np.zeros(shape=shape)
    mid_x     = int(shape[0] / 2)
    mid_y     = int(shape[1] / 2)
    mid_z     = int(shape[2] / 2)
    # Calculating only len(floater_sized) masks and simply shifting them instead of calculating a mask for every point,
    mid_masks = [utils.get_half_ball_mask(floater_individual_counts_maps[:, :, :, 0], mid_x, mid_y, mid_z,
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
    shape    = maps.shape
    new_maps = np.zeros(shape=shape)
    mid_x    = int(shape[0] / 2)
    mid_y    = int(shape[1] / 2)
    mid_z    = int(shape[2] / 2)
    mask     = utils.get_circle_mask_3d(maps[:, :, :, 0], mid_x, mid_y, mid_z, r)
    for x, y, z, i in zip(*np.nonzero(maps)):
        shift_x  = int(x - mid_x)
        shift_y  = int(y - mid_y)
        shift_z  = int(z - mid_z)
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
    if args["read_from_gzip"]:
        with gzip.open(f"{args['input_path']}/{time}{args['input_suffix']}", "rb") as f:
            f = h5py.File(f, "r")
            return process_individual_counts_maps(args, f)
    else:
        with h5py.File(f"{args['input_path']}/{time}{args['input_suffix']}", "r") as f:
            return process_individual_counts_maps(args, f)


def process_individual_counts_maps(args: dict[str, any], f: h5py.File) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x_size  = args["max_x_coord"] - args["min_x_coord"]
    y_size  = args["max_y_coord"] - args["min_y_coord"]
    z_size  = args["max_z_coord"] - args["min_z_coord"]
    
    # Process fg density
    fg_data = f["fg_xyz_hist"]
    n_fg    = len(fg_data.keys())
    if args["separate_n_c"]:
        fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, int(n_fg / 2)))
    else:
        fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, n_fg))
    if args["fgs_resistance"]:
        for i, key in enumerate(fg_data.keys()):
            cur_data = np.array(
                        fg_data[key][args["min_x_coord"]:args["max_x_coord"],
                                     args["min_y_coord"]:args["max_y_coord"],
                                     args["min_z_coord"]:args["max_z_coord"]])
            if args["separate_n_c"]:
                if i % 2 == 0:
                    fg_individual_counts_maps[:, :, :, int(i / 2)] += cur_data
                else:
                    fg_individual_counts_maps[:, :, :, int((i - 1) / 2)] += cur_data
            else:
                fg_individual_counts_maps[:, :, :, i] = cur_data
                
    # Process floater density
    floater_data                   = f["floater_xyz_hist"]
    floater_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(floater_data.keys())))
    floater_sizes                  = []
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


def get_hdf5_size(filename: str, gzipped: bool) -> tuple[int, int, int]:
    if gzipped:
        with gzip.open(filename, "rb") as f:
            f    = h5py.File(f, "r")
            data = f["fg_xyz_hist"]
            arr  = np.array(data[list(data.keys())[0]])
            return arr.shape
    else:
        with h5py.File(filename, "r") as f:
            data = f["fg_xyz_hist"]
            arr  = np.array(data[list(data.keys())[0]])
            return arr.shape


if __name__ == "__main__":
    main()


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
