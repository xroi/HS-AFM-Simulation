import h5py
import numpy as np
import statsmodels.api as statsmodels
import height_funcs
import args as arguments
import utils
import output
from itertools import product

"""
todo:
refactor needed with height funcs
PROBLEM: if there is something above the membrane, it doesn't get detected. why? should the membrane have density?
"""


def get_combined_density_map(time, args):
    """
    Combines density maps of all floaters in a HDF5 file, by summing them up.
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]
    combined_density_map = np.zeros(shape=(x_size, y_size, z_size))
    with h5py.File(f"{args['existing_files_path']}/{time}.pb.hdf5", "r") as f:
        data = f["floater_xyz_hist"]
        for key in data.keys():
            combined_density_map += np.array(data[key])[args["min_x_coord"]:args["max_x_coord"],
                                    args["min_y_coord"]:args["max_y_coord"],
                                    args["min_z_coord"]:args["max_z_coord"]]
    return combined_density_map


def get_height_map(density_map, height_func, needle_threshold, slab_top_z, center_x, center_y, args):
    height_map = np.zeros(shape=density_map.shape[:2])
    for x, y in product(range(density_map.shape[0]), range(density_map.shape[1])):
        is_in_tunnel = utils.is_in_circle(x + args["min_x_coord"],
                                          y + args["min_y_coord"],
                                          args["tunnel_radius_a"] / args["voxel_size_a"],
                                          center_x,
                                          center_y)
        height_map[x, y] = height_func(x, y, density_map, needle_threshold, slab_top_z, is_in_tunnel)
    return height_map


def get_needle_maps(real_time_maps, args):
    """
    Given real time maps, calculates height maps from the AFM needle 'point of view', i.e. according to its speed.
    The real time map resolution affects this, since for each pixel, the time floors to the most recent image.
    """
    size_x = real_time_maps[0].shape[0]
    size_y = real_time_maps[0].shape[1]
    time_per_line, time_per_pixel = get_times(args, size_x)
    needle_maps = []
    total_time = float(args["simulation_start_time_ns"])
    cur_needle_map_index = 0
    while total_time < args["simulation_end_time_ns"]:
        needle_maps.append(np.zeros(shape=real_time_maps[0].shape))
        for y in range(size_y):
            for x in range(size_x):
                needle_maps[cur_needle_map_index][x, y] = real_time_maps[int(total_time / args["interval_ns"])][x, y]
                total_time += time_per_pixel
                if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                    break
            total_time += time_per_line
            if int(total_time / args["interval_ns"]) >= len(real_time_maps):
                break
        total_time += args["needle_time_between_scans_ns"]
        if int(total_time / args["interval_ns"]) >= len(real_time_maps):
            break
        cur_needle_map_index += 1
    return needle_maps[:cur_needle_map_index]


def get_times(args, size_x):
    if args["needle_time_per_line_ns"] is not None:
        time_per_line = args["needle_time_per_line_ns"]
        time_per_pixel = args["needle_time_per_line_ns"] / size_x
    else:  # args["needle_time_per_pixel_ns"] is not None
        time_per_line = args["needle_time_per_pixel_ns"] * size_x
        time_per_pixel = args["needle_time_per_pixel_ns"]
    return time_per_line, time_per_pixel


def get_hdf5_size(filename):
    with h5py.File(filename, "r") as f:
        data = f["floater_xyz_hist"]
        arr = np.array(data[list(data.keys())[0]])
        return arr.shape


def main():
    args = arguments.parse_arguments()
    print(args)

    real_time_maps = get_real_time_maps(args)
    needle_maps = get_needle_maps(real_time_maps, args)
    # todo (working but not used)
    real_time_acorrs = temporal_auto_correlate(real_time_maps)
    output.visualize_auto_corr(real_time_acorrs)
    # needle_acorrs = temporal_auto_correlate(real_time_maps)

    if args["output_gif"]:
        output.output_gif(args, scale_maps(real_time_maps, args["min_z_coord"], args["max_z_coord"]),
                          f"{args['output_gif_path']}_real_time.gif")
        if len(needle_maps) > 0:
            output.output_gif(args, scale_maps(needle_maps, args["min_z_coord"], args["max_z_coord"]),
                              f"{args['output_gif_path']}_needle.gif")
    if args["output_hdf5"]:
        output.output_hdf5(real_time_maps)


def scale_maps(maps, min_z, max_z):
    """Scale z values to be between 0 and 1 (for visualization)"""
    scaled_maps = []
    for i in range(len(maps)):
        scaled_maps.append((maps[i] - min_z) / (max_z - 1 - min_z))
    return scaled_maps


def get_needle_threshold(args, density_maps):
    if args["calc_needle_threshold"] is True:
        threshold = utils.median_threshold(density_maps, args["calc_threshold_r_px"], args["calc_threshold_frac"])
        print(f"Calculated needle threshold is: {threshold}")
        return threshold
    else:
        return args["needle_custom_threshold"]


def get_real_time_maps(args):
    density_maps = []
    real_time_maps = []
    height_func = height_funcs.get_height_func(args["z_func"])
    for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
        print(f"{i}")
        density_maps.append(get_combined_density_map(i, args))
    needle_threshold = get_needle_threshold(args, density_maps)
    original_shape = get_hdf5_size(f"{args['existing_files_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    slab_top_z = int(original_shape[2] / 2 + args["slab_thickness_a"] / args["voxel_size_a"])
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    for i, density_map in enumerate(density_maps):
        height_map = get_height_map(density_map, height_func, needle_threshold, slab_top_z, center_x, center_y, args)
        real_time_maps.append(height_map)
    return real_time_maps


def temporal_auto_correlate(maps):
    """
    Calculates the temporal auto correlation of each pixel with itself over different time lags.
    """
    stacked_maps = np.dstack(maps)
    nlags = int(min(10 * np.log10(stacked_maps.shape[2]), stacked_maps.shape[2] - 1))
    temporal_auto_correlations = np.zeros(shape=(stacked_maps.shape[0], stacked_maps.shape[1], nlags + 1))
    for x, y in product(range(stacked_maps.shape[0]), range(stacked_maps.shape[1])):
        temporal_auto_correlations[x, y, :] = statsmodels.tsa.stattools.acf(stacked_maps[x, y, :])
    return temporal_auto_correlations


if __name__ == "__main__":
    main()
