import h5py
import numpy as np
from itertools import product

import height_funcs
import args as arguments
import utils
import output
import auto_corr


def main():
    args = arguments.parse_arguments()
    print(args)

    real_time_maps = get_real_time_maps(args)
    needle_maps = get_needle_maps(real_time_maps, args)
    # todo (working but not used)
    real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 1)
    taus = auto_corr.calculate_taus(real_time_acorrs)
    output.visualize_taus(taus)
    real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 3)
    taus = auto_corr.calculate_taus(real_time_acorrs)
    output.visualize_taus(taus)
    # output.visualize_auto_corr(real_time_acorrs)
    # needle_acorrs = temporal_auto_correlate(real_time_maps)

    if args["output_gif"]:
        output.output_gif(args, scale_maps(real_time_maps, args["min_z_coord"], args["max_z_coord"]),
                          f"{args['output_gif_path']}_real_time.gif")
        if len(needle_maps) > 0:
            output.output_gif(args, scale_maps(needle_maps, args["min_z_coord"], args["max_z_coord"]),
                              f"{args['output_gif_path']}_needle.gif")
    if args["output_hdf5"]:
        output.output_hdf5(real_time_maps)


def get_real_time_maps(args):
    counts_fgs_maps = []
    real_time_maps = []
    for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
        print(f"{i}")
        counts_fgs_maps.append(get_individual_counts_maps(i, args))
    needle_threshold = get_needle_threshold(args, counts_fgs_maps)
    original_shape = get_hdf5_size(f"{args['existing_files_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    slab_top_z = int(original_shape[2] / 2 + args["slab_thickness_a"] / args["voxel_size_a"])
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    for i, counts_fgs_map in enumerate(counts_fgs_maps):
        height_map = get_height_map(counts_fgs_map, needle_threshold, slab_top_z, center_x, center_y, args)
        print(i)
        real_time_maps.append(height_map)
    return real_time_maps


def get_combined_counts_map(time, args):
    """
    Combines counts maps of all floaters in a HDF5 file, by summing them up.
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]

    with h5py.File(f"{args['existing_files_path']}/{time}.pb.hdf5", "r") as f:
        data = f["floater_xyz_hist"]
        combined_counts_map = np.zeros(shape=(x_size, y_size, z_size))
        for key in data.keys():
            combined_counts_map += np.array(data[key])[args["min_x_coord"]:args["max_x_coord"],
                                   args["min_y_coord"]:args["max_y_coord"],
                                   args["min_z_coord"]:args["max_z_coord"]]
    return combined_counts_map


def get_individual_counts_maps(time, args):
    """
    return 4d array of size (size_x, size_y, size_z, floaters_amount)
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]

    with h5py.File(f"{args['existing_files_path']}/{time}.pb.hdf5", "r") as f:
        data = f["floater_xyz_hist"]
        individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(data.keys())))
        for i, key in enumerate(data.keys()):
            individual_counts_maps[:, :, :, i] = np.array(data[key])[args["min_x_coord"]:args["max_x_coord"],
                                                 args["min_y_coord"]:args["max_y_coord"],
                                                 args["min_z_coord"]:args["max_z_coord"]]
    return individual_counts_maps


def get_height_map(counts_fgs_map, needle_threshold, slab_top_z, center_x, center_y, args):
    summed_counts_map = np.sum(counts_fgs_map, axis=3)
    density_map = counts_fgs_map / (args["interval_ns"] / args["statistics_interval_ns"])
    height_map = np.zeros(shape=counts_fgs_map.shape[:2])
    for x, y in product(range(counts_fgs_map.shape[0]), range(counts_fgs_map.shape[1])):
        is_in_tunnel = utils.is_in_circle(x + args["min_x_coord"],
                                          y + args["min_y_coord"],
                                          args["tunnel_radius_a"] / args["voxel_size_a"],
                                          center_x,
                                          center_y)
        height_map[x, y] = height_funcs.height_func_wrapper(args["z_func"], x, y, counts_fgs_map,
                                                            summed_counts_map,
                                                            density_map,
                                                            needle_threshold,
                                                            slab_top_z,
                                                            is_in_tunnel, args)
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


if __name__ == "__main__":
    main()
