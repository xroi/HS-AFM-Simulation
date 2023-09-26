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
    post_analysis(args, real_time_maps, needle_maps)

    if args["output_gif"]:
        original_shape = get_hdf5_size(f"{args['existing_files_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
        center_z = int(original_shape[0] / 2)
        output.output_gif(args, real_time_maps,
                          f"{args['output_gif_path']}_real_time.gif", center_z, args["min_z_coord"],
                          args["max_z_coord"], args["color_gif"])
        if len(needle_maps) > 0:
            output.output_gif(args, needle_maps,
                              f"{args['output_gif_path']}_needle.gif", center_z, args["min_z_coord"],
                              args["max_z_coord"], args["color_gif"])
    if args["output_hdf5"]:
        output.output_hdf5(real_time_maps)


def post_analysis(args, real_time_maps, needle_maps):
    real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 1)
    taus = auto_corr.calculate_taus(real_time_acorrs)
    original_shape = get_hdf5_size(f"{args['existing_files_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
                          args["max_y_coord"], center_x, center_y, 10)
    # real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 3)
    # taus = auto_corr.calculate_taus(real_time_acorrs)
    # output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
    #                       args["max_y_coord"], center_x, center_y, 10)


def get_real_time_maps(args):
    counts_fgs_maps = []
    real_time_maps = []
    for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
        print(f"{i}")
        counts_fgs_maps.append(get_individual_counts_maps(i, args))
    needle_threshold = get_needle_threshold(args, counts_fgs_maps)
    original_shape = get_hdf5_size(f"{args['existing_files_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    # slab_top_z = int(original_shape[2] / 2 + args["slab_thickness_a"] / args["voxel_size_a"])
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    center_z = int(original_shape[2] / 2)
    for i, counts_fgs_map in enumerate(counts_fgs_maps):
        height_map = get_height_map(counts_fgs_map, needle_threshold, center_x, center_y, center_z, args)
        print(i)
        real_time_maps.append(height_map)
    return real_time_maps


# def get_combined_counts_map(time, args):
#     """
#     Deprecated, Combines counts Maps of all floaters in a HDF5 file, by summing them up.
#     """
#     x_size = args["max_x_coord"] - args["min_x_coord"]
#     y_size = args["max_y_coord"] - args["min_y_coord"]
#     z_size = args["max_z_coord"] - args["min_z_coord"]
#
#     with h5py.File(f"{args['existing_files_path']}/{time}.pb.hdf5", "r") as f:
#         combined_counts_map = np.zeros(shape=(x_size, y_size, z_size))
#         data = f["fg_xyz_hist"]
#         for key in data.keys():
#             combined_counts_map += np.array(data[key])[args["min_x_coord"]:args["max_x_coord"],
#                                    args["min_y_coord"]:args["max_y_coord"],
#                                    args["min_z_coord"]:args["max_z_coord"]]
#     return combined_counts_map


def get_individual_counts_maps(time, args):
    """
    return 4d array of size (size_x, size_y, size_z, floaters_amount)
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]

    with h5py.File(f"{args['existing_files_path']}/{time}.pb.hdf5", "r") as f:
        fg_data = f["fg_xyz_hist"]
        fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(fg_data.keys())))
        for i, key in enumerate(fg_data.keys()):
            fg_individual_counts_maps[:, :, :, i] = np.array(fg_data[key])[
                                                    args["min_x_coord"]:args["max_x_coord"],
                                                    args["min_y_coord"]:args["max_y_coord"],
                                                    args["min_z_coord"]:args["max_z_coord"]]
        floater_data = f["floater_xyz_hist"]
        floater_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(floater_data.keys())))
        for i, key in enumerate(floater_data.keys()):
            floater_individual_counts_maps[:, :, :, i] = np.array(floater_data[key])[
                                                         args["min_x_coord"]:args["max_x_coord"],
                                                         args["min_y_coord"]:args["max_y_coord"],
                                                         args["min_z_coord"]:args["max_z_coord"]]
    return np.append(fg_individual_counts_maps, floater_individual_counts_maps, axis=3)


def get_height_map(counts_fgs_map, needle_threshold, center_x, center_y, center_z, args):
    summed_counts_map = np.sum(counts_fgs_map, axis=3)
    density_map = counts_fgs_map / (args["interval_ns"] / args["statistics_interval_ns"])
    height_map = np.zeros(shape=counts_fgs_map.shape[:2])
    for x, y in product(range(counts_fgs_map.shape[0]), range(counts_fgs_map.shape[1])):
        if args["torus_slab"]:
            slab_top_z = utils.get_torus_top_z(x,
                                               y,
                                               center_x,
                                               center_y,
                                               center_z,
                                               args["tunnel_radius_a"] / args["voxel_size_a"],
                                               (args["slab_thickness_a"] / args["voxel_size_a"]) / 2)
        else:
            slab_top_z = -1 if utils.is_in_circle(x, y, args["tunnel_radius_a"] / args["voxel_size_a"], center_x,
                                                  center_y) else center_z * (args["slab_thickness_a"] / 2) / args[
                "voxel_size_a"]
        height_map[x, y] = height_funcs.z_test(x, y, summed_counts_map, needle_threshold, slab_top_z, center_z,
                                               args["min_z_coord"])
    return height_map


def get_needle_maps(real_time_maps, args):
    """
    Given real time Maps, calculates height Maps from the AFM needle 'point of view', i.e. according to its speed.
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
        data = f["fg_xyz_hist"]
        arr = np.array(data[list(data.keys())[0]])
        return arr.shape


def get_needle_threshold(args, density_maps):
    if args["calc_needle_threshold"] is True:
        threshold = utils.median_threshold(density_maps, args["calc_threshold_r_px"], args["calc_threshold_frac"])
        print(f"Calculated needle threshold is: {threshold}")
        return threshold
    else:
        return args["needle_custom_threshold"]


if __name__ == "__main__":
    main()
    # print((utils.get_coordinate_list(4, 12, 480.0, 150.0, 900.0)))
    # output.make_bw_legend(70)
    # output.make_matplot_legend(700, 'RdBu')
