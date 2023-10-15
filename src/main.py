import h5py
import numpy as np
from alive_progress import alive_bar
import scipy.stats

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
    if args["output_pickle"]:
        output.save_pickle(real_time_maps, needle_maps, args, f"{args['output_path_prefix']}.pickle")
    if args["output_gif"]:
        original_shape = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
        center_z = int(original_shape[0] / 2)
        output.output_gif(args, np.array(real_time_maps),
                          f"{args['output_path_prefix']}_real_time.gif", center_z, args["min_z_coord"],
                          args["max_z_coord"])
        if len(needle_maps) > 0:
            output.output_gif(args, needle_maps,
                              f"{args['output_path_prefix']}_needle.gif", center_z, args["min_z_coord"],
                              args["max_z_coord"])

    post_analysis(args, real_time_maps, needle_maps)


def post_analysis(args, real_time_maps, needle_maps):
    real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 1)
    taus = auto_corr.calculate_taus(real_time_acorrs)
    original_shape = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    center_x = int(original_shape[0] / 2)
    center_y = int(original_shape[1] / 2)
    center_z = int(original_shape[2] / 2)
    output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
                          args["max_y_coord"], center_x, center_y, 10,
                          f"{args['output_path_prefix']}_taus_real_time.png")
    # real_time_acorrs = auto_corr.temporal_auto_correlate(real_time_maps, 3)
    # taus = auto_corr.calculate_taus(real_time_acorrs)
    # output.visualize_taus(taus, args["voxel_size_a"], args["min_x_coord"], args["max_x_coord"], args["min_y_coord"],
    #                       args["max_y_coord"], center_x, center_y, 10)

    max_r = utils.get_max_r(real_time_maps[0].shape, real_time_maps[0].shape[0] / 2, real_time_maps[0].shape[1] / 2)
    ring_means = utils.get_ring_means_array(real_time_maps, real_time_maps[0].shape[0] / 2,
                                            real_time_maps[0].shape[1] / 2, max_r)
    ring_means = (ring_means - center_z)
    output.visualize_ring_means(ring_means, args["voxel_size_a"],
                                f"{args['output_path_prefix']}_ring_means_real_time.png")


def calculate_normal_pdf(min_z, max_z, mu, sigma):
    vals = {}
    norm = scipy.stats.norm(mu, sigma)
    for i in range(min_z, max_z):
        vals[i] = norm.pdf(i)
    return vals


def get_real_time_maps(args):
    real_time_maps = []
    needle_threshold = args["needle_custom_threshold"]  # todo not using get_needle_threshold
    original_shape = get_hdf5_size(f"{args['input_path']}/{args['simulation_start_time_ns']}.pb.hdf5")
    centers = (int(original_shape[0] / 2), int(original_shape[1] / 2), int(original_shape[2] / 2))
    fg_pdfs = calculate_normal_pdf(0, args["max_z_coord"] - args["min_z_coord"] + 1, 0,
                                   2 * args["slab_thickness_a"] / args["voxel_size_a"])
    floater_pdfs = calculate_normal_pdf(0, centers[2] * 2, centers[2],
                                        args["slab_thickness_a"] / args["voxel_size_a"])
    pdfs = (fg_pdfs, floater_pdfs)
    with alive_bar(int(args["simulation_end_time_ns"] / args["interval_ns"]), force_tty=True) as bar:
        for i in range(args["simulation_start_time_ns"], args["simulation_end_time_ns"], args["interval_ns"]):
            fgs_counts_map, floaters_counts_map = get_individual_counts_maps(i, args)
            height_map = height_funcs.z_test2(fgs_counts_map, floaters_counts_map, needle_threshold, centers, pdfs,
                                              args)
            real_time_maps.append(height_map)
            bar()
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
    returns a tuple of two 4d array of size (size_x, size_y, size_z, mol_number), one for floaters and one for fgs.
    """
    x_size = args["max_x_coord"] - args["min_x_coord"]
    y_size = args["max_y_coord"] - args["min_y_coord"]
    z_size = args["max_z_coord"] - args["min_z_coord"]

    with h5py.File(f"{args['input_path']}/{time}.pb.hdf5", "r") as f:
        fg_data = f["fg_xyz_hist"]
        fg_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(fg_data.keys())))
        for i, key in enumerate(fg_data.keys()):
            fg_individual_counts_maps[:, :, :, i] = np.array(fg_data[key])[
                                                    args["min_x_coord"]:args["max_x_coord"],
                                                    args["min_y_coord"]:args["max_y_coord"],
                                                    args["min_z_coord"]:args["max_z_coord"]]
        floater_data = f["floater_xyz_hist"]
        floater_individual_counts_maps = np.zeros(shape=(x_size, y_size, z_size, len(floater_data.keys())))
        if args["floaters_resistance"]:
            for i, key in enumerate(floater_data.keys()):
                floater_individual_counts_maps[:, :, :, i] = np.array(floater_data[key])[
                                                             args["min_x_coord"]:args["max_x_coord"],
                                                             args["min_y_coord"]:args["max_y_coord"],
                                                             args["min_z_coord"]:args["max_z_coord"]]
    # return np.append(fg_individual_counts_maps, floater_individual_counts_maps, axis=3)
    return fg_individual_counts_maps, floater_individual_counts_maps


def get_needle_maps(real_time_maps, args):
    """
    Given real time Maps, calculates height Maps from the AFM needle 'point of view', i.e. according to its speed.
    The real time map resolution affects this, since for each pixel, the time floors to the most recent image.
    """
    if len(real_time_maps) <= 1:
        return []
    size_x = real_time_maps[0].shape[0]
    size_y = real_time_maps[0].shape[1]
    time_per_line, time_per_pixel = get_times(args, size_x)
    needle_maps = []
    total_time = float(args["interval_ns"])
    cur_needle_map_index = 0
    if int(total_time / args["interval_ns"]) >= len(real_time_maps):
        return needle_maps
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
    # print((utils.get_coordinate_list(4, 8, 185.0, 75.0)))
    # output.make_bw_legend(70)
    # output.make_matplot_legend(0, 80, 'gist_rainbow')

    # pickle_dict = output.load_pickle("Outputs/08-10-2023-NTR/08-10-2023-NTR.pickle")
    # post_analysis(pickle_dict["args"], pickle_dict["real_time_maps"], pickle_dict["needle_maps"])

    # print(utils.concentration_to_amount(0.001, 1000.0))
    # print(utils.amount_to_concentration(100.0, 1500.0))
