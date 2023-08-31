import argparse
import h5py
import numpy as np
import scipy as scipy
from PIL import Image, ImageSequence
import height_funcs
import pandas as pd

"""
todo:
1. Add hdf5 output option.
2. Single pixel height as average of box.
"""


def validate_args(args):
    if args["npc_simulation"]:
        # todo
        raise Exception("ERROR: Integrated npc simulation not yet implemented.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="HS-AFM-Simulation",
        description="A model of high speed atomic force microscopy, based on density maps from imp's nuclear pore "
                    "complex transport module.")
    # ========================= #
    # NPC SIMULATION PARAMETERS #
    # ========================= #
    parser.add_argument('--npc-simulation',
                        action=argparse.BooleanOptionalAction,
                        help="In the case of --npc-simulation flag, the npc simulations run 'live'. In the case of "
                             "--no-npc-simulation flag, the program uses files in the folder specified with "
                             "--existing_files_path",
                        required=True)
    parser.add_argument("--existing-files-path",
                        type=str,
                        help="Path to the folder containing hdf5 density map files, to be used in the case of "
                             "--no-npc-simulation flag. Files should be named <delta_time_in_ns>.pb.hdf5")
    parser.add_argument("--simulation-time-ns",
                        type=int,
                        help="How long the simulation runs, in nanoseconds.",
                        required=True)
    parser.add_argument("--interval-ns",
                        type=int,
                        help="Interval between calculation of the AFM map, in nanoseconds. This should correlate with"
                             "the time the AFM needle stays on a 'pixel'",
                        required=True)
    # ============== #
    # AFM PARAMETERS #
    # ============== #
    parser.add_argument("--min-x-coord",
                        type=int,
                        help="Specifies the first pixel on the X axis on which the simulation is ran (inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--max-x-coord",
                        type=int,
                        help="Specifies the last pixel on the X axis on which the simulation is ran (not inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--min-y-coord",
                        type=int,
                        help="Specifies the first pixel on the Y axis on which the simulation is ran (inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--max-y-coord",
                        type=int,
                        help="Specifies the last pixel on the Y axis on which the simulation is ran (not inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--min-z-coord",
                        type=int,
                        help="Specifies the first pixel on the Z axis on which the simulation is ran (inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--max-z-coord",
                        type=int,
                        help="Specifies the last pixel on the Z axis on which the simulation is ran (not inclusive). "
                             "Count starting from 0.",
                        required=True)
    parser.add_argument("--needle-threshold",
                        type=float,
                        help="The density under which the needle ignores.",
                        required=True)
    parser.add_argument("--needle-radius-px",
                        type=int,
                        help="Determines how far around the origin pixel the needle considers for determining pixel "
                             "height. (Assuming ball shape). Should be greater than 1.",
                        required=True)
    parser.add_argument("--needle-time-per-line-ns",
                        type=float,
                        help="Determines the amount of time it takes for a needle to pass a full line.",
                        required=True)
    parser.add_argument("--needle-time-between-scans-ns",
                        type=float,
                        help="Determines the amount of time it takes for the needle to return to the starting point "
                             "to start the next frame.",
                        required=True)

    # ================= #
    # OUTPUT PARAMETERS #
    # ================= #
    parser.add_argument('--output-gif',
                        action=argparse.BooleanOptionalAction,
                        help="Outputs a gif if '--output-gif', doesn't if --no-output-gif",
                        required=True)
    parser.add_argument("--output-gif-path",
                        type=str,
                        help="Path to output gif file.",
                        required=True)
    parser.add_argument("--output_resolution_x",
                        type=int,
                        help="x axis Resolution of output gif in pixels. (Up-scaled from original height maps size)",
                        required=True)
    parser.add_argument("--output_resolution_y",
                        type=int,
                        help="y axis Resolution of output gif in pixels. (Up-scaled from original height maps size)",
                        required=True)
    parser.add_argument('--output-hdf5',
                        action=argparse.BooleanOptionalAction,
                        help="Outputs a hdf5 file if '--output-hdf5', doesn't if --no-output-hdf5",
                        required=True)
    parser.add_argument("--output-hdf5-path",
                        type=str,
                        help="Path to output hdf5 file.",
                        required=True)

    args = vars(parser.parse_args())
    validate_args(args)
    return args


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


def get_height_map(combined_density_map, args):
    height_map = np.zeros(shape=combined_density_map.shape[:2])
    for x in range(combined_density_map.shape[0]):
        for y in range(combined_density_map.shape[1]):
            height_map[x, y] = height_funcs.get_single_pixel_height_new(x, y, combined_density_map, args)
    # Min max scale the data. todo, maybe bad approach (good for visualization) (maybe add as parameter?)
    height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    return height_map


def output_hdf5(maps):
    # todo
    raise Exception("not yet implemented.")


def get_needle_maps(real_time_maps, args):
    size_x = real_time_maps[0].shape[0]
    size_y = real_time_maps[0].shape[1]
    time_per_pixel = args["needle_time_per_line_ns"] / size_x
    needle_maps = []
    total_time = 0.0
    cur_needle_map_index = 0
    while total_time < args["simulation_time_ns"]:
        needle_maps.append(np.zeros(shape=real_time_maps[0].shape))
        for y in range(size_y):
            for x in range(size_x):
                needle_maps[cur_needle_map_index][x, y] = real_time_maps[int(total_time / args["interval_ns"])][x, y]
                total_time += time_per_pixel
                if total_time >= total_time < args["simulation_time_ns"]:
                    break
            total_time += args["needle_time_per_line_ns"]
            if total_time >= total_time < args["simulation_time_ns"]:
                break
        total_time += args["needle_time_between_scans_ns"]
        if total_time >= total_time < args["simulation_time_ns"]:
            break
        cur_needle_map_index += 1
    return needle_maps[:cur_needle_map_index]


def main():
    args = parse_arguments()
    print(args)

    real_time_maps = []
    for i in range(args["interval_ns"], args["simulation_time_ns"], args["interval_ns"]):
        print(i)
        combined_density_map = get_combined_density_map(i, args)
        height_map = get_height_map(combined_density_map, args)
        real_time_maps.append(height_map)
    acorrs = temporal_auto_correlate(real_time_maps)
    needle_maps = get_needle_maps(real_time_maps, args)
    if args["output_gif"]:
        output_gif(args, real_time_maps, f"{args['output_gif_path']}_real_time.gif")
        if len(needle_maps) > 0:
            output_gif(args, needle_maps, f"{args['output_gif_path']}_needle.gif")
    if args["output_hdf5"]:
        output_hdf5(real_time_maps)


def temporal_auto_correlate(maps):
    """
    Calculates the temporal auto correlation of each pixel with itself over different time lags.
    """
    stacked_maps = np.dstack(maps)
    temporal_auto_correlations = np.zeros(stacked_maps.shape)
    for x in range(stacked_maps.shape[0]):
        for y in range(stacked_maps.shape[1]):
            pixel_time_series = pd.Series(stacked_maps[x, y, :])
            auto_corrs = []
            for delta_t in range(stacked_maps.shape[2]):
                auto_corrs.append(pixel_time_series.autocorr(lag=delta_t))
            temporal_auto_correlations[x, y, :] = np.array(auto_corrs)
    return temporal_auto_correlations


def output_gif(args, maps, filename):
    images = []
    for height_map in maps:
        im = Image.fromarray((height_map * 255).astype(np.uint8)).resize(
            (args["output_resolution_x"], args["output_resolution_y"]), resample=Image.BOX)
        images.append(im)
    images[0].save(filename, append_images=images[1:], save_all=True, duration=100,
                   loop=0)


if __name__ == "__main__":
    main()
