import argparse
import h5py
import numpy as np
from PIL import Image, ImageSequence


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="HS-AFM-Simulation",
        description="A model of high speed atomic force microscopy, based on density maps from imp's nuclear pore "
                    "complex transport module.")
    # ========================= #
    # SUB-SIMULATION PARAMETERS #
    # ========================= #
    parser.add_argument('--npc-simulation',
                        action=argparse.BooleanOptionalAction,
                        help="In the case of --npc-simulation flag, the npc simulations run 'live'. In the case of "
                             "--no-npc-simulation flag, the program uses files in the folder specified with "
                             "--existing_files_path")
    parser.add_argument("--existing-files-path",
                        type=str,
                        help="Path to the folder containing hdf5 density map files, to be used in the case of "
                             "--no-npc-simulation flag. Files should be named <delta_time_in_ns>.pb.hdf5")
    parser.add_argument("--simulation-time-ns",
                        type=int,
                        help="How long the simulation runs, in nanoseconds.")
    parser.add_argument("--interval-ns",
                        type=int,
                        help="Interval between calculation of the AFM map, in nanoseconds.")

    # parser.add_argument("--afm-image-resolution",
    #                     type=int,
    #                     help="Resolution of simulated AFM image. A single int. i.e. if 80 was provided and "
    #                          "--afm_dimension is 2 then the resolution will be 80x80")
    parser.add_argument("--afm-dimension",
                        type=int,
                        choices=[1, 2])
    parser.add_argument("--x-coord",
                        type=int,
                        help="To be used with --afm_dimension=1. Specifies the pixel on the X axis on which the "
                             "simulation is ran. Mutually exclusive with --y_coord")
    parser.add_argument("--y-coord",
                        type=int,
                        help="To be used with --afm_dimension=1. Specifies the pixel on the Y axis on which the "
                             "simulation is ran. Mutually exclusive with --x_coord")
    # ================= #
    # OUTPUT PARAMETERS #
    # ================= #
    parser.add_argument("--output-path",
                        type=str,
                        help="Path to output gif file.")
    parser.add_argument("--output_resolution_x",
                        type=int,
                        help="x axis Resolution of output gif in pixels.")
    parser.add_argument("--output_resolution_y",
                        type=int,
                        help="y axis Resolution of output gif in pixels.")
    return vars(parser.parse_args())


def get_combined_density_map(time, existing_files_path, afm_image_resolution):
    combined_density_map = np.zeros(
        shape=(afm_image_resolution, afm_image_resolution, afm_image_resolution))
    with h5py.File(f"{existing_files_path}/{time}.pb.hdf5", "r") as f:
        data = f["floater_xyz_hist"]
        for key in data.keys():
            combined_density_map += np.array(data[key])
    return combined_density_map


def get_single_pixel_height(x, y, combined_density_map):
    # todo: currently getting the highest pixel that has density grater than 0, maybe this isn't the correct approach
    for z in range(combined_density_map.shape[2] - 1, -1, -1):
        if combined_density_map[x, y, z] > 0:
            return z / combined_density_map.shape[2]
    return 0


def get_height_map(combined_density_map):
    height_map = np.zeros(shape=combined_density_map.shape[:2])
    for x in range(combined_density_map.shape[0]):
        for y in range(combined_density_map.shape[1]):
            height_map[x, y] = get_single_pixel_height(x, y, combined_density_map)
    return height_map


def main():
    args = parse_arguments()
    print(args)
    if args["npc_simulation"]:
        raise Exception("ERROR: Integrated npc simulation not yet implemented.")
    images = []
    for i in range(args["interval_ns"], args["simulation_time_ns"], args["interval_ns"]):
        combined_density_map = get_combined_density_map(i, args["existing_files_path"], args["afm_image_resolution"])
        height_map = get_height_map(combined_density_map)
        im = Image.fromarray((height_map * 255).astype(np.uint8))
        im = im.resize((600, 600), resample=Image.BOX)
        im.save(f"{i}.png")
        images.append(im)

    images[0].save(args["output_path"], append_images=images[1:], save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    main()
