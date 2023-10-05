import argparse
import sys
import shlex


def validate_args(args):
    if args["npc_simulation"]:
        raise Exception("Integrated npc simulation not yet implemented.")
    if args["calc_needle_threshold"] is not None:
        if args["calc_threshold_r_px"] is None or args["calc_threshold_frac"] is None:
            raise Exception("calc-threshold is true. therefore please supply the following arguments: "
                            "calc_threshold_r_px, calc_threshold_frac")
    if args["calc_needle_threshold"] is None and args["needle_custom_threshold"] in None:
        raise Exception(
            "Needle threshold is needed, therefore please supply either the needle-custom-threshold or enable "
            "calc-needle-threshold.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="HS-AFM-Simulation",
        description="A model of high speed atomic force microscopy, based on density Maps from imp's nuclear pore "
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
    parser.add_argument("--input-path",
                        type=str,
                        help="Path to the folder containing hdf5 density map files, to be used in the case of "
                             "--no-npc-simulation flag. Files should be named <delta_time_in_ns>.pb.hdf5")
    parser.add_argument("--simulation-start-time-ns",
                        type=int,
                        help="Start time of the AFM simulation in nanoseconds. (NPC will be simulated since 0)",
                        required=True)
    parser.add_argument("--simulation-end-time-ns",
                        type=int,
                        help="End time of the AFM simulation in nanoseconds. (NPC will be simulated since 0)",
                        required=True)
    parser.add_argument("--interval-ns",
                        type=int,
                        help="Interval time of the NPC simulation in nanoseconds.",
                        required=True)
    parser.add_argument("--voxel-size-a",
                        type=float,
                        help="Size of each voxel in the NPC simulation in Angstrom. (Should be the same as specified in"
                             "the npc simulation configuration file).",
                        required=True)
    parser.add_argument("--tunnel-radius-a",
                        type=float,
                        help="Radius of the NPC tunnel in Angstrom. (Should be the same as specified in"
                             "the npc simulation configuration file).",
                        required=True)
    parser.add_argument("--slab-thickness-a",
                        type=float,
                        help="Thickness of the NPC slab in angstrom. (Should be the same as specified in"
                             "the npc simulation configuration file).",
                        required=True)
    parser.add_argument("--statistics-interval-ns",
                        type=float,
                        help="(Should be the same as specified in the npc simulation configuration file).",
                        required=True)
    parser.add_argument('--torus-slab',
                        action=argparse.BooleanOptionalAction,
                        help="Specified if the slab is a torus. should be the same as in the original simulation.",
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

    # Z height functions:
    parser.add_argument("--z-func",
                        type=str,
                        choices=["z_test", "z_test2"],
                        required=True)
    parser.add_argument('--calc-needle-threshold',
                        action=argparse.BooleanOptionalAction,
                        help="If true, calculates the threshold using the median_threshold function in utils.py.")
    parser.add_argument("--calc-threshold-r-px",
                        type=int,
                        help="The radius around the center to get the median value from in pixels. (for calculating "
                             "the threshold if calc_needle_threshold is true)")
    parser.add_argument("--calc-threshold-frac",
                        type=float,
                        help="The fraction of the median to take for the threshold. (for calculating "
                             "the threshold if calc_needle_threshold is true)")
    parser.add_argument("--needle-custom-threshold",
                        type=float,
                        help="The density under which the needle ignores. Used for all z funcs.")
    parser.add_argument("--needle-radius-px",
                        type=int,
                        help="Determines how far around the origin pixel the needle considers for determining pixel "
                             "height. (Assuming ball shape). Should be greater than 1. Only used for z_top z func.",
                        required=True)
    parser.add_argument("--needle-fraction",
                        type=float,
                        help="Determined the fraction of the sum of density needed to be above the z value in order "
                             "to return in. Should be between 0 and 1 (inclusive). only used for z_fraction z func.",
                        required=True)

    # Needle speed:
    speed_grp = parser.add_mutually_exclusive_group(required=True)
    speed_grp.add_argument("--needle-time-per-line-ns",
                           type=float,
                           help="Determines the amount of time it takes for a needle to pass a full line. Mutually "
                                "exclusive with needle-time-per-pixel-ns.")
    speed_grp.add_argument("--needle-time-per-pixel-ns",
                           type=float,
                           help="Determines the amount of time it takes for a needle to pass a single pixel. Mutually "
                                "exclusive with needle-time-per-line-ns.")
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
    parser.add_argument('--output-pickle',
                        action=argparse.BooleanOptionalAction,
                        help="Outputs a pickle if '--output-pickle', doesn't if --no-output-pickle",
                        required=True)
    parser.add_argument("--output-path-prefix",
                        type=str,
                        help="Path to output gif file.",
                        required=True)
    parser.add_argument('--output-gif-color-map',
                        type=str,
                        help="Determines the matplotlib color map used for the output gif. Some useful ones are: "
                             "'gist_gray' for black and white, 'RdBu' for diverging Red-white-blue.",
                        required=True)
    parser.add_argument("--output_resolution_x",
                        type=int,
                        help="x axis Resolution of output gif in pixels. (Up-scaled from original height Maps size)",
                        required=True)
    parser.add_argument("--output_resolution_y",
                        type=int,
                        help="y axis Resolution of output gif in pixels. (Up-scaled from original height Maps size)",
                        required=True)
    parser.add_argument('--output-hdf5',
                        action=argparse.BooleanOptionalAction,
                        help="Outputs a hdf5 file if '--output-hdf5', doesn't if --no-output-hdf5",
                        required=True)
    parser.add_argument("--output-hdf5-path",
                        type=str,
                        help="Path to output hdf5 file.",
                        required=True)
    if sys.argv[1].startswith('@'):
        args = vars(parser.parse_args(shlex.split(open(sys.argv[1][1:]).read())))
    else:
        args = vars(parser.parse_args())
    validate_args(args)
    return args
