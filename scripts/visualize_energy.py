import argparse
import sys
from IMP.npctransport import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to folder")
    parser.add_argument("--output-path", type=str, required=True)
    return vars(parser.parse_args())


def main():
    args = parse_arguments()
    with open(args["input_path"], "rb") as f:
        output = Output()
        fstring = f.read()
        output.ParseFromString(fstring)
        time_ns_arr = []
        energy_arr = []
        for i in range(len(output.statistics.global_order_params)):
            time_ns_arr.append(output.statistics.global_order_params[i].time_ns)
            energy_arr.append(output.statistics.global_order_params[i].energy)
        print(time_ns_arr)
        print(energy_arr)


if __name__ == "__main__":
    main()
