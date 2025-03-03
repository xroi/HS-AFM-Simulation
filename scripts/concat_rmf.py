import RMF
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to folder")
    parser.add_argument("--start-time-ns", type=int, required=True)
    parser.add_argument("--end-time-ns", type=int, required=True)
    parser.add_argument("--interval-ns", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return vars(parser.parse_args())


def main():
    args = parse_arguments()
    out_fh = RMF.create_rmf_file(args["output_path"])
    in_fh = RMF.open_rmf_file_read_only(f"{args['input_path']}/{args['start_time_ns']}.movie.rmf")
    RMF.clone_file_info(in_fh, out_fh)
    RMF.clone_hierarchy(in_fh, out_fh)
    RMF.clone_static_frame(in_fh, out_fh)
    for i in range(args["start_time_ns"], args["end_time_ns"], args["start_time_ns"]):
        print(i)
        in_fh = RMF.open_rmf_file_read_only(f"{args['input_path']}/{i}.movie.rmf")
        for f_id, f in enumerate(in_fh.get_frames()):
            in_fh.set_current_frame(f)
            out_fh.add_frame(in_fh.get_name(f), in_fh.get_type(f))
            RMF.clone_loaded_frame(in_fh, out_fh)


if __name__ == "__main__":
    main()
