import os
import argparse
from subprocess import call
from utils.tools import read_data, write_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file_path", required=True)
    parser.add_argument("--tgt_file_path", required=True)

    # optional argument
    parser.add_argument("--combined_file_path")
    parser.add_argument("--align_data_path")

    args, unknown = parser.parse_known_args()

    src_data = read_data(args.src_file_path)
    tgt_data = read_data(args.tgt_file_path)

    combined_data = []
    for src, tgt in zip(src_data, tgt_data):
        src = src.strip()
        tgt = tgt.strip()
        combined_data.append(src + " ||| " + tgt)

    if args.combined_file_path:
        write_data(combined_data, args.combined_file_path)
    else:
        directory, src_file_name = os.path.split(args.src_file_path)
        new_file_name = "combined_data.txt"
        new_file_path = os.path.join(directory, new_file_name)
        write_data(combined_data, new_file_path)
        args.combined_file_path = new_file_path

    if not args.align_data_path:
        directory, src_file_name = os.path.split(args.src_file_path)
        new_file_name = "align_data.txt"
        new_file_path = os.path.join(directory, new_file_name)
        args.align_data_path = new_file_path

    command = "/data/rrjin/fast_align/build/fast_align -i {} -d -o -v > {}".format(args.combined_file_path,
                                                                                   args.align_data_path)
    call(command, shell=True)


if __name__ == "__main__":
    main()
