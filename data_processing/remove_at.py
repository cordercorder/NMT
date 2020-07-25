import argparse
import os
from utils.tools import read_data, write_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", required=True)

    args, unknown = parser.parse_known_args()

    data = read_data(args.file_path)

    directory, file_name = os.path.split(args.file_path)

    data_after_processing = []

    for line in data:

        line = "".join(c for c in line if c != "@")

        data_after_processing.append(line)

    idx = file_name.rfind(".")

    if idx == -1:
        new_file_name = file_name + "_remove_at"
    else:
        new_file_name = file_name[:idx] + "_remove_at" + file_name[idx:]

    write_data(data_after_processing, os.path.join(directory, new_file_name))


if __name__ == "__main__":
    main()
