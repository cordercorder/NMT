import os
import argparse
from utils.tools import read_data, write_data


def remove_blank(args):
    for file in args.zh_corpus_list:

        data = read_data(file)

        data = ["".join(sentence.strip().split()) for sentence in data]

        directory, file_name = os.path.split(file)
        idx = file_name.rfind(".")

        assert idx != -1

        new_file_name = file_name[:idx] + "_no_blank" + file_name[idx:]
        new_file = os.path.join(directory, new_file_name)
        write_data(data, new_file)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--zh_corpus_list", nargs="+")
    args, unknown = parser.parse_known_args()
    remove_blank(args)


if __name__ == "__main__":
    main()
