import argparse
import os
from utils.tools import read_data, write_data


def get_file_path(file_path):

    directory, file_name = os.path.split(file_path)

    idx = file_name.rfind(".")
    if idx == -1:
        new_file_name = file_name + "_filtered"
    else:
        new_file_name = file_name[:idx] + "_filtered" + file_name[idx:]

    new_file_path = os.path.join(directory, new_file_name)

    return new_file_path


def remove_long_sentence(src_file_path_list: list, tgt_file_path_list: list, max_sentence_length: int):

    assert len(src_file_path_list) == len(tgt_file_path_list)

    for src_file_path, tgt_file_path in zip(src_file_path_list, tgt_file_path_list):

        src_data = read_data(src_file_path)
        tgt_data = read_data(tgt_file_path)

        src_sentence_filtered = []
        tgt_sentence_filtered = []

        for src_sentence, tgt_sentence in zip(src_data, tgt_data):

            if len(src_sentence.split()) <= max_sentence_length and len(tgt_sentence.split()) <= max_sentence_length:
                src_sentence_filtered.append(src_sentence)
                tgt_sentence_filtered.append(tgt_sentence)

        new_src_file_path = get_file_path(src_file_path)
        new_tgt_file_path = get_file_path(tgt_file_path)

        write_data(src_sentence_filtered, new_src_file_path)
        write_data(tgt_sentence_filtered, new_tgt_file_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file_path_list", nargs="+")
    parser.add_argument("--tgt_file_path_list", nargs="+")
    parser.add_argument("--max_sentence_length", type=int)
    args, unknown = parser.parse_known_args()
    remove_long_sentence(args.src_file_path_list, args.tgt_file_path_list, args.max_sentence_length)


if __name__ == "__main__":
    main()
