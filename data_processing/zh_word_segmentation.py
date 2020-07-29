import argparse
from subprocess import call
import os


def segmentation(args):

    for file in args.zh_corpus_list:

        directory, file_name = os.path.split(file)
        idx = file_name.rfind(".")

        assert idx != -1

        new_file_name = file_name[:idx] + "_seg" + file_name[idx:]
        new_file = os.path.join(directory, new_file_name)

        command = "python -m jieba -d ' ' {} > {}".format(file, new_file)
        call(command, shell=True)
        print(command)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--zh_corpus_list", nargs="+")
    args, unknown = parser.parse_known_args()
    segmentation(args)


if __name__ == "__main__":
    main()
