import argparse
from subprocess import call


def segmentation(args):

    assert len(args.zh_corpus_list) == len(args.segmented_zh_corpus_list)

    for file_path, new_file_path in zip(args.zh_corpus_list, args.segmented_zh_corpus_list):

        command = "python -m jieba -d ' ' {} > {}".format(file_path, new_file_path)
        call(command, shell=True)
        print(command)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--zh_corpus_list", nargs="+")
    parser.add_argument("--segmented_zh_corpus_list", nargs="+")

    args, unknown = parser.parse_known_args()
    segmentation(args)


if __name__ == "__main__":
    main()
