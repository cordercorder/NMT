import sacrebleu
import argparse
import os

from utils.tools import read_data, write_data


def tokenize(args):

    assert os.path.isdir(args.raw_corpus_dir)

    if not os.path.isdir(args.tokenized_corpus_dir):
        os.makedirs(args.tokenized_corpus_dir)

    tokenizer = sacrebleu.TOKENIZERS[sacrebleu.DEFAULT_TOKENIZER]

    for file in os.listdir(args.raw_corpus_dir):

        file_path = os.path.join(args.raw_corpus_dir, file)

        idx = file.rfind(".")
        assert idx != -1

        new_file_name = "{}.{}.{}".format(file[:idx], "tok", file[idx+1:])

        data = read_data(file_path)

        data_tok = [tokenizer(sentence) for sentence in data]

        new_file_path = os.path.join(args.tokenized_corpus_dir, new_file_name)
        write_data(data_tok, new_file_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_corpus_dir", required=True)
    parser.add_argument("--tokenized_corpus_dir", required=True)

    args, unknown = parser.parse_known_args()
    tokenize(args)


if __name__ == "__main__":
    main()
