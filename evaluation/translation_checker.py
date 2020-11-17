import argparse
import pickle

from utils.tools import read_data


def check_sentence(args):
    pass


def check_word(args):

    assert args.vocab_path is not None

    with open(args.vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    vocab_data_all = []
    for vocab in vocab_data.values():
        vocab_data_all.extend(list(vocab))

    vocab_data_all = set(vocab_data_all)

    lang_identifier = read_data(args.lang_identifier_path)

    for translation_file_path in args.translation_file_path_list:
        translation_data = read_data(translation_file_path)
        wrong_token = 0
        num_token = 0
        unk_token = 0
        for i, sentence in enumerate(translation_data):
            lang_code = lang_identifier[i]
            sentence = sentence.split()

            for token in sentence:
                if token not in vocab_data[lang_code]:
                    assert token in vocab_data_all
                    if token == "UNK":
                        unk_token += 1
                    else:
                        wrong_token += 1

            num_token += len(sentence)

        print("Number of wrong tokens: {}, radio of wrong tokens: {}, number of UNK tokens: {}, radio of UNK tokens: {}"
              .format(wrong_token, wrong_token / num_token, unk_token, unk_token / num_token))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation_file_path_list", nargs="+")
    parser.add_argument("--lang_identifier_path", required=True)
    parser.add_argument("--operation", required=True, choices=["check_sentence", "check_word"])
    parser.add_argument("--vocab_path")

    args, unknown = parser.parse_known_args()

    if args.operation == "check_sentence":
        check_sentence(args)
    else:
        check_word(args)


if __name__ == "__main__":
    main()
