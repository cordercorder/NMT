import argparse
import os

from utils.tools import read_data

from nltk.translate.bleu_score import corpus_bleu


def bleu_calculation(args):

    src_file = read_data(args.src_file_path)
    lang_identifier = {}

    for i, sentence in enumerate(src_file):

        sentence = sentence.split()
        lang_code = sentence[0]

        assert lang_code.startswith("<") and lang_code.endswith(">")

        lang_code = lang_code[1:-1]

        if lang_code not in lang_identifier:
            lang_identifier[lang_code] = [i]
        else:
            lang_identifier[lang_code].append(i)

    reference_data = read_data(args.reference_path)
    reference_data = [[sentence.split()] for sentence in reference_data]

    reference_data_per_language = {}
    for k, v in lang_identifier.items():
        reference_data_per_language[k] = [reference_data[line_number] for line_number in v]

    for translation_path in args.translation_path_list:

        _, file_name = os.path.split(translation_path)

        print("Translation in: {}\n".format(file_name))

        translation_data = read_data(translation_path)
        translation_data = [sentence.split() for sentence in translation_data]

        translation_data_per_language = {}
        for k, v in lang_identifier.items():
            translation_data_per_language[k] = [translation_data[line_number] for line_number in v]

        for lang in lang_identifier:
            bleu_score = corpus_bleu(reference_data_per_language[lang], translation_data_per_language[lang]) * 100
            print("language: {}, bleu: {}".format(lang, bleu_score))

        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file_path", required=True, help="The first token in sentences of "
                                                               "src file is language identifier")
    parser.add_argument("--translation_path_list", nargs="+")
    parser.add_argument("--reference_path", required=True)

    args, unknown = parser.parse_known_args()
    bleu_calculation(args)


if __name__ == "__main__":
    main()
