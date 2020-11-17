import argparse
import os
import json
import logging

from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu as sacre_corpus_bleu

from typing import Dict

from utils.tools import read_data

logging.disable(logging.WARNING)


def show_corpus_statistics(lang_identifier: Dict, language_dict: Dict):

    for lang in lang_identifier:
        print("IOS639-2:{},IOS639-3:{},language name:{},"
              "sentence number:{}".format(lang, language_dict[lang]["ISO639-3"], language_dict[lang]["language name"],
                                          len(lang_identifier[lang])))


def cmp(item: str):
    _, file_name = os.path.split(item)
    file_name = file_name.split("_")
    return int(file_name[1])


def multilingual_bleu_calculation(args):

    assert args.lang_identifier_file_path is not None and args.language_data is not None

    bleu_score_type = args.bleu_score_type

    corpus_bleu = nltk_corpus_bleu if bleu_score_type == "nltk_bleu" else sacre_corpus_bleu

    lang_identifier_list = read_data(args.lang_identifier_file_path)
    lang_identifier = {}

    with open(args.language_data) as f:
        language_dict = json.load(f)

    for i, sentence in enumerate(lang_identifier_list):

        lang_code = sentence

        assert lang_code.startswith("<") and lang_code.endswith(">")

        lang_code = lang_code[1:-1]

        if lang_code not in lang_identifier:
            lang_identifier[lang_code] = [i]
        else:
            lang_identifier[lang_code].append(i)

    show_corpus_statistics(lang_identifier, language_dict)

    reference_data = read_data(args.reference_path)

    if bleu_score_type == "nltk_bleu":
        reference_data = [[sentence.split()] for sentence in reference_data]

    reference_data_per_language = {}

    for k, v in lang_identifier.items():

        reference_data_per_language[k] = [reference_data[line_number] for line_number in v]

        if bleu_score_type == "sacrebleu":
            reference_data_per_language[k] = [reference_data_per_language[k]]

    args.translation_path_list.sort(key=cmp)

    bleu_score_dict = {lang: [] for lang in lang_identifier}

    for translation_path in args.translation_path_list:

        print("Translation in: {}\n".format(translation_path))

        translation_data = read_data(translation_path)

        if bleu_score_type == "nltk_bleu":
            translation_data = [sentence.split() for sentence in translation_data]

        translation_data_per_language = {}
        for k, v in lang_identifier.items():
            translation_data_per_language[k] = [translation_data[line_number] for line_number in v]

        for lang in lang_identifier:

            if bleu_score_type == "nltk_bleu":
                bleu_score = corpus_bleu(reference_data_per_language[lang], translation_data_per_language[lang]) * 100
            else:
                bleu_score = corpus_bleu(translation_data_per_language[lang], reference_data_per_language[lang])

            print("IOS639-2:{},IOS639-3:{},language name:{},bleu:{}".format(lang, language_dict[lang]["ISO639-3"],
                                                                            language_dict[lang]["language name"],
                                                                            bleu_score))
            bleu_score_dict[lang].append(bleu_score if bleu_score_type == "nltk_bleu" else bleu_score.score)
        print()

    print("Writing data to: {}".format(args.bleu_score_data_path))

    with open(args.bleu_score_data_path, "w") as f:
        json.dump(bleu_score_dict, f)


def bilingual_bleu_calculation(args):

    bleu_score_type = args.bleu_score_type

    corpus_bleu = nltk_corpus_bleu if bleu_score_type == "nltk_bleu" else sacre_corpus_bleu

    reference_data = read_data(args.reference_path)

    if bleu_score_type == "nltk_bleu":
        reference_data = [[sentence.split()] for sentence in reference_data]
    else:
        reference_data = [reference_data]

    args.translation_path_list.sort(key=cmp)

    blue_score_data = []

    for translation_path in args.translation_path_list:

        print("Translation in: {}\n".format(translation_path))

        translation_data = read_data(translation_path)

        if bleu_score_type == "nltk_bleu":
            translation_data = [sentence.split() for sentence in translation_data]
            bleu_score = corpus_bleu(reference_data, translation_data) * 100

        else:
            bleu_score = corpus_bleu(translation_data, reference_data)

        blue_score_data.append(bleu_score)

        print("bleu:{}".format(bleu_score))

    print("Writing data to: {}".format(args.bleu_score_data_path))

    with open(args.bleu_score_data_path) as f:
        json.dump(blue_score_data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_identifier_file_path",
                        help="The first token in sentences of src file is language identifier")
    parser.add_argument("--translation_path_list", nargs="+")
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--language_data")

    parser.add_argument("--bleu_score_data_path", required=True)
    parser.add_argument("--multilingual", action="store_true")

    parser.add_argument("--bleu_score_type", required=True, choices=["nltk_bleu", "sacrebleu"])

    args, unknown = parser.parse_known_args()

    if args.multilingual:
        multilingual_bleu_calculation(args)
    else:
        bilingual_bleu_calculation(args)


if __name__ == "__main__":
    main()
