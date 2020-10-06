import os
import argparse

from utils.tools import read_data, write_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu_translation_dir", required=True)
    parser.add_argument("--is_tok", action="store_true")
    parser.add_argument("--merged_translation_dir", required=True)

    args, unknown = parser.parse_known_args()

    translations_dict_per_model = {}

    for file in os.listdir(args.multi_gpu_translation_dir):

        file_name_prefix, extension = os.path.splitext(file)

        if args.is_tok and not file_name_prefix.endswith("_tok"):
            continue
        elif not args.is_tok and file_name_prefix.endswith("_tok"):
            continue

        assert extension[1:5] == "rank"
        rank = int(extension[5:])

        data = read_data(os.path.join(args.multi_gpu_translation_dir, file))

        if file_name_prefix in translations_dict_per_model:
            translations_dict_per_model[file_name_prefix].append((data, rank))
        else:
            translations_dict_per_model[file_name_prefix] = [(data, rank)]

    for file_name_prefix in translations_dict_per_model:
        translations_dict_per_model[file_name_prefix].sort(key=lambda item: item[1])

    if not os.path.isdir(args.merged_translation_dir):
        os.makedirs(args.merged_translation_dir)

    for file_name_prefix in translations_dict_per_model:

        merged_translations = []
        for translations, rank in translations_dict_per_model[file_name_prefix]:
            merged_translations.extend(translations)

        write_data(merged_translations, os.path.join(args.merged_translation_dir, "{}.txt".format(file_name_prefix)))


if __name__ == "__main__":
    main()
