import pycountry
import argparse
import json

from collections import OrderedDict
from utils.tools import read_data


def test():

    for lang in pycountry.languages:
        assert len(lang.alpha_3) == 3
        try:
            assert len(lang.alpha_2) == 2
        except AttributeError:
            continue


def convert(language_code_path: str, output_file_path: str):

    language_code = read_data(language_code_path)

    language_dict = OrderedDict()

    for code in language_code:

        alpha = {}

        code = code[1:-1]

        if len(code) == 2:
            alpha["alpha_2"] = code
        elif len(code) == 3:
            alpha["alpha_3"] = code
        else:
            language_dict[code] = {"ISO639-3": "unknown", "language name": "unknown"}
            continue

        language_data = pycountry.languages.get(**alpha)

        if language_data is None:
            language_dict[code] = {"ISO639-3": "unknown", "language name": "unknown"}
        else:
            language_dict[code] = {"ISO639-3": language_data.alpha_3, "language name": language_data.name}

    with open(output_file_path, "w") as f:
        json.dump(language_dict, f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language_code_path", required=True)
    parser.add_argument("-o", "--output_file_path", required=True)

    args, unknown = parser.parse_known_args()

    convert(args.language_code_path, args.output_file_path)


if __name__ == "__main__":
    # test()
    main()
