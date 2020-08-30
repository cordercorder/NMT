import json
import argparse
import matplotlib.pyplot as plt

from typing import Dict

from utils.tools import read_data

plt.switch_backend("agg")


def plot_data(sentence_num_per_language: Dict, language_dict: Dict, picture_path: str):
    
    lang_list = list(sentence_num_per_language.keys())
    sentence_num_list = list(num/1000 for num in sentence_num_per_language.values())

    x = list(range(len(lang_list)))

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 1, figsize=(15, 20))
    axes.set_yticks(x)

    axes.barh(x, sentence_num_list, label="number of sentences", color="#D2ACA3")
    axes.grid(linewidth=0.5, which="major", axis="x")
    axes.set_yticklabels([language_dict[lang_code]["language name"] for lang_code in lang_list])

    for i in range(len(lang_list)):
        axes.text(sentence_num_list[i], x[i], "{:.2f} k".format(sentence_num_list[i]))

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    axes.legend(loc='best')

    axes.set_title("Number of Sentences Per Language")

    plt.savefig(picture_path, dpi=600)
    plt.close()


def cmp(item):
    return -item[1]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--picture_path", required=True)
    parser.add_argument("--language_data", required=True)

    args, unknown = parser.parse_known_args()

    with open(args.language_data) as f:
        language_dict = json.load(f)

    data = read_data(args.data_path)

    sentence_num_per_language = {}

    for sentence in data:

        token_list = sentence.split()
        lang_code = token_list[0]

        assert lang_code.startswith("<") and lang_code.endswith(">")
        lang_code = lang_code[1:-1]

        sentence_num_per_language[lang_code] = sentence_num_per_language.get(lang_code, 0) + 1

    sentence_num_per_language = {k: v for k, v in sorted(sentence_num_per_language.items(), key=cmp)}

    plot_data(sentence_num_per_language, language_dict, args.picture_path)


if __name__ == "__main__":
    main()
