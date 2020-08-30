import json
import argparse
import matplotlib.pyplot as plt

from typing import Dict

from utils.tools import read_data

plt.switch_backend("agg")


def plot_data(sentence_num_per_language: Dict, picture_path: str):
    
    lang_list = list(sentence_num_per_language.keys())
    sentence_num_list = list(num/1000 for num in sentence_num_per_language.values())

    x = list(range(len(lang_list)))

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 1, figsize=(20, 5))
    axes.set_xticks(x)

    axes.bar(x, sentence_num_list, label="number of sentences", color="#D2ACA3", width=0.4)
    axes.grid(linewidth=0.5, which="major", axis='y')
    axes.set_xticklabels(lang_list)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    axes.legend(loc='best')

    axes.set_title("Number of Sentences Per Language")

    plt.savefig(picture_path, dpi=200)
    plt.close()


def cmp(item):
    return -item[1]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--picture_path", required=True)

    args, unknown = parser.parse_known_args()

    data = read_data(args.data_path)

    sentence_num_per_language = {}

    for sentence in data:

        token_list = sentence.split()
        lang_code = token_list[0]

        assert lang_code.startswith("<") and lang_code.endswith(">")
        lang_code = lang_code[1:-1]

        sentence_num_per_language[lang_code] = sentence_num_per_language.get(lang_code, 0) + 1

    sentence_num_per_language = {k: v for k, v in sorted(sentence_num_per_language.items(), key=cmp)}

    plot_data(sentence_num_per_language, args.picture_path)


if __name__ == "__main__":
    main()
