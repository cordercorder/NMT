import json
import argparse
import matplotlib.pyplot as plt
import math

from typing import Dict

plt.switch_backend("agg")


def plot_bleu_score_data(bleu_score_dict: Dict, language_dict: Dict, picture_path: str):

    fig_num_per_picture = 6
    lang_list = list(bleu_score_dict.keys())

    num_picture = int(math.ceil(len(lang_list) / fig_num_per_picture))

    cur = 0

    fontdict = {"family": "Times New Roman", "size": 18}

    for i in range(num_picture):

        fig, axes = plt.subplots(3, 2, figsize=(50, 30))

        for row in range(3):
            for col in range(2):
                epoch_list = bleu_score_dict[lang_list[cur]]
                axes[row, col].plot(list(range(len(epoch_list))), epoch_list, color="#DE6B58", marker="x",
                                    linestyle="-", linewidth=2, label="BLEU point")
                axes[row, col].set_xlabel("epoch", fontdict=fontdict)
                axes[row, col].set_ylabel(language_dict[lang_list[cur]]["language name"], fontdict=fontdict)
                axes[row, col].set_xticks(list(range(len(epoch_list))))
                axes[row, col].grid(which="major", axis="y", linewidth=0.5)
                axes[row, col].legend(loc="best")

                for j in range(len(epoch_list)):
                    axes[row, col].text(j, epoch_list[j], "{:.2f}".format(epoch_list[j]))

                cur += 1
                if cur == len(lang_list):
                    break

            if cur == len(lang_list):
                break

        plt.savefig("{}/{}.jpg".format(picture_path, i), dpi=200)
        plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bleu_score_dict_path", required=True)
    parser.add_argument("--language_data", required=True)
    parser.add_argument("--picture_path", required=True)

    args, unknown = parser.parse_known_args()

    with open(args.bleu_score_dict_path) as f:
        bleu_score_dict = json.load(f)

    with open(args.language_data) as f:
        language_dict = json.load(f)

    plot_bleu_score_data(bleu_score_dict, language_dict, args.picture_path)


if __name__ == "__main__":
    main()
