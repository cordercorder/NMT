import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn import manifold
from collections import OrderedDict

from lang_vec.lang_vec_tools import load_lang_vec

plt.switch_backend("agg")


def plot_lang_vec(args):

    lang_vec_dict = load_lang_vec(args.lang_vec_path)

    with open(args.language_data) as f:
        language_data = json.load(f)

    with open(args.language_family_data) as f:
        language_family_data = json.load(f)

    labels_639_2 = [lang_code[1:-1] for lang_code in lang_vec_dict.keys()]
    labels_639_3 = [language_data[label]["ISO639-3"] for label in labels_639_2]

    labels_639_3_to_index = {lang_code: i for i, lang_code in enumerate(labels_639_3)}

    family_to_language = OrderedDict()

    for lang_code in language_family_data:

        family = language_family_data[lang_code][0]
        if family not in family_to_language:
            family_to_language[family] = [labels_639_3_to_index[lang_code]]
        else:
            family_to_language[family].append(labels_639_3_to_index[lang_code])

    X = np.array(list(lang_vec_dict.values()))

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=123)
    X_tsne = tsne.fit_transform(X)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 1)

    for family in family_to_language:
        if family == "unknown":
            for lang_id in family_to_language[family]:
                axes.scatter(X_tsne[lang_id, 0], X_tsne[lang_id, 1])
        else:
            language_id_set = family_to_language[family]
            axes.scatter(X_tsne[language_id_set, 0], X_tsne[language_id_set, 1])

    assert len(labels_639_3) == X_tsne.shape[0]

    for i in range(X_tsne.shape[0]):
        axes.text(X_tsne[i][0], X_tsne[i][1], labels_639_3[i])
    axes.set_title("Language Embedding Visualization")
    plt.savefig(args.picture_path, dpi=300)
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_vec_path", required=True)
    parser.add_argument("--picture_path", required=True)
    parser.add_argument("--language_family_data", required=True)
    parser.add_argument("--language_data", required=True)

    args, unknown = parser.parse_known_args()
    plot_lang_vec(args)


if __name__ == "__main__":
    main()
