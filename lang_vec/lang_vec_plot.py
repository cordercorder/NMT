import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold

from lang_vec.lang_vec_tools import load_lang_vec

plt.switch_backend("agg")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_vec_path", required=True)
    parser.add_argument("--picture_path", required=True)

    args, unknown = parser.parse_known_args()

    lang_vec_dict = load_lang_vec(args.lang_vec_path)

    X = np.array(list(lang_vec_dict.values()))

    labels = [lang_code[1:-1] for lang_code in lang_vec_dict.keys()]

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=123)
    X_tsne = tsne.fit_transform(X)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 1)

    axes.scatter(X_tsne[:, 0], X_tsne[:, 1])

    assert len(labels) == X_tsne.shape[0]

    for i in range(X_tsne.shape[0]):
        axes.text(X_tsne[i][0], X_tsne[i][1], labels[i])

    plt.savefig(args.picture_path)
    plt.close()


if __name__ == "__main__":
    main()
