import numpy as np
import argparse
import pycountry
import lang2vec.lang2vec as l2v
from sklearn import preprocessing, linear_model, svm
from lang_vec.lang_vec_tools import load_lang_vec
from utils.tools import read_data
from typing import Dict


def get_language_alpha3(language_code: str):
    if len(language_code) == 2:
        ans = pycountry.languages.get(alpha_2 = language_code)
    elif len(language_code) == 3:
        ans = pycountry.languages.get(alpha_3 = language_code)
    else:
        return "-1"
    if ans is not None:
        return ans.alpha_3
    else:
        return "unknown language"


def check_alpha3(alpha3: str):
    if alpha3 != "unknown language" and alpha3 in l2v.LANGUAGES:
        return True
    return False


def train(args: argparse.Namespace, lang_vec: Dict, lang_alpha3: Dict, features: Dict):

    print("Classify method: {}".format(args.classify_method))

    # fix order
    lang_alpha3 = {k: lang_alpha3[k] for k in sorted(lang_alpha3.keys())}

    X = [lang_vec[lang] for lang in lang_alpha3.keys()]
    train_data_rate = 0.7

    score_dict = {}

    f = open(args.output_file_path, "w")

    for feat in range(len(features["CODE"])):

        Y = [features[lang][feat] if features[lang][feat] != "--" else -1 for lang in lang_alpha3.values()]

        idx = [i for i in range(len(Y)) if Y[i] != -1]

        train_set = np.array([[X[i], Y[i]] for i in idx])

        if len(train_set) == 0:
            print("Feature {} is not available in all 101 languages!".format(features["CODE"][feat]))
            f.write("Feature {} is not available in all 101 languages!\n".format(features["CODE"][feat]))
            continue

        lab_enc = preprocessing.LabelEncoder()
        train_set[:, 1] = lab_enc.fit_transform(train_set[:, 1])

        x_train = train_set[:int(len(train_set) * train_data_rate), 0]
        y_train = train_set[:int(len(train_set) * train_data_rate), 1]

        x_test = train_set[int(len(train_set) * train_data_rate):, 0]
        y_test = train_set[int(len(train_set) * train_data_rate):, 1]

        if len(x_train) == 0:
            print("Feature {} has no train data!".format(features["CODE"][feat]))
            f.write("Feature {} has no train data!\n".format(features["CODE"][feat]))
            continue

        if len(x_test) == 0:
            print("Feature {} has no test data!".format(features["CODE"][feat]))
            f.write("Feature {} has no test data!\n".format(features["CODE"][feat]))
            continue

        if np.all(y_train == y_train[0]):
            print("Feature {} has only one class!".format(features["CODE"][feat]))
            f.write("Feature {} has only one class!\n".format(features["CODE"][feat]))
            continue

        if args.classify_method == "logistic":
            logistic_model = linear_model.LogisticRegression(max_iter=3000)
            clf = logistic_model.fit(x_train.tolist(), y_train.tolist())
        else:
            svm_model = svm.SVC()
            clf = svm_model.fit(x_train.tolist(), y_train.tolist())

        score = clf.score(x_test.tolist(), y_test.tolist())
        score_dict[features["CODE"][feat]] = score
        print("Feature {} accuracy is {}, train dataset has {} element, test dataset has {} element".format(
              features["CODE"][feat], score, len(x_train), len(x_test)))
        f.write("Feature {} accuracy is {}, train dataset has {} element, test dataset has {} element\n".format(
                features["CODE"][feat], score, len(x_train), len(x_test)))

    f.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_name", required=True)
    parser.add_argument("--classify_method", required=True, choices=["svm", "logistic"])
    parser.add_argument("--lang_vec_path", required=True)
    parser.add_argument("--lang_name_path", required=True)
    parser.add_argument("--output_file_path", required=True)

    args, unknown = parser.parse_known_args()

    lang_name = read_data(args.lang_name_path)
    lang_vec = load_lang_vec(args.lang_vec_path)

    lang_alpha3 = {}

    for lang in lang_name:
        alpha3 = get_language_alpha3(lang[1:-1])
        if check_alpha3(alpha3):
            lang_alpha3[lang] = alpha3

    feature_name = args.feature_name
    features = l2v.get_features(list(lang_alpha3.values()), feature_name, header=True)

    train(args, lang_vec, lang_alpha3, features)


if __name__ == "__main__":
    main()
