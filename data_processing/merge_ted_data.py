import sys
import os
sys.path.append("../")
from utils.tools import read_data, write_data


def check(src_data: list, tgt_data: list):
    return len(src_data) == len(tgt_data)


def merge(corpus_path: str):

    train_data_src = []
    train_data_tgt = []

    dev_data_src = []
    dev_data_tgt = []

    test_data_src = []
    test_data_tgt = []

    un_uesd_corpus = {"pt-br_en", "fr-ca_en", "eo_en", "calv_en"}

    for corpus_dir in os.listdir(corpus_path):

        if corpus_dir in un_uesd_corpus:
            continue

        corpus_dir = os.path.join(corpus_path, corpus_dir)

        for corpus_file_name in os.listdir(corpus_dir):

            corpus_file_path = os.path.join(corpus_dir, corpus_file_name)

            data = read_data(corpus_file_path)

            is_en = True

            if not corpus_file_name.endswith(".en"):

                idx = corpus_file_name.rfind(".")
                assert idx != -1

                idx += 1
                lang_identify_token = "".join(["<", corpus_file_name[idx:], ">"])
                data = [" ".join([lang_identify_token, sentence]) for sentence in data]
                is_en = False

            if corpus_file_name.startswith("train"):

                if is_en:
                    train_data_tgt.extend(data)
                else:
                    train_data_src.extend(data)

            elif corpus_file_name.startswith("dev"):

                if is_en:
                    dev_data_tgt.extend(data)
                else:
                    dev_data_src.extend(data)

            elif corpus_file_name.startswith("test"):

                if is_en:
                    test_data_tgt.extend(data)
                else:
                    test_data_src.extend(data)

    assert check(train_data_src, train_data_tgt) and check(dev_data_src, dev_data_tgt) and \
           check(test_data_src, test_data_tgt)

    output_dir = "/data/rrjin/NMT/data/ted_data_new/corpus"

    write_data(train_data_src, os.path.join(output_dir, "raw_train_data_src.combine"))
    write_data(train_data_tgt, os.path.join(output_dir, "raw_train_data_tgt.en"))

    write_data(dev_data_src, os.path.join(output_dir, "raw_dev_data_src.combine"))
    write_data(dev_data_tgt, os.path.join(output_dir, "raw_dev_data_tgt.en"))

    write_data(test_data_src, os.path.join(output_dir, "raw_test_data_src.combine"))
    write_data(test_data_tgt, os.path.join(output_dir, "raw_test_data_tgt.en"))


def main():
    corpus_path = "/data/rrjin/corpus_data/ted_data"
    merge(corpus_path)


if __name__ == "__main__":
    main()
