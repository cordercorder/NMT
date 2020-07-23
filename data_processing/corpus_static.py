import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import sys
sys.path.append("../")
from utils.tools import read_data


def frequency(data):

    freq = defaultdict(lambda :0)

    for line in data:
        length = len(line.split())
        freq[length] = freq[length] + 1

    return freq


def sort_freq(freq_data):

    freq_data = sorted(freq_data.items(), key=lambda item: -item[0])
    print(freq_data[:10])

    return freq_data


def plot_freq(freq_data, fig_name):

    length, freq = list(zip(*freq_data))

    fig = plt.figure()

    idx = fig_name.rfind(".") + 1
    name = fig_name[:idx]

    ax = fig.add_subplot(1, 1, 1)
    plt.title(name)
    ax.plot(length, freq, label=name)

    plt.xlabel("frequency")
    plt.ylabel("length")
    plt.grid()
    plt.legend()
    plt.savefig("./" + fig_name)
    plt.close()


def main():

    max_sentence_num = 100

    src_path = "/data/rrjin/NMT/data/bible_data/corpus/train_src_combine_joint_bpe_22000.txt"
    tgt_path = "/data/rrjin/NMT/data/bible_data/corpus/train_tgt_en_joint_bpe_22000.txt"

    src_data = read_data(src_path)
    tgt_data = read_data(tgt_path)

    src_freq = frequency(src_data)
    tgt_freq = frequency(tgt_data)

    src_freq = sort_freq(src_freq)
    tgt_freq = sort_freq(tgt_freq)

    _, src_file_name = os.path.split(src_path)
    _, tgt_file_name = os.path.split(tgt_path)

    plot_freq(src_freq, "statistic of " + src_file_name + ".jpg")
    plot_freq(tgt_freq, "statistic of " + tgt_file_name + ".jpg")

    sentence_number = len(src_data)

    cnt = 0

    for length, freq in src_freq:

        if length <= max_sentence_num:
            cnt += freq

    print(cnt)
    print(sentence_number)


if __name__ == "__main__":
    main()
