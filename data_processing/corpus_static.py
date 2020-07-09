from collections import defaultdict
import matplotlib.pyplot as plt

src_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/train_src_combine_bpe_32000.txt"
tgt_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/train_tgt_en_bpe_32000.txt"

def read_data(data_path):

    with open(data_path) as f:

        data = f.read().split("\n")
        data = ["".join(c for c in line if c != "@") for line in data]
        return data

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

def sort_sentence(data):

    data.sort(key=lambda item: -len(item))
    return data

def plot_freq(freq_data, fig_name):

    length, freq = list(zip(*freq_data))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(length, freq)

    plt.xlabel("frequency")
    plt.ylabel("length")
    plt.savefig("./" + fig_name)

src_data = sort_sentence(read_data(src_path))
tgt_data = sort_sentence(read_data(tgt_path))

print(src_data[:5])
print(tgt_data[:5])

src_freq = frequency(src_data)
tgt_freq = frequency(tgt_data)

src_freq = sort_freq(src_freq)
tgt_freq = sort_freq(tgt_freq)

plot_freq(src_freq, "statics of src3.jpg")
plot_freq(tgt_freq, "statics of tgt3.jpg")

sentence_number = len(src_data)

cnt = 0

for length, freq in src_freq:

    if length <= 500:
        cnt += 1

print(cnt)
print(sentence_number)
