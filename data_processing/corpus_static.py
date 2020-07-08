from utils.process import normalizeString
from collections import defaultdict

src_path = "/data/rrjin/corpus_data/ted_data/src_combine_train_bpe_20000.txt"
tgt_path = "/data/rrjin/corpus_data/ted_data/tgt_en_train_bpe_20000.txt"

def read_data(data_path):

    with open(data_path, "w") as f:

        data = f.read().split("\n")

        data = [normalizeString(line, to_ascii=False) for line in data]
        return data

def frequency(data):

    freq = defaultdict(lambda :0)

    for line in data:
        length = len(line.split())
        freq[length] = freq[length] + 1

    return freq

src_data = read_data(src_path)
tgt_data = read_data(tgt_path)

src_freq = frequency(src_data)
tgt_freq = frequency(tgt_data)

