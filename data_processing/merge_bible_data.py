import os
import random
from utils.process import normalizeString

source_dir = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/parallel_text"
output_dir = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data"

random.seed(998244353)

combine_all_data_src = []
combine_all_data_tgt = []

def read_data(data_path):
    with open(data_path) as f:
        return f.read().split("\n")

def write_data(data, data_name):
    with open(os.path.join(output_dir, data_name), "w") as f:
        f.write("\n".join(data))

for sub_dir in os.listdir(source_dir):

    sub_dir = os.path.join(source_dir, sub_dir)

    src_data = None
    tgt_data = None

    token = None

    for i, file in enumerate(os.listdir(sub_dir)):

        assert i < 2

        file_path = os.path.join(sub_dir, file)
        data = read_data(file_path)

        if file.endswith(".en"):
            tgt_data = data
        else:
            idx = file.rfind(".") + 1
            token = "<{}>".format(file[idx:])
            src_data = data

    assert token is not None
    assert src_data is not None
    assert tgt_data is not None

    assert len(src_data) == len(tgt_data)

    for src_line, tgt_line in zip(src_data, tgt_data):

        if len(src_line) == 0 or len(tgt_line) == 0:
            continue

        src_line = normalizeString(src_line, to_ascii=False)
        tgt_line = normalizeString(tgt_line, to_ascii=False)

        src_line = token + " " + src_line

        combine_all_data_src.append(src_line)
        combine_all_data_tgt.append(tgt_line)

combine_all_data = list(zip(combine_all_data_src, combine_all_data_tgt))

print(len(combine_all_data))

# shuffle two times
random.shuffle(combine_all_data)
random.shuffle(combine_all_data)

combine_all_data_src, combine_all_data_tgt = zip(*combine_all_data)

write_data(combine_all_data_src, "merge_bible_data_src_combine.txt")
write_data(combine_all_data_tgt, "merge_bible_data_tgt_en.txt")
