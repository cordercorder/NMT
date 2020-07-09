import os

src_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/merge_bpe_data_src_combine.txt"
tgt_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/merge_bpe_data_tgt_en.txt"

output_dir = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data"

def read_data(data_path):
    with open(data_path) as f:
        data = f.read().split("\n")
        data = [line.split() for line in data]
        return data

def write_data(data, data_path):

    if isinstance(data[0], list):
        data = [" ".join(line) for line in data]

    with open(data_path, "w") as f:

        f.write("\n".join(data))

raw_src_data = read_data(src_path)
raw_tgt_data = read_data(tgt_path)

assert len(raw_src_data) == len(raw_tgt_data)

threshold = 500

src_data = []
tgt_data = []

for i in range(len(raw_src_data)):

    if len(raw_src_data[i]) > threshold:
        continue

    src_data.append(raw_src_data[i])
    tgt_data.append(raw_tgt_data[i])

write_data(src_data, os.path.join(output_dir, "filter_merge_bpe_data_src_combine.txt"))
write_data(tgt_data, os.path.join(output_dir, "filter_merge_bpe_data_tgt_en.txt"))
