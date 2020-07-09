import os

def read_data(data_path):
    with open(data_path) as f:
        data = f.read().split("\n")
        return data

def write_data(data_path, data):
    with open(data_path, "w") as f:
        f.write("\n".join(data))

src_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/filter_merge_bible_data_src_combine.txt"
tgt_path = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/filter_merge_bible_data_tgt_en.txt"

output_dir = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data"

src_data = read_data(src_path)
tgt_data = read_data(tgt_path)

dev_sentence_num = 10000
test_sentence_num = 10000

cnt = 0

assert len(src_data) == len(tgt_data)

combine_all_data = list(zip(src_data, tgt_data))

dev_src = []
dev_tgt = []

for i in range(dev_sentence_num):

    src_line, tgt_line = combine_all_data[- (i + 1)]
    dev_src.append(src_line)
    dev_tgt.append(tgt_line)

test_src = []
test_tgt = []

for i in range(dev_sentence_num, dev_sentence_num + test_sentence_num):

    src_line, tgt_line = combine_all_data[- (i + 1)]
    test_src.append(src_line)
    test_tgt.append(tgt_line)

train_src = src_data[: - (dev_sentence_num + test_sentence_num)]
train_tgt = tgt_data[: - (dev_sentence_num + test_sentence_num)]

write_data(os.path.join(output_dir, "train_src_combine.txt"), train_src)
write_data(os.path.join(output_dir, "train_tgt_en.txt"), train_tgt)

write_data(os.path.join(output_dir, "dev_src_combine.txt"), dev_src)
write_data(os.path.join(output_dir, "dev_tgt_en.txt"), dev_tgt)

write_data(os.path.join(output_dir, "test_src_combine.txt"), test_src)
write_data(os.path.join(output_dir, "test_tgt_en.txt"), test_tgt)

print(len(train_src))