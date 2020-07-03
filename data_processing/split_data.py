import random
import os

def read_data(data_path):

    with open(data_path) as f:

        data = f.read().split("\n")
        return data

def write_data(data_path, data):

    with open(data_path, "w") as f:

        f.write("\n".join(data))

src_path = "/data/rrjin/NMT/tmpdata/src.spa"
tgt_path = "/data/rrjin/NMT/tmpdata/tgt.en"

output_dir = "/data/rrjin/NMT/tmpdata"

src_data = read_data(src_path)
tgt_data = read_data(tgt_path)

test_sentence_num = 2000

cnt = 0

assert len(src_data) == len(tgt_data)

is_select = set()

test_src = []
test_tgt = []

while cnt < test_sentence_num:

    idx = random.randint(0, len(src_data) - 1)

    if idx in is_select:
        continue

    test_src.append(src_data[idx])
    test_tgt.append(tgt_data[idx])
    is_select.add(idx)
    cnt += 1

train_src = []
train_tgt = []

for i in range(len(src_data)):

    if i not in is_select:

        train_src.append(src_data[i])
        train_tgt.append(tgt_data[i])

write_data(os.path.join(output_dir, "train_src.spa"), train_src)
write_data(os.path.join(output_dir, "train_tgt.en"), train_tgt)

write_data(os.path.join(output_dir, "test_src.spa"), test_src)
write_data(os.path.join(output_dir, "test_tgt.en"), test_tgt)
