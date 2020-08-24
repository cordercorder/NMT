import os
from utils.tools import read_data, write_data

output_dir = "/data/rrjin/corpus_data/ted_data/zh_en"

prefix = "/data/rrjin/corpus_data"

corpus_name = ["zh_en", "zh-tw_en"]

train_src = []
train_tgt = []

dev_src = []
dev_tgt = []

test_src = []
test_tgt = []

for directory in corpus_name:

    directory = os.path.join(prefix, directory)

    for file in os.listdir(directory):

        file_path = os.path.join(directory, file)
        data = read_data(file_path)

        if file.startswith("train"):

            if file.endswith(".en"):
                train_tgt.extend(data)
            else:
                train_src.extend(data)

        elif file.startswith("dev"):

            if file.endswith(".en"):
                dev_tgt.extend(data)
            else:
                dev_src.extend(data)

        elif file.startswith("test"):

            if file.endswith(".en"):
                test_tgt.extend(data)
            else:
                test_src.extend(data)

        else:
            raise Exception("Error!")

assert len(train_src) == len(train_tgt)
assert len(dev_src) == len(dev_tgt)
assert len(test_src) == len(test_tgt)


def remove_blank(src_data):

    src_data = ["".join(sentence.strip().split()) for sentence in src_data]
    return src_data


train_src = remove_blank(train_src)
dev_src = remove_blank(dev_src)
test_src = remove_blank(test_src)

write_data(train_src, os.path.join(output_dir, "train.zh"))
write_data(train_tgt, os.path.join(output_dir, "train.en"))

write_data(dev_src, os.path.join(output_dir, "dev.zh"))
write_data(dev_tgt, os.path.join(output_dir, "dev.en"))

write_data(test_src, os.path.join(output_dir, "test.zh"))
write_data(test_tgt, os.path.join(output_dir, "test.en"))
