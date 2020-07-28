import os
import glob
import random


source_dir = "/data/rrjin/corpus_data/ted_data"
output_dir = "/data/rrjin/NMT/data/ted_data/corpus"

src_data_prefix = "src_combine_"
tgt_data_prefix = "tgt_en_"


def read_data(p, token=None):

    with open(p) as f:
        
        data = f.read().split("\n")
        
        if token:
            data = [token + " " + line for line in data]
        
        return data


def save_data(p, data):

    with open(p, "w") as f:
        f.write("\n".join(data))


def create(corpus_type):

    src_data = []
    tgt_data = []

    for d in os.listdir(source_dir):

        corpus_dir = os.path.join(source_dir, d)
        pattern = os.path.join(corpus_dir, corpus_type + "*")

        for file in glob.glob(pattern):

            _, file_name = os.path.split(file)

            idx = file_name.find(".") + 1

            token = "".join(["<", file_name[idx:], ">"])

            if file.endswith(".en"):
                tgt_data.extend(read_data(file))
            
            else:
                src_data.extend(read_data(file, token))

    assert len(src_data) == len(tgt_data)

    print(len(src_data))

    all_data = list(zip(src_data, tgt_data))

    for i in range(3):
        random.shuffle(all_data)

    src_data = []
    tgt_data = []

    for src_line, tgt_line in all_data:
        src_data.append(src_line)
        tgt_data.append(tgt_line)

    save_path = os.path.join(output_dir, src_data_prefix + corpus_type + ".txt")
    save_data(save_path, src_data)

    save_path = os.path.join(output_dir, tgt_data_prefix + corpus_type + ".txt")
    save_data(save_path, tgt_data)


create("train") # 4669326
create("test") # 151890
create("dev") # 121756