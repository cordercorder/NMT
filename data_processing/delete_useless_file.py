import os
import subprocess

source_dir = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/parallel_text"

for sub_dir in os.listdir(source_dir):

    sub_dir = os.path.join(source_dir, sub_dir)

    for file in os.listdir(sub_dir):

        if "tokenize" in file:

            file_path = os.path.join(sub_dir, file)
            subprocess.call("rm {}".format(file_path), shell=True)