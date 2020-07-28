import glob
import os
from subprocess import call

source_dir = "/data/rrjin/NMT/data/bible_data/models/attention_model"
prefix = "attention_multi_gpu_lstm*"

for file in glob.glob(os.path.join(source_dir, prefix)):

    file_dir, file_name = os.path.split(file)
    file_name = file_name + "_old"

    new_file = os.path.join(file_dir, file_name)

    command = "mv {} {}".format(file, new_file)
    print(command)
    call(command, shell=True)

