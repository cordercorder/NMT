import subprocess
import os

num_operations = 32000

prefix = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data"

learn_bpe_file = ["train_src_combine.txt", "train_tgt_en.txt"]

learn_bpe_file_path = [os.path.join(prefix, file) for file in learn_bpe_file]

codes_file_path = os.path.join(prefix, "codes_file_bpe_" + str(num_operations))
src_vocab_path = os.path.join(prefix, "src_vocab_bpe_" + str(num_operations))
tgt_vocab_path = os.path.join(prefix, "tgt_vocab_bpe_" + str(num_operations))

apply_bpe_file = ["train_src_combine.txt", "dev_src_combine.txt", "test_src_combine.txt",
                  "train_tgt_en.txt", "dev_tgt_en.txt", "test_tgt_en.txt"]


command_learn_bpe = "subword-nmt learn-joint-bpe-and-vocab --input {} {} -s {} -o {} --write-vocabulary {} {}".format(
                    learn_bpe_file_path[0], learn_bpe_file_path[1], num_operations, codes_file_path, src_vocab_path,
                    tgt_vocab_path)

subprocess.call(command_learn_bpe, shell=True)
print(command_learn_bpe)


for f in apply_bpe_file:

    p = os.path.join(prefix, f)

    idx = f.find(".")
    
    file_bpe_name = f[:idx] + "_bpe_" + str(num_operations) + ".txt"
    file_bpe_path = os.path.join(prefix, file_bpe_name)

    if "src" in f:
        command_apply_bpe = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 0 < {} > {}".format(
                            codes_file_path, src_vocab_path, p, file_bpe_path)

    else:
        command_apply_bpe = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 0 < {} > {}".format(
                            codes_file_path, tgt_vocab_path, p, file_bpe_path)

    subprocess.call(command_apply_bpe, shell=True)
    print(command_apply_bpe)
