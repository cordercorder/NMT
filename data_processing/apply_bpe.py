import os
from subprocess import call


class JointBPE:

    def __init__(self, num_operations: int, train_file_src_path: str, train_file_tgt_path: str,
                 vocabulary_threshold: int, directory_path: str = None):

        self.num_operations = num_operations
        self.train_file_src_path = train_file_src_path
        self.train_file_tgt_path = train_file_tgt_path
        self.vocabulary_threshold = vocabulary_threshold
        self.directory_path = directory_path

        self.codes_file_path = None
        self.src_vocab_path = None
        self.tgt_vocab_path = None

    def learn_joint_bpe(self):

        print("Learn Joint BPE")

        if self.directory_path is None:
            self.directory_path, _ = os.path.split(self.train_file_src_path)

        self.codes_file_path = os.path.join(self.directory_path, "codes_file_joint_bpe_" + str(self.num_operations))
        self.src_vocab_path = os.path.join(self.directory_path, "src_vocab_joint_bpe_" + str(self.num_operations))
        self.tgt_vocab_path = os.path.join(self.directory_path, "tgt_vocab_joint_bpe_" + str(self.num_operations))

        learn_joint_bpe_command = "subword-nmt learn-joint-bpe-and-vocab --input {} {} -s {} -o {} " \
                                  "--write-vocabulary {} {}".format(self.train_file_src_path, self.train_file_tgt_path,
                                                                    self.num_operations, self.codes_file_path,
                                                                    self.src_vocab_path, self.tgt_vocab_path)
        call(learn_joint_bpe_command, shell=True)
        print(learn_joint_bpe_command)

    def apply_joint_bpe(self, file_src_path_list: list, file_tgt_path_list: list):

        print("Apply Joint BPE")

        assert self.directory_path is not None

        assert len(file_src_path_list) == len(file_tgt_path_list)

        for i, file in enumerate(file_src_path_list + file_tgt_path_list):

            file_dir, file_name = os.path.split(file)

            idx = file_name.rfind(".")

            if idx == -1:
                file_bpe_name = file_name[:idx] + "_joint_bpe_" + str(self.num_operations) + ".txt"
            else:
                file_bpe_name = file_name[:idx] + "_joint_bpe_" + str(self.num_operations) + file_name[idx:]

            file_bpe_path = os.path.join(self.directory_path, file_bpe_name)

            if i < len(file_src_path_list):
                vocab_path = self.src_vocab_path
            else:
                vocab_path = self.tgt_vocab_path

            apply_joint_bpe_command = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold {} < " \
                                      "{} > {}".format(self.codes_file_path, vocab_path,
                                                       self.vocabulary_threshold, file, file_bpe_path)
            call(apply_joint_bpe_command, shell=True)
            print(apply_joint_bpe_command)


class IndividualBPE:

    def __init__(self, num_operations: int, train_file_src_path: str, train_file_tgt_path: str,
                 directory_path: str = None):

        self.num_operations = num_operations
        self.train_file_src_path = train_file_src_path
        self.train_file_tgt_path = train_file_tgt_path
        self.directory_path = directory_path

        self.codes_file_src_path = None
        self.codes_file_tgt_path = None

    def learn_bpe(self):

        print("Learn BPE")

        if self.directory_path is None:
            self.directory_path, _ = os.path.split(self.train_file_src_path)

        self.codes_file_src_path = os.path.join(self.directory_path, "codes_file_src_bpe_" + str(self.num_operations))
        self.codes_file_tgt_path = os.path.join(self.directory_path, "codes_file_tgt_bpe_" + str(self.num_operations))

        learn_src_bpe_command = "subword-nmt learn-bpe -s {} < {} > {}".format(self.num_operations,
                                                                               self.train_file_src_path,
                                                                               self.codes_file_src_path)
        call(learn_src_bpe_command, shell=True)

        print(learn_src_bpe_command)

        learn_tgt_bpe_command = "subword-nmt learn-bpe -s {} < {} > {}".format(self.num_operations,
                                                                               self.train_file_tgt_path,
                                                                               self.codes_file_tgt_path)
        call(learn_tgt_bpe_command, shell=True)
        print(learn_tgt_bpe_command)

    def apply_bpe(self, file_src_path_list: list, file_tgt_path_list: list):

        print("Apply BPE")

        assert self.directory_path is not None

        assert len(file_src_path_list) == len(file_tgt_path_list)

        for i, file in enumerate(file_src_path_list + file_tgt_path_list):

            file_dir, file_name = os.path.split(file)

            idx = file_name.rfind(".")

            if idx == -1:
                file_bpe_name = file_name[:idx] + "_bpe_" + str(self.num_operations) + ".txt"
            else:
                file_bpe_name = file_name[:idx] + "_bpe_" + str(self.num_operations) + file_name[idx:]

            file_bpe_path = os.path.join(self.directory_path, file_bpe_name)

            if i < len(file_src_path_list):
                codes_file_path = self.codes_file_src_path
            else:
                codes_file_path = self.codes_file_tgt_path

            apply_bpe_command = "subword-nmt apply-bpe -c {} < {} > {}".format(codes_file_path, file, file_bpe_path)
            call(apply_bpe_command, shell=True)
            print(apply_bpe_command)


def main():

    prefix = "/data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data"
    num_operations = 22000
    train_file_src_path = os.path.join(prefix, "train_src_combine.txt")
    train_file_tgt_path = os.path.join(prefix, "train_tgt_en.txt")

    file_src_list = ["train_src_combine.txt", "dev_src_combine.txt", "test_src_combine.txt"]
    file_tgt_list = ["train_tgt_en.txt", "dev_tgt_en.txt", "test_tgt_en.txt"]

    file_src_path_list = [os.path.join(prefix, file) for file in file_src_list]
    file_tgt_path_list = [os.path.join(prefix, file) for file in file_tgt_list]

    directory_path = "/data/rrjin/NMT/data/bible_data"

    bpe = JointBPE(num_operations, train_file_src_path, train_file_tgt_path, 0, directory_path)
    bpe.learn_joint_bpe()
    bpe.apply_joint_bpe(file_src_path_list, file_tgt_path_list)


if __name__ == '__main__':
    main()
