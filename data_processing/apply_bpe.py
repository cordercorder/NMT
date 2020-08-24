import os
import argparse
from subprocess import call


class JointBPE:

    def __init__(self, num_operations: int, train_file_src_path: str, train_file_tgt_path: str,
                 vocabulary_threshold: int, directory_path: str):

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
                 directory_path: str):

        self.num_operations = num_operations
        self.train_file_src_path = train_file_src_path
        self.train_file_tgt_path = train_file_tgt_path
        self.directory_path = directory_path

        self.codes_file_src_path = None
        self.codes_file_tgt_path = None

    def learn_bpe(self):

        print("Learn BPE")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_src_path", required=True)
    parser.add_argument("--train_file_tgt_path", required=True)
    parser.add_argument("--num_operations", required=True, type=int)
    parser.add_argument("--vocabulary_threshold", required=True, type=int)

    parser.add_argument("--file_src_path_list", nargs="+")
    parser.add_argument("--file_tgt_path_list", nargs="+")

    parser.add_argument("--output_directory", required=True)

    args, unknown = parser.parse_known_args()

    bpe = JointBPE(args.num_operations, args.train_file_src_path, args.train_file_tgt_path,
                   args.vocabulary_threshold, args.output_directory)

    bpe.learn_joint_bpe()
    bpe.apply_joint_bpe(args.file_src_path_list, args.file_tgt_path_list)


if __name__ == '__main__':
    main()
