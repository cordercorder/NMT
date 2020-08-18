import random
import argparse
from utils.tools import read_data, write_data


class ShuffleData:

    def __init__(self, src_file_path: str, tgt_file_path: str, seed: int):

        self.src_data = read_data(src_file_path)
        self.tgt_data = read_data(tgt_file_path)

        self.seed = seed

        assert len(self.src_data) == len(self.tgt_data)

        self.shuffled_src_data = []
        self.shuffled_tgt_data = []

    def shuffle(self):

        random.seed(self.seed)
        combined_data = list(zip(self.src_data, self.tgt_data))
        random.shuffle(combined_data)

        self.shuffled_src_data, self.shuffled_tgt_data = list(zip(*combined_data))

    def write_shuffled_data(self, shuffled_src_file_path, shuffled_tgt_file_path):

        write_data(self.shuffled_src_data, shuffled_src_file_path)
        write_data(self.shuffled_tgt_data, shuffled_tgt_file_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--src_file_path", required=True)
    parser.add_argument("--tgt_file_path", required=True)

    parser.add_argument("--shuffled_src_file_path", required=True)
    parser.add_argument("--shuffled_tgt_file_path", required=True)

    args, unknown = parser.parse_known_args()

    data = ShuffleData(args.src_file_path, args.tgt_file_path, args.seed)

    data.shuffle()
    data.write_shuffled_data(args.shuffled_src_file_path, args.shuffled_tgt_file_path)


if __name__ == "__main__":
    main()
