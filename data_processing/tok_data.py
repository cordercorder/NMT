import sys
sys.path.append("../")

from utils.tools import read_data, write_data, normalizeString
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", nargs="+")
parser.add_argument("--remove_punctuation", action="store_true", default=False)
parser.add_argument("--to_ascii", action="store_true", default=False)

args, unknown = parser.parse_known_args()

print("remove_punctuation: {}, to_ascii: {}".format(args.remove_punctuation, args.to_ascii))

for data_path in args.data_path:

    data = read_data(data_path)

    data = [normalizeString(line, args.remove_punctuation, args.to_ascii) for line in data]

    idx = data_path.rfind(".")
    tok_data_path = data_path[:idx] + "_tok" + data_path[idx:]

    print("{} > {}".format(data_path, tok_data_path))

    write_data(data, tok_data_path)
