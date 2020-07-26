import argparse
import json
from utils.tools import read_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file_path", "-lf", required=True)
    parser.add_argument("--json_file_path", "-jf", required=True)

    args, unknown = parser.parse_known_args()

    log_data = read_data(args.log_file_path)

    data = []

    element = {}

    for line in log_data:

        line = line.lower()

        if line.startswith("load model"):
            element["model_name"] = line[16:]

        elif line.startswith("bleu"):

            l = line.find("=")
            assert l != -1
            l += 2

            r = line.find(",")
            assert r != -1

            bleu_score = float(line[l:r])
            element["bleu"] = bleu_score
            element["bleu_details"] = line

            data.append(element)

            element = {}

    data.sort(key=lambda item: -item["bleu"])

    with open(args.json_file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
