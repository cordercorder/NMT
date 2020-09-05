import argparse

from utils.tools import read_data, write_data, sort_src_sentence_by_length


def sort_sentence(args):
    for src_file_path, tgt_file_path, sorted_src_file_path, sorted_tgt_file_path in zip(args.src_file_path_list,
                                                                                        args.tgt_file_path_list,
                                                                                        args.output_src_file_path_list,
                                                                                        args.output_tgt_file_path_list):
        src_data = read_data(src_file_path)
        tgt_data = read_data(tgt_file_path)

        assert len(src_data) == len(tgt_data)

        src_data = [sentence.split() for sentence in src_data]
        tgt_data = [sentence.split() for sentence in tgt_data]

        src_data, tgt_data = sort_src_sentence_by_length(list(zip(src_data, tgt_data)))

        write_data(src_data, sorted_src_file_path)
        write_data(tgt_data, sorted_tgt_file_path)


def remove_long_sentence(args):

    assert args.max_sentence_length is not None

    for src_file_path, tgt_file_path, filtered_src_file_path, filtered_tgt_file_path in zip(args.src_file_path_list,
                                                                                            args.tgt_file_path_list,
                                                                                            args.output_src_file_path_list,
                                                                                            args.output_tgt_file_path_list):

        src_data = read_data(src_file_path)
        tgt_data = read_data(tgt_file_path)

        assert len(src_data) == len(tgt_data)

        src_sentence_filtered = []
        tgt_sentence_filtered = []

        for src_sentence, tgt_sentence in zip(src_data, tgt_data):

            if len(src_sentence.split()) <= args.max_sentence_length and \
                    len(tgt_sentence.split()) <= args.max_sentence_length:
                src_sentence_filtered.append(src_sentence)
                tgt_sentence_filtered.append(tgt_sentence)

        write_data(src_sentence_filtered, filtered_src_file_path)
        write_data(tgt_sentence_filtered, filtered_tgt_file_path)


def remove_same_sentence(args):

    if args.src_memory_path:
        src_memory_list = []

        for memory_path in args.src_memory_path:
            src_memory_list.extend(read_data(memory_path))

        src_memory = set(src_memory_list)

    else:
        src_memory = set()

    for src_file_path, tgt_file_path, filtered_src_file_path, filtered_tgt_file_path in zip(args.src_file_path_list,
                                                                                            args.tgt_file_path_list,
                                                                                            args.output_src_file_path_list,
                                                                                            args.output_tgt_file_path_list):

        src_data = read_data(src_file_path)
        tgt_data = read_data(tgt_file_path)

        assert len(src_data) == len(tgt_data)

        src_sentence_filtered = []
        tgt_sentence_filtered = []

        sentence_visited = set()
        removed_sentence_id = set()

        for i, sentence in enumerate(src_data):

            if sentence in sentence_visited:
                removed_sentence_id.add(i)
            elif sentence in src_memory:
                removed_sentence_id.add(i)
                sentence_visited.add(sentence)
            else:
                sentence_visited.add(sentence)

        for i, (src_sentence, tgt_sentence) in enumerate(zip(src_data, tgt_data)):

            if i not in removed_sentence_id:
                src_sentence_filtered.append(src_sentence)
                tgt_sentence_filtered.append(tgt_sentence)

        write_data(src_sentence_filtered, filtered_src_file_path)
        write_data(tgt_sentence_filtered, filtered_tgt_file_path)


def main():
    parser = argparse.ArgumentParser()

    # src_file_path_list and tgt_file_path_list must be one-to-one match

    parser.add_argument("--src_file_path_list", nargs="+")
    parser.add_argument("--tgt_file_path_list", nargs="+")

    parser.add_argument("--output_src_file_path_list", nargs="+")
    parser.add_argument("--output_tgt_file_path_list", nargs="+")

    parser.add_argument("--operation", required=True, choices=["sort_sentence", "remove_long_sentence",
                                                               "remove_same_sentence"])
    parser.add_argument("--max_sentence_length", type=int)

    parser.add_argument("--src_memory_path",  nargs="+", help="the set of sentences that need to be removed")

    args, unknown = parser.parse_known_args()

    assert len(args.src_file_path_list) == len(args.tgt_file_path_list) == len(args.output_src_file_path_list) == \
           len(args.output_tgt_file_path_list)

    if args.operation == "remove_long_sentence":
        remove_long_sentence(args)

    elif args.operation == "sort_sentence":
        sort_sentence(args)

    else:
        remove_same_sentence(args)


if __name__ == "__main__":
    main()
