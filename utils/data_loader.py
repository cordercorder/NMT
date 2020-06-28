from .process import normalizeString
from .Vocab import Vocab
import torch


def load_corpus_data(data_path, language_name, start_token, end_token, vocab_path, unk="UNK", threshold=0):

    v = Vocab(language_name, start_token, end_token, threshold)

    corpus = []

    with open(data_path) as f:

        data = f.read().split("\n")

        for line in data:
            line = " ".join([start_token, normalizeString(line, to_ascii=False), end_token])
            v.add_sentence(line, normalize=False)
            corpus.append(line)

    data2index = []

    v.add_unk(unk)

    for line in corpus:

        data2index.append([v.get_index(token) for token in line.split()])

    v.save(vocab_path)

    return data2index, v


def pad_data(data, padding_value, device):

    data_tensor = []

    max_length = max(len(line) for line in data)

    for i in range(max_length):
        data_tensor.append([line[i] if i < len(line) else padding_value for line in data])

    return torch.tensor(data_tensor).to(device)


def batch_data(src_data, tgt_data, train_order, batch_size, padding_value, device):

    for order in train_order:

        tmp_src_data = src_data[order:order+batch_size]
        tmp_tgt_data = tgt_data[order:order+batch_size]

        yield pad_data(tmp_src_data, padding_value, device), pad_data(tmp_tgt_data, padding_value, device)


if __name__ == "__main__":
    import sys
    print(sys.path)
    data_, v = load_corpus_data("D:/jinrenren/NLP_study/codes/NMT/data/test_input.en", "en", "<s>", "<e>",
                                "D:/jinrenren/NLP_study/codes/NMT/data/vocab.en")
    print(data_)
