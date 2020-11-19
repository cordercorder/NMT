import torch

from typing import List

from torch.utils.data import Dataset, DataLoader
from utils.Vocab import Vocab


class NMTDataset(Dataset):

    def __init__(self, src_data, tgt_data):
        self.data = list(zip(src_data, tgt_data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_corpus_data(data_path, language_name, start_token, end_token, mask_token, vocab_path, rebuild_vocab,
                     unk="UNK", threshold=0):
    if rebuild_vocab:
        v = Vocab(language_name, start_token, end_token, mask_token, threshold=threshold)

    corpus = []

    with open(data_path) as f:

        data = f.read().strip().split("\n")

        for line in data:
            line = line.strip()
            line = " ".join([start_token, line, end_token])

            if rebuild_vocab:
                v.add_sentence(line)

            corpus.append(line)

    data2index = []

    if rebuild_vocab:
        v.add_unk(unk)
        v.save(vocab_path)
    else:
        v = Vocab.load(vocab_path)

    for line in corpus:
        data2index.append([v.get_index(token) for token in line.split()])

    return data2index, v


def convert_data_to_index(data: List[str], vocab: Vocab):
    data2index = []

    for sentence in data:
        sentence = " ".join([vocab.start_token, sentence, vocab.end_token])
        data2index.append([vocab.get_index(token) for token in sentence.split()])

    return data2index


def pad_data(data, padding_value, batch_first):
    data_tensor = []

    max_length = max(len(line) for line in data)

    if batch_first:

        for line in data:
            data_tensor.append([line[i] if i < len(line) else padding_value for i in range(max_length)])
    else:

        for i in range(max_length):
            data_tensor.append([line[i] if i < len(line) else padding_value for line in data])

    return torch.tensor(data_tensor)


def collate(batch, padding_value, batch_first=False):
    src_batch, tgt_batch = zip(*batch)
    src_batch_tensor = pad_data(src_batch, padding_value, batch_first)
    tgt_batch_tensor = pad_data(tgt_batch, padding_value, batch_first)

    return ParallelSentenceBatch(src_batch_tensor, tgt_batch_tensor)


def collate_eval(batch, padding_value, batch_first):
    if isinstance(batch[0], tuple):
        src_batch, tgt_prefix_batch = zip(*batch)
    else:
        src_batch = batch
        tgt_prefix_batch = None
    src_batch_tensor = pad_data(src_batch, padding_value, batch_first)
    return SrcDataBatch(src_batch_tensor, tgt_prefix_batch)


class ParallelSentenceBatch:

    def __init__(self, src_batch_tensor, tgt_batch_tensor):
        self.src_batch_tensor = src_batch_tensor
        self.tgt_batch_tensor = tgt_batch_tensor

    def pin_memory(self):
        # custom memory pinning method on custom type
        self.src_batch_tensor = self.src_batch_tensor.pin_memory()
        self.tgt_batch_tensor = self.tgt_batch_tensor.pin_memory()
        return self.src_batch_tensor, self.tgt_batch_tensor


class SrcData(Dataset):

    def __init__(self, data: List[List[int]], tgt_prefix_data: List[str] = None):
        self.data = data
        self.tgt_prefix_data = tgt_prefix_data

    def __getitem__(self, item: int):

        if self.tgt_prefix_data is None:
            return self.data[item]
        else:
            return self.data[item], self.tgt_prefix_data[item]

    def __len__(self):
        return len(self.data)


class SrcDataBatch:

    def __init__(self, src_batch_tensor, tgt_prefix_batch):
        self.src_batch_tensor = src_batch_tensor
        self.tgt_prefix_batch = tgt_prefix_batch

    def pin_memory(self):
        # custom memory pinning method on custom type
        self.src_batch_tensor = self.src_batch_tensor.pin_memory()
        return self.src_batch_tensor, self.tgt_prefix_batch


if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    x = [[1, 2, 3], [4, 5], [7, 8, 9]]
    y = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

    train_data = NMTDataset(x, y)
    train_loader = DataLoader(train_data, 2, shuffle=True, collate_fn=lambda batch: collate(batch, 0))

    for s, t in train_loader:
        print(s)
        print(t)
        print("------")
