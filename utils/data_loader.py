from utils.process import normalizeString
from utils.Vocab import Vocab
import torch
from torch.utils.data import Dataset, DataLoader


class NMTDataset(Dataset):

    def __init__(self, src_data, tgt_data):

        self.data = list(zip(src_data,tgt_data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def load_corpus_data(data_path, language_name, start_token, end_token, mask_token, vocab_path, rebuild_vocab,
                     unk="UNK", threshold=0, normalize=True):

    if rebuild_vocab:
        v = Vocab(language_name, start_token, end_token, mask_token, threshold=threshold)

    corpus = []

    with open(data_path) as f:

        data = f.read().split("\n")

        for line in data:

            if normalize:
                line = " ".join([start_token, normalizeString(line, to_ascii=False), end_token])
            else:
                line = "".join(c for c in line if c != "@")
                line = " ".join([start_token, line, end_token])

            if rebuild_vocab:
                v.add_sentence(line, normalize=False)

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


def pad_data(data, padding_value, device):

    data_tensor = []

    max_length = max(len(line) for line in data)

    for i in range(max_length):
        data_tensor.append([line[i] if i < len(line) else padding_value for line in data])

    return torch.tensor(data_tensor, device=device)


def collate(batch, padding_value, device):

    src_batch, tgt_batch = zip(*batch)
    src_batch_tensor = pad_data(src_batch, padding_value, device)
    tgt_batch_tensor = pad_data(tgt_batch, padding_value, device)

    return src_batch_tensor, tgt_batch_tensor

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    x = [[1, 2, 3], [4, 5], [7, 8, 9]]
    y = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

    train_data = NMTDataset(x, y)
    train_loader = DataLoader(train_data, 2, shuffle=True, collate_fn=lambda batch: collate(batch, 0, device))

    for s, t in train_loader:
        print(s)
        print(t)
        print("------")

