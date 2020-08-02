import torch
import torch.nn as nn
import argparse
from translation_memory.edit_distance import edit_distance
from utils.Vocab import Vocab
from utils.tools import load_transformer, read_data, write_data


@torch.no_grad()
def cosine_similarity(sentence_embedding: torch.tensor, memory: torch.tensor):

    """
    :param sentence_embedding: type: torch.FloatTensor. shape: (number_of_sentences1, embedding_size)
    :param memory: type: torch.FloatTensor. shape: (number_of_sentences2, embedding_size)
    :return: similarity, the indices of best match sentence
    """

    # inner_product: (number_of_sentences1, number_of_sentences2)
    inner_product = torch.matmul(sentence_embedding, memory.t())

    # sentence_embedding: (number_of_sentences1, 1)
    sentence_embedding = sentence_embedding.norm(dim=1)
    sentence_embedding = sentence_embedding.unsqueeze(1)

    # memory: (number_of_sentences1, )
    memory = memory.norm(dim=1)

    # inner_product: (number_of_sentences1, number_of_sentences2)
    inner_product = inner_product / (sentence_embedding * memory)

    # values: (number_of_sentences1, 1)
    # index: (number_of_sentences1, 1)
    values, index = inner_product.topk(1, dim=1)

    values = values.squeeze(1)
    index = index.squeeze(1)
    return values.tolist(), index.tolist()


@torch.no_grad()
def string2embedding(string: str, v: Vocab, token_embedding: nn.Embedding, device: torch.device):

    # string2index: (input_length, )
    string2index = torch.tensor([v.get_index(token) for token in string.strip().split()], device=device)

    # sentence_embedding: (input_length, embedding_size)
    sentence_embedding = token_embedding(string2index)

    # sentence_embedding: (embedding_size, )
    sentence_embedding = sentence_embedding.mean(dim=0)
    return sentence_embedding


def data2embedding(data: list, v: Vocab, token_embedding: nn.Embedding, device: torch.device):

    embedding = [string2embedding(line, v, token_embedding, device).tolist() for line in data if len(line) > 0]
    embedding = torch.tensor(embedding, device=device)
    return embedding


def fuzzy_match_score(string1: str, string2: str):

    distance = edit_distance(string1, string2)
    fms = 1.0 - distance / max(len(string1), len(string2))
    return fms


def main():

    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--load", required=True)

    parser.add_argument("--test_src_path", required=True)

    parser.add_argument("--src_memory_path", required=True)

    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)
    parser.add_argument("--test_src_with_memory_path", required=True)

    args, unknown = parser.parse_known_args()

    device = args.device

    time_data = [time.time()]

    src_data = read_data(args.test_src_path)

    src_memory = read_data(args.src_memory_path)

    src_vocab = Vocab.load(args.src_vocab_path)
    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    padding_value = src_vocab.get_index(src_vocab.mask_token)
    assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)
    s2s = load_transformer(args.load, len(src_vocab), 1, len(tgt_vocab), 1, padding_value, device=device)
    s2s.eval()

    time_data.append(time.time())

    token_embedding = s2s.encoder.token_embedding

    src_embedding = data2embedding(src_data, src_vocab, token_embedding, device)

    src_memory_embedding = data2embedding(src_memory, src_vocab, token_embedding, device)

    time_data.append(time.time())

    similarity_value, index = cosine_similarity(src_embedding, src_memory_embedding)

    time_data.append(time.time())

    src_with_memory_data = []

    for i, src_line in enumerate(src_data):

        if len(src_line) == 0:
            assert i + 1 == len(src_data)
            continue

        src_line = src_line.strip()
        src_line_memory = src_memory[index[i]].strip()

        src_line_with_memory = "{} ||| {}\nSimilarity: {}, FMS: {}".format(src_line, src_line_memory,
                                                                           str(similarity_value[i]),
                                                                           fuzzy_match_score(src_line, src_line_memory))
        src_with_memory_data.append(src_line_with_memory)

    write_data(src_with_memory_data, args.test_src_with_memory_path)

    for i in range(1, len(time_data)):
        print("Time: {}".format(time_data[i] - time_data[i-1]))


if __name__ == "__main__":
    main()
