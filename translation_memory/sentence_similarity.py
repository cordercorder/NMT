import torch
from models import transformer


def cosine_similarity(sentence_index: torch.LongTensor, memory: torch.FloatTensor):

    """
    :param sentence_index: index of every token in the vocabulary. type: torch.LongTensor. shape(input_length, )
    :param memory: sentence embedding. type: torch.FloatTensor. shape: (number_of_sentences, embedding_size)
    :return: similarity, the indices of best src sentence
    """

    # embedding: (input_length, embedding_size)
    embedding = transformer.S2S.encoder.token_embedding(sentence_index)

    # embedding: (embedding_size, )
    embedding = embedding.mean(dim=0)

    # inner_product: (number_of_sentences, embedding_size)
    inner_product = embedding * memory

    # similarity: (number_of_sentences, )
    similarity = inner_product.norm(dim=1)

    # values: (1, )
    # index: (1, )
    values, index = similarity.topk(1)
    return values, index

