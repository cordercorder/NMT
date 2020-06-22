from collections import defaultdict
from .process import normalizeString


class Vocab:

    def __init__(self, language_name, start_token, end_token, threshold=0):

        """
        :param language_name: name of language that needed to be processing. type: str
        :param start_token: start token before a sentence. type: str
        :param end_token: end token at the end of a sentence. type: str
        :param threshold: minimum frequency of the token to be taken into account, default 0. type: int
        :return: None
        """

        self.language_name = language_name
        self.start_token = start_token
        self.end_token = end_token
        self.threshold = threshold

        self.token2index = {}
        self.index2token = {}
        self.token_count = defaultdict(lambda: int())

        self.addtoken(self.start_token)
        self.addtoken(self.end_token)

    def addtoken(self, token):

        """
        :param token: token to be added. type: str
        :return: None
        """

        self.token_count[token] = self.token_count[token] + 1

        if token not in self.token2index:

            num = len(self.token2index)
            self.token2index[token] = num
            self.index2token[num] = token

    def add_sentence(self, sentence):

        """
        :param sentence: sentence to be added. type: str
        :return: None
        """
        sentence = normalizeString(sentence)
        for token in sentence.split():
            self.addtoken(token)

    def add_corpus(self, corpus_path):

        """
        :param corpus_path: location of corpus. type: str
        :return: None
        """

        with open(corpus_path) as f:
            data = f.read().split("\n")

            for line in data:
                self.add_sentence(line)

    def __len__(self):

        """
        :return: the number of different tokens
        """

        return len(self.token_count)

    def __repr__(self):

        """
        :return: randomly return 10 tokens if the number of tokens in self.token_count if larger than 10,
                 else return all tokens
        """

        tmp_key = list(self.token_count.keys())
        if len(tmp_key) > 10:
            s = ", ".join(tmp_key[:10]) + "......"
        else:
            s = ", ".join(tmp_key)

        return s


if __name__ == "__main__":

    v = Vocab("en", "<s>", "<e>")

    v.addtoken("test1")
    v.addtoken("test2")

    v.add_corpus("D:/jinrenren/NLP_study/codes/NMT\data/test_input.en")

    for tok in v.token2index:
        print(tok)
        print(v.token2index[tok])

    for index in v.index2token:
        print(index)
        print(v.index2token[index])

    print(v.index2token)
    print(v.token2index)