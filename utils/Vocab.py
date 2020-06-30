from collections import defaultdict
from utils.process import normalizeString
import pickle
import os


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

        self.__token2index = {}
        self.__index2token = {}
        self.__token_count = defaultdict(lambda: int())

        self.unk_token = None
        self.unk_token_index = None

        self.addtoken(self.start_token)
        self.addtoken(self.end_token)

    def get_index(self, token):
        """
        :param token: token to be encoded. type: str
        :return: unique id of the token. type: int
        """
        if token not in self.__token2index:

            if self.unk_token_index is None:
                raise Exception("Token not found")
            else:
                return self.unk_token_index

        return self.__token2index.get(token)

    def get_token(self, index):
        """
        :param index: unique id of the token. type: int
        :return: the token which id is equal to index. type: str
        """
        if index not in self.__index2token:

            if self.unk_token is None:
                raise Exception("Index not found")
            else:
                return self.unk_token

        return self.__index2token.get(index)

    def addtoken(self, token):

        """
        :param token: token to be added. type: str
        :return: None
        """

        self.__token_count[token] = self.__token_count[token] + 1

        if token not in self.__token2index:

            num = len(self.__token2index)
            self.__token2index[token] = num
            self.__index2token[num] = token

    def add_sentence(self, sentence, normalize=True):

        """
        :param sentence: sentence to be added. type: str
        :param normalize: whether normalize the sentence or not, default True. type: bool
        :return: None
        """
        if normalize:
            sentence = normalizeString(sentence)
        for token in sentence.split():
            self.addtoken(token)

    def add_corpus(self, corpus_path, normalize=True):

        """
        :param corpus_path: location of corpus. type: str
        :param normalize: whether normalize sentence in the corpus or not, default True. type: bool
        :return: None
        """

        with open(corpus_path) as f:
            data = f.read().split("\n")

            for line in data:
                self.add_sentence(line, normalize=normalize)

    def add_unk(self, unk="UNK"):

        """
        Add unknown words
        :return: None
        """

        self.unk_token = unk

        self.__token_count = defaultdict(lambda: int(), {
                                            k: value for k, value in self.__token_count.items()
                                            if value >= self.threshold
                                       })

        self.__token_count[self.unk_token] = 1

        self.__token2index = {}
        self.__index2token = {}

        for tok in self.__token_count:

            if tok not in self.__token2index:
                num = len(self.__token2index)
                self.__token2index[tok] = num
                self.__index2token[num] = tok

        self.unk_token_index = self.__token2index[self.unk_token]

    def save(self, save_path):
        """
        save the entity to disk
        :param save_path: path to save the entity. type: str
        :return: None
        """

        directory, file_name = os.path.split(save_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        entity = {
            "language_name": self.language_name,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "threshold": self.threshold,
            "__token2index": self.__token2index,
            "__index2token": self.__index2token,
            "__token_count": dict(self.__token_count),
            "unk_token": self.unk_token,
            "unk_token_index": self.unk_token_index
        }

        with open(save_path, "wb") as f:
            pickle.dump(entity, f)

    @classmethod
    def load(cls, load_path):
        """
        load the entity from disk
        :param load_path: path to load the entity. type: str
        :return: the load entity. type: object
        """
        with open(load_path, "rb") as f:
            entity = pickle.load(f)
            language_name = entity["language_name"]
            start_token = entity["start_token"]
            end_token = entity["end_token"]
            threshold = entity["threshold"]
            __token2index = entity["__token2index"]
            __index2token = entity["__index2token"]
            __token_count = defaultdict(lambda: int(), entity["__token_count"])
            unk_token = entity["unk_token"]
            unk_token_index = entity["unk_token_index"]

            v = cls(language_name, start_token, end_token, threshold)

            v.__token_count = __token_count
            v.__index2token = __index2token
            v.__token2index = __token2index
            v.unk_token = unk_token
            v.unk_token_index = unk_token_index
            return v

    def __len__(self):

        """
        :return: the number of different tokens
        """

        return len(self.__token_count)

    def __repr__(self):

        """
        :return: randomly return 10 tokens if the number of tokens in self.__token_count is larger than 10,
                 else return all tokens
        """

        tmp_key = list(self.__token_count.keys())
        if len(tmp_key) > 10:
            s = ", ".join(tmp_key[:10]) + "......"
        else:
            s = ", ".join(tmp_key)

        return s


if __name__ == "__main__":

    def test(v):
        print("------")
        print(v.language_name)
        print("------\n")

        print("------")
        print(v._Vocab__index2token)
        print("------\n")

        print("------")
        print(v._Vocab__token2index)
        print("------\n")

        print("------")
        print(v._Vocab__token_count)
        print("------\n")

        print("------")
        print(v.start_token)
        print("------\n")

        print("------")
        print(v.end_token)
        print("------\n")

        print("------")
        print(v.threshold)
        print("------\n")

        print("------")
        print(v.unk_token)
        print("------\n")

        print("------")
        print(v.unk_token_index)
        print("------\n")


    v = Vocab("en", "<s>", "<e>")

    v.addtoken("test1")
    v.addtoken("test2")

    v.add_corpus("D:/jinrenren/NLP_study/codes/NMT/data/test_input.en")

    # print(v.get_index("test"))

    test(v)

    v.add_unk()

    print(v.get_index("test"))

    v.save("D:/jinrenren/NLP_study/codes/NMT/data/vocab.en")

    test(v)

    v2 = Vocab.load("D:/jinrenren/NLP_study/codes/NMT/data/vocab.en")

    test(v2)