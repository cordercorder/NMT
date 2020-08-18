import argparse
import unicodedata

from utils.tools import read_data, write_data


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case, strip_accents, tokenize_chinese_chars):
        """Constructs a BasicTokenizer.

        Args:
        do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text):
        """Tokenizes a piece of text."""

        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                if self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        # output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # return output_tokens
        return (" ".join(split_tokens)).strip()

    @staticmethod
    def _run_strip_accents(text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    @staticmethod
    def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", nargs="+")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--tokenize_chinese_chars", action="store_true")

    args, unknown = parser.parse_known_args()

    print("do_lower_case: {}, strip_accents: {}, tokenize_chinese_chars: {}".format(args.do_lower_case,
                                                                                    args.strip_accents,
                                                                                    args.tokenize_chinese_chars))

    tokenizer = BasicTokenizer(do_lower_case=args.do_lower_case, strip_accents=args.strip_accents,
                               tokenize_chinese_chars=args.tokenize_chinese_chars)

    for data_path in args.data_path:

        data = read_data(data_path)
        tok_data = []

        if data_path.endswith(".en"):

            tok_data = [tokenizer.tokenize(sentence) for sentence in data]

        else:
            for sentence in data:

                sentence = sentence.strip()

                first_blank_pos = sentence.find(" ")

                if first_blank_pos != -1:

                    # do not process language identify token
                    lang_identify_token = sentence[:first_blank_pos]

                    sentence_after_process = tokenizer.tokenize(sentence[first_blank_pos+1:])
                    sentence_after_process = " ".join([lang_identify_token, sentence_after_process])

                    tok_data.append(sentence_after_process)

                else:
                    tok_data.append(sentence)

        idx = data_path.rfind(".")

        assert idx != -1

        tok_data_path = data_path[:idx] + "_tok" + data_path[idx:]

        print("{} > {}".format(data_path, tok_data_path))

        write_data(tok_data, tok_data_path)


def test():

    s = "ninid,dsadada()dsaddada"
    print(BasicTokenizer._run_split_on_punc(s))

    tokenizer = BasicTokenizer(False, False, False)

    print(tokenizer.tokenize("<cs> svou přednášku jsem uvedl slovy mistrů pera, oddělených od sebe oceánem a staletí"
                             "，。m ."))

    print(tokenizer.tokenize("你好"))


if __name__ == "__main__":
    main()
    # test()
