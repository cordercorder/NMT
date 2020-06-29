import unicodedata
import re


def unicodeToAscii(s):

    """
    change unicode character into ASCII.
    eg: s: Ślusàrski, return: Slusarski
    """

    return "".join(
        letter for letter in unicodedata.normalize("NFD", s)
        if unicodedata.category(letter) != "Mn"
    )


def normalizeString(s, remove_punctuation=False, to_ascii=True):

    """
    :param s: string to be processed. type: str
    :param remove_punctuation: if True, remove all punctuation in the string, default False. type: bool
    :param to_ascii: convert the character in s to ascii character, default True. type: bool
    :return: string after processed
    """
    s = s.lower().strip()

    if to_ascii:
        s = unicodeToAscii(s)

    # add a space between normal character and punctuation
    s = re.sub(r"([.!?,¿，。；'‘\"？，“”！])", r" \1 ", s)

    if remove_punctuation:
        # remove punctuation
        s = re.sub(r"[.!?,¿，。；'‘\"？，“”！]+", r" ", s)
    else:
        # remove quotation marks
        s = re.sub(r"['‘\"“”]+", r" ", s)
    
    s = "".join(c for c in s if c != "@")


    # change multiple spaces into one space
    s = re.sub(r"[ ]+", " ", s)

    s = s.strip()

    return s


def sort_sentence_by_length(data):

    return data.sort(key=lambda item: len(item))


if __name__ == "__main__":

    print(unicodeToAscii("Ślusàrski  "))

    s = "¿Puedo     omar    prestado este libro? \" "
    print(s)
    print(normalizeString(s))

    s = "tom prefers his dirty and and a \" . . . . . . . . . . . . \" . . \" \""
    print(normalizeString((s)))