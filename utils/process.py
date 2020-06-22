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


def normalizeString(s, remove_punctuation=False):

    """
    :param s: string to be processed. type: str
    :param remove_punctuation: if True, remove all punctuation in the string, default False. type: bool
    :return: string after processed
    """

    s = unicodeToAscii(s.lower().strip())

    # add a space between normal character and punctuation
    s = re.sub(r"([.!?,¿，。；‘\"？，“！])", r" \1 ", s)

    # change multiple spaces into one space
    s = re.sub(r"[ ]+", " ", s)

    if remove_punctuation:
        # remove punctuation
        s = re.sub(r"[^a-zA-Z]+", r" ", s)
    else:
        s = re.sub(r"[^a-zA-Z.!?,¿，。；‘\"？，“！]+", r" ", s)

    s = s.strip()

    return s


if __name__ == "__main__":

    print(unicodeToAscii("Ślusàrski  "))

    s = "¿Puedo     omar    prestado este libro? \" "
    print(s)
    print(normalizeString(s))