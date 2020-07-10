import unicodedata
import re
from models import S2S_attention, S2S_basic
import torch


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
    # s = re.sub(r"([.!?,:;'\"¿，。；：‘、？．“”！])", r" \1 ", s)
    s = re.sub(r"([!\"#$%&'()*+,-./:;<=>?@\[\]\\^_`{|}~¿，。；：‘、？．“”！])", r" \1 ", s)

    if remove_punctuation:
        # remove regular punctuation
        s = re.sub(r"[.!?,:;'\"¿，。；：、‘？．“”！]+", r" ", s)
    else:
        # remove quotation marks
        s = re.sub(r"['‘\"“”]+", r" ", s)
    
    s = "".join(c for c in s if c != "@")

    # change multiple spaces into one space
    s = re.sub(r"[ ]+", " ", s)

    s = s.strip()

    return s


def sort_src_sentence_by_length(data):

    data.sort(key=lambda item: len(item[0]))

    src_data, tgt_data = list(zip(*data))
    return src_data, tgt_data


def save_model(s2s_model, optimizer, args):

    if isinstance(s2s_model, torch.nn.parallel.distributed.DistributedDataParallel):

        s2s_model = s2s_model.module

    if hasattr(s2s_model.decoder, "attention"):
        return {
            "model_dict": s2s_model.state_dict(),
            "encoder": {
                "rnn_type": s2s_model.encoder.rnn_type,
                "vocab_size": s2s_model.encoder.vocab_size,
                "embedding_size": s2s_model.encoder.embedding_size,
                "hidden_size": s2s_model.encoder.hidden_size,
                "num_layers": s2s_model.encoder.num_layers,
                "dropout_": s2s_model.encoder.dropout_,
                "bidirectional_": s2s_model.encoder.bidirectional_
            },
            "attention":{
                "hidden_size1": s2s_model.decoder.attention.hidden_size1,
                "hidden_size2": s2s_model.decoder.attention.hidden_size2,
                "attention_size": s2s_model.decoder.attention.attention_size
            },
            "decoder": {
                "rnn_type": s2s_model.decoder.rnn_type,
                "vocab_size": s2s_model.decoder.vocab_size,
                "embedding_size": s2s_model.decoder.embedding_size,
                "input_size": s2s_model.decoder.input_size,
                "hidden_size": s2s_model.decoder.hidden_size,
                "num_layers": s2s_model.decoder.num_layers,
                "dropout_": s2s_model.decoder.dropout_
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args
        }

    else:
        return {
            "model_dict": s2s_model.state_dict(),
            "encoder": {
                "rnn_type": s2s_model.encoder.rnn_type,
                "vocab_size": s2s_model.encoder.vocab_size,
                "embedding_size": s2s_model.encoder.embedding_size,
                "hidden_size": s2s_model.encoder.hidden_size,
                "num_layers": s2s_model.encoder.num_layers,
                "dropout_": s2s_model.encoder.dropout_,
                "bidirectional_": s2s_model.encoder.bidirectional_
            },
            "decoder": {
                "rnn_type": s2s_model.decoder.rnn_type,
                "vocab_size": s2s_model.decoder.vocab_size,
                "embedding_size": s2s_model.decoder.embedding_size,
                "hidden_size": s2s_model.decoder.hidden_size,
                "num_layers": s2s_model.decoder.num_layers,
                "dropout_": s2s_model.decoder.dropout_
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args
        }

def load_model(model_path, training=False, device="cpu"):

    model_ckpt = torch.load(model_path, map_location="cpu")

    if "attention" in model_ckpt:

        print("Attention Model")
        encoder = S2S_attention.Encoder(**model_ckpt["encoder"])
        attention = S2S_attention.BahdanauAttention(**model_ckpt["attention"])
        decoder = S2S_attention.AttentionDecoder(**model_ckpt["decoder"], attention=attention)
        s2s = S2S_attention.S2S(encoder, decoder).to(device)

    else:

        print("Basic Model")
        encoder = S2S_basic.Encoder(**model_ckpt["encoder"])
        decoder = S2S_basic.Decoder(**model_ckpt["decoder"])
        s2s = S2S_basic.S2S(encoder, decoder).to(device)

    s2s.load_state_dict(model_ckpt["model_dict"])

    if training:

        # load for training

        optimizer_state_dict = model_ckpt["optimizer_state_dict"]

        return s2s, optimizer_state_dict

    return s2s


if __name__ == "__main__":

    print(unicodeToAscii("Ślusàrski  "))

    s = "¿Puedo     omar    prestado este libro? \" "
    print(s)
    print(normalizeString(s))

    s = "tom prefers his dirty and and a \" . . . . . . . . . . . . \" . . \" \""
    print(normalizeString((s)))