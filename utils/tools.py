import unicodedata
import re
import torch
import logging
import os

import numpy as np
import random

from models import S2S_attention, S2S_basic, transformer
from utils.adam_inverse_sqrt_with_warmup import AdamInverseSqrtWithWarmup

logging.basicConfig(level=logging.DEBUG)


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
    s = re.sub(r"([!\"#$%&'()*+,-./:;<=>?\[\]\\^_`{|}~¿，。；：‘、？．“”！])", r" \1 ", s)

    if remove_punctuation:
        # remove regular punctuation
        s = re.sub(r"[.!?,:;'\"¿，。；：、‘？．“”！]+", r" ", s)
    else:
        # remove quotation marks
        s = re.sub(r"['‘\"“”]+", r" ", s)

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
            "attention": {
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

        logging.info("Attention Model")
        encoder = S2S_attention.Encoder(**model_ckpt["encoder"])
        attention = S2S_attention.BahdanauAttention(**model_ckpt["attention"])
        decoder = S2S_attention.AttentionDecoder(**model_ckpt["decoder"], attention=attention)
        s2s = S2S_attention.S2S(encoder, decoder).to(device)

    else:

        logging.info("Basic Model")
        encoder = S2S_basic.Encoder(**model_ckpt["encoder"])
        decoder = S2S_basic.Decoder(**model_ckpt["decoder"])
        s2s = S2S_basic.S2S(encoder, decoder).to(device)

    s2s.load_state_dict(model_ckpt["model_dict"])

    if training:

        # load for training

        optimizer_state_dict = model_ckpt["optimizer_state_dict"]

        return s2s, optimizer_state_dict

    return s2s


def save_transformer(s2s_model, optimizer, args):

    if isinstance(s2s_model, torch.nn.parallel.distributed.DistributedDataParallel):
        s2s_model = s2s_model.module

    return {
        "model_dict": s2s_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args
    }


def build_transformer(args, device):

    encoder = transformer.Encoder(args.src_vocab_size, args.max_src_len, args.d_model, args.num_layers, args.num_heads,
                                  args.d_ff, args.encoder_max_rpe, args.dropout, device)
    decoder = transformer.Decoder(args.tgt_vocab_size, args.max_tgt_len, args.d_model, args.num_layers, args.num_heads,
                                  args.d_ff, args.share_dec_pro_emb, args.decoder_max_rpe, args.dropout, device)
    s2s = transformer.S2S(encoder, decoder, args.padding_value, device).to(device)
    return s2s


def load_transformer(args, training=False, device="cpu"):

    model_ckpt = torch.load(args.load, map_location="cpu")
    args = model_ckpt["args"]
    s2s = build_transformer(args, device)
    s2s.load_state_dict(model_ckpt["model_dict"])

    if training:
        optimizer_state_dict = model_ckpt["optimizer_state_dict"]
        return s2s, optimizer_state_dict

    return s2s


def get_optimizer(parameters, args):

    if args.optim_method == "fix_learning_rate":
        opt_params = {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "weight_decay": args.weight_decay
        }
        return torch.optim.Adam(parameters, **opt_params)
    else:
        opt_params = {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "warmup_updates": args.warmup_updates,
            "warmup_init_lr": args.warmup_init_lr
        }
        return AdamInverseSqrtWithWarmup(parameters, **opt_params)


def write_data(data, write_path):

    directory, _ = os.path.split(write_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if data and isinstance(data[0], list):
        data = [" ".join(line) for line in data]

    with open(write_path, "w") as f:
        f.write("\n".join(data))


def read_data(data_path):

    with open(data_path) as f:

        data = f.read().strip().split("\n")
        return data


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_tools():
    print(unicodeToAscii("Ślusàrski  "))

    s = "¿Puedo     omar    prestado este libro? \" "
    print(s)
    print(normalizeString(s))

    s = "tom prefers his dirty and and a \" . . . . . . . . . . . . \" . . \" \""
    print(normalizeString((s)))


if __name__ == "__main__":
    test_tools()
