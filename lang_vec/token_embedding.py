import argparse
from utils.Vocab import Vocab
from utils.tools import load_model, read_data, load_transformer
from lang_vec.lang_vec_tools import save_lang_vec
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--lang_token_list_path", required=True)
parser.add_argument("--lang_vec_path", required=True)

parser.add_argument("--transformer", action="store_true")

# only for transformer
parser.add_argument("--tgt_vocab_path")

args, unknown = parser.parse_known_args()

device = args.device

src_vocab = Vocab.load(args.src_vocab_path)

if args.transformer:

    assert args.tgt_vocab_path is not None

    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    padding_value = src_vocab.get_index(src_vocab.mask_token)

    assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

    s2s = load_transformer(args.load, len(src_vocab), 10, len(tgt_vocab), 10, padding_value, device=device)

else:
    s2s = load_model(args.load, device=device)

lang_token_list = read_data(args.lang_token_list_path)

lang_vec = {}

for lang_token in lang_token_list:

    if len(lang_token) == 0:
        continue

    assert lang_token in src_vocab, lang_token

    print(lang_token)

    index = src_vocab.get_index(lang_token)

    with torch.no_grad():

        if args.transformer:
            token_embedding = s2s.encoder.token_embedding(torch.tensor(index, dtype=torch.long, device=device))
        else:
            token_embedding = s2s.encoder.embedding(torch.tensor(index, dtype=torch.long, device=device))

    lang_vec[lang_token] = token_embedding.tolist()

save_lang_vec(lang_vec, args.lang_vec_path)
