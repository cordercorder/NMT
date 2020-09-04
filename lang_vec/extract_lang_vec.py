import argparse
import torch

from utils.Vocab import Vocab
from utils.tools import load_model, read_data, load_transformer
from lang_vec.lang_vec_tools import save_lang_vec

parser = argparse.ArgumentParser()

parser.add_argument("--load", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--lang_token_path", required=True)
parser.add_argument("--lang_vec_path", required=True)

parser.add_argument("--transformer", action="store_true")
parser.add_argument("--lang_vec_type", required=True, choices=["token_embedding", "layer_token_embedding"])
parser.add_argument("--layer_id", type=int)

# only for transformer
parser.add_argument("--tgt_vocab_path")

args, unknown = parser.parse_known_args()

src_vocab = Vocab.load(args.src_vocab_path)

if args.transformer:

    assert args.tgt_vocab_path is not None

    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    padding_value = src_vocab.get_index(src_vocab.mask_token)

    assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

    s2s = load_transformer(args.load, len(src_vocab), 10, len(tgt_vocab), 10, padding_value)

else:

    # multi_layer_token_embedding only support transformer now
    assert args.lang_vec_type == "token_embedding"
    s2s = load_model(args.load)

s2s.eval()

lang_token_list = read_data(args.lang_token_list_path)

lang_vec = {}

for lang_token in lang_token_list:

    if len(lang_token) == 0:
        continue

    assert lang_token in src_vocab, lang_token

    print(lang_token)

    index = src_vocab.get_index(lang_token)

    with torch.no_grad():

        if args.lang_vec_type == "token_embedding":

            if args.transformer:
                token_embedding = s2s.encoder.token_embedding(torch.tensor(index, dtype=torch.long))
            else:
                token_embedding = s2s.encoder.embedding(torch.tensor(index, dtype=torch.long))

            lang_vec[lang_token] = token_embedding.tolist()

        else:
            # src: (1, 1)

            assert args.layer_id is not None

            src = torch.tensor([[index]])
            src_mask = s2s.make_src_mask(src)
            src = s2s.encoder.token_embedding(src) * s2s.encoder.scale

            for i, layer in enumerate(s2s.encoder.layers):
                src = layer(src, src_mask)
                # src.squeeze(): (d_model, )
                if i == args.layer_id:
                    lang_vec[lang_token] = src.squeeze().tolist()
                    break

if not lang_vec:
    print("language vector is empty")
else:
    save_lang_vec(lang_vec, args.lang_vec_path)
