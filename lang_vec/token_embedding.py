import argparse
from utils.Vocab import Vocab
from utils.tools import load_model, read_data
from lang_vec.lang_vec_tools import save_lang_vec
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--lang_token_list_path", required=True)
parser.add_argument("--lang_vec_path", required=True)

args, unknown = parser.parse_known_args()

device = args.device

src_vocab = Vocab.load(args.src_vocab_path)
s2s = load_model(args.load, device=device)

lang_token_list = read_data(args.lang_token_list_path)

lang_vec = {}

for lang_token in lang_token_list:

    assert lang_token in src_vocab

    index = src_vocab.get_index(lang_token)

    with torch.no_grad():
        token_embedding = s2s.encoder.embedding(torch.tensor(index, dtype=torch.long, device=device))

    lang_vec[lang_token] = token_embedding.tolist()

save_lang_vec(lang_vec, args.lang_vec_path)
