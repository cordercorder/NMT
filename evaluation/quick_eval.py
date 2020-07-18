import glob
import argparse
from utils.process import read_data, load_model, load_transformer, write_data
from utils.Vocab import Vocab
from evaluation.S2S_translation import greedy_decoding
import os
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
parser.add_argument("--model_prefix", required=True)
parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--translation_output_dir", required=True)
parser.add_argument("--transformer", action="store_true", default=False)

args, unknown = parser.parse_known_args()

src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

src_data = read_data(args.test_src_path)

device = args.device


for model_path in glob.glob(args.model_prefix + "*"):

    print("Load model from {}".format(model_path))

    if args.transformer:

        max_src_len = max(len(line.split()) for line in src_data) + 2
        max_tgt_len = (max_src_len - 2) * 3

        padding_value = src_vocab.get_index(src_vocab.mask_token)

        assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

        s2s = load_transformer(model_path, len(src_vocab), max_src_len, len(tgt_vocab), max_tgt_len, padding_value,
                               device=device)

    else:
        s2s = load_model(model_path, device=device)

    s2s.eval()

    pred_data = []

    for line in src_data:
        pred_data.append(greedy_decoding(s2s, line, src_vocab, tgt_vocab, device, args.transformer))

    if not os.path.exists(args.translation_output_dir):
        os.makedirs(args.translation_output_dir)

    _, model_name = os.path.split(model_path)

    p = os.path.join(args.translation_output_dir, model_name + "_translations.txt")

    write_data(pred_data, p)

    p_tok = os.path.join(args.translation_output_dir, model_name + "_translations_tok.txt")

    tok_command = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(p, p_tok)

    call(tok_command, shell=True)

    bleu_calculation_command = "perl /data/rrjin/NMT/scripts/multi-bleu.perl {} < {}".format(args.test_tgt_path, p_tok)

    call(bleu_calculation_command, shell=True)
