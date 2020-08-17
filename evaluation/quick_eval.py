import glob
import argparse
from utils.tools import read_data, load_model, load_transformer, write_data
from utils.Vocab import Vocab
from evaluation.S2S_translation import greedy_decoding, beam_search_decoding
import os
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)

parser.add_argument("--model_load", required=True)
parser.add_argument("--transformer", action="store_true")
parser.add_argument("--is_prefix", action="store_true")

parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)

parser.add_argument("--translation_output_dir", required=True)
parser.add_argument("--beam_size", type=int)

parser.add_argument("--record_time", action="store_true")
parser.add_argument("--need_tok", action="store_true")

args, unknown = parser.parse_known_args()

src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

src_data = read_data(args.test_src_path)

device = args.device

if args.beam_size:
    print("Beam size: {}".format(args.beam_size))

if args.is_prefix:
    args.model_load = args.model_load + "*"

for model_path in glob.glob(args.model_load):

    print("Load model from {}".format(model_path))

    if args.transformer:

        max_src_len = max(len(line.split()) for line in src_data) + 2
        max_tgt_len = max_src_len * 3

        padding_value = src_vocab.get_index(src_vocab.mask_token)

        assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

        s2s = load_transformer(model_path, len(src_vocab), max_src_len, len(tgt_vocab), max_tgt_len, padding_value,
                               device=device)

    else:
        s2s = load_model(model_path, device=device)

    s2s.eval()

    pred_data = []

    if args.record_time:
        import time
        start_time = time.time()

    for line in src_data:
        if args.beam_size:
            pred_data.append(beam_search_decoding(s2s, line, src_vocab, tgt_vocab, args.beam_size, device))
        else:
            pred_data.append(greedy_decoding(s2s, line, src_vocab, tgt_vocab, device, args.transformer))

    if args.record_time:
        end_time = time.time()
        print("Time spend: {} seconds".format(end_time - start_time))

    if not os.path.exists(args.translation_output_dir):
        os.makedirs(args.translation_output_dir)

    _, model_name = os.path.split(model_path)

    if args.beam_size:
        translation_file_name_prefix = "{}_beam_size{}".format(model_name, args.beam_size)
    else:
        translation_file_name_prefix = model_name
    p = os.path.join(args.translation_output_dir, translation_file_name_prefix + "_translations.txt")

    write_data(pred_data, p)

    if args.need_tok:

        # replace '@@ ' with ''

        p_tok = os.path.join(args.translation_output_dir, translation_file_name_prefix + "_translations_tok.txt")

        tok_command = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(p, p_tok)

        call(tok_command, shell=True)

        bleu_calculation_command = "perl /data/rrjin/NMT/scripts/multi-bleu.perl {} < {}".format(args.test_tgt_path,
                                                                                                 p_tok)

    else:
        bleu_calculation_command = "perl /data/rrjin/NMT/scripts/multi-bleu.perl {} < {}".format(args.test_tgt_path, p)

    call(bleu_calculation_command, shell=True)

    print("")
