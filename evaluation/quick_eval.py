import glob
import argparse
import os
import torch

from torch.utils.data import DataLoader
from subprocess import call

from utils.tools import read_data, load_model, load_transformer, write_data
from utils.data_loader import SrcData
from utils.Vocab import Vocab
from evaluation.S2S_translation import greedy_decoding, beam_search_decoding
from utils.data_loader import convert_data_to_index, pad_data


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)

parser.add_argument("--model_load", required=True)
parser.add_argument("--transformer", action="store_true")
parser.add_argument("--is_prefix", action="store_true")

parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--share_dec_pro_emb", type=bool, default=True, help="share decoder input and project embedding")

parser.add_argument("--translation_output_dir", required=True)
parser.add_argument("--beam_size", type=int)

parser.add_argument("--record_time", action="store_true")
parser.add_argument("--need_tok", action="store_true")

parser.add_argument("--batch_size", type=int)

parser.add_argument("--bleu_script_path", required=True)

args, unknown = parser.parse_known_args()

device = args.device

torch.cuda.set_device(device)

src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

src_data = read_data(args.test_src_path)

# |------ for transformer ------|
max_src_len = max(len(line.split()) for line in src_data) + 2
max_tgt_len = max_src_len * 3
print("max src sentence length: {}".format(max_src_len))
# |------------ end ------------|

src_data = convert_data_to_index(src_data, src_vocab)
src_data = SrcData(src_data)

padding_value = src_vocab.get_index(src_vocab.mask_token)
assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

if args.batch_size:
    assert args.beam_size is None, "batch translation do not support bream search now"

if args.beam_size:
    print("Beam size: {}".format(args.beam_size))

data_loader = DataLoader(src_data, batch_size=(args.batch_size if args.batch_size else 1), shuffle=False,
                         collate_fn=lambda batch: pad_data(batch, padding_value,
                                                           batch_first=(True if args.transformer else False)),
                         pin_memory=True, drop_last=False)

if args.is_prefix:
    args.model_load = args.model_load + "*"

for model_path in glob.glob(args.model_load):

    print("Load model from {}".format(model_path))

    if args.transformer:

        s2s = load_transformer(model_path, len(src_vocab), max_src_len, len(tgt_vocab), max_tgt_len, padding_value,
                               training=False, share_dec_pro_emb=args.share_dec_pro_emb,device=device)

    else:
        s2s = load_model(model_path, device=device)

    s2s.eval()

    pred_data = []

    if args.record_time:
        import time
        start_time = time.time()

    for data in data_loader:
        if args.beam_size:
            pred_data.append(beam_search_decoding(s2s, data.to(device, non_blocking=True), tgt_vocab, args.beam_size, device))
        else:
            pred_data.extend(greedy_decoding(s2s, data.to(device, non_blocking=True), tgt_vocab, device))

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

        bleu_calculation_command = "perl {} {} < {}".format(args.bleu_script_path, args.test_tgt_path,
                                                                                                 p_tok)

    else:
        bleu_calculation_command = "perl {} {} < {}".format(args.bleu_script_path, args.test_tgt_path, p)

    call(bleu_calculation_command, shell=True)

    print("")
