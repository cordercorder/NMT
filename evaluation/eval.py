import argparse
from utils.Vocab import Vocab
from utils.tools import load_model, read_data, write_data
from nltk.translate.bleu_score import corpus_bleu
from evaluation.S2S_translation import beam_search_decoding


parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--translation_output", required=True)

parser.add_argument("--beam_size", default=3, type=int)

args, unknown = parser.parse_known_args()

device = args.device

print("Load from: {}".format(args.load))

s2s = load_model(args.load, device=device)

print("Output is: {}".format(args.translation_output))


src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

pred_data = []

src_data = read_data(args.test_src_path)
tgt_data = read_data(args.test_tgt_path)

s2s.eval()

for line in src_data:
    pred_data.append(beam_search_decoding(s2s, line, src_vocab, tgt_vocab, args.beam_size, device))

write_data(pred_data, args.translation_output)

with open(args.translation_output, "w") as f:

    for tgt, pred in zip(tgt_data, pred_data):

        pred = " ".join(pred)
        f.write(tgt + "\n")
        f.write(pred + "\n")
        f.write("\n")
    
    bleu_value = corpus_bleu([[line.split()] for line in tgt_data], pred_data)
    f.write(str(bleu_value))

print(pred_data[-10:])
print(tgt_data[-10:])
print(bleu_value)