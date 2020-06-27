import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.Vocab import Vocab
from utils.process import normalizeString
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--test_src_vocab_path", required=True)
parser.add_argument("--test_tgt_vocab_path", required=True)
parser.add_argument("--translation_output", required=True)

parser.add_argument("--beam_size", default=1, type=int)

args, unknown = parser.parse_known_args()

device = args.device

s2s = (torch.load(args.load)["model"]).to(device)

src_vocab = Vocab.load(args.test_src_vocab_path)
tgt_vocab = Vocab.load(args.test_tgt_vocab_path)

src_data = []
pred_data = []

with open(args.test_tgt_path) as f:
    data = f.read().split("\n")
    tgt_data = [normalizeString(line) for line in data]

max_tgt_length = max(len(line.split()) for line in tgt_data)

with torch.no_grad():
    s2s.eval()

    with open(args.test_src_path) as f:

        data = f.read().split("\n")

    for line in data:

        pred_line = []

        line = " ".join([src_vocab.start_token, normalizeString(line)])

        # inputs: (input_length,)
        inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()]).to(device)

        # inputs: (input_length, 1)
        inputs = inputs.view(-1, 1)

        # encoder_output: (input_length, batch_size, num_directions * hidden_size)
        # encoder_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
        encoder_output, encoder_hidden_state = s2s.encoder(inputs)

        # decoder_input: (1, 1)
        decoder_input = torch.tensor([[src_vocab.get_index(src_vocab.start_token)]]).to(device)

        if s2s.encoder.bidirectional_:

            if s2s.encoder.rnn_type == "lstm":

                # hn: (num_layers * num_directions, batch_size, hidden_size)
                # cn: (num_layers * num_directions, batch_size, hidden_size)
                hn, cn = encoder_hidden_state

                hn = hn.view(-1, 2, hn.size(1), hn.size(2))
                hn = torch.cat([hn[:, 0, :, :], hn[:, 1, :, :]], dim=2)

                cn = cn.view(-1, 2, cn.size(1), cn.size(2))
                cn = torch.cat([cn[:, 0, :, :], cn[:, 1, :, :]], dim=2)
                decoder_hidden_state = (hn, cn)

            else:
                encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                                 encoder_hidden_state.size(2))
                decoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :],
                                                  encoder_hidden_state[:, 1, :, :]], dim=2)

        for t in range(max_tgt_length):

            # decoder_output: (1, 1, hidden_state)
            decoder_output, decoder_hidden_state = s2s.decoder.decode_batch(decoder_input, decoder_hidden_state)

            # pred: (1, 1, vocab_size)
            pred = F.softmax(decoder_output, dim=2)

            # pred_token_prob: (1, 1, args.beam_size)
            # pred_token_index: (1, 1, args.beam_size)
            pred_token_prob, pred_token_index = torch.topk(pred, args.beam_size, dim=2)

            pred_token_index = pred_token_index.view(args.beam_size)

            pred_token = tgt_vocab.get_token(pred_token_index[0].item())

            pred_line.append(pred_token)

            if pred_token == tgt_vocab.end_token:
                break

            decoder_input = pred_token_index[0].view(-1, 1)

        pred_data.append(pred_line)

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