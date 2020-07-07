import glob
import argparse
from utils.process import normalizeString, load_model
from utils.Vocab import Vocab
import torch
from models import S2S_attention
from nltk.translate.bleu_score import corpus_bleu
import os


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
parser.add_argument("--model_prefix", required=True)
parser.add_argument("--test_src_path", required=True)
parser.add_argument("--test_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--translation_output_dir", required=True)


args, unknown = parser.parse_known_args()

def read_data(data_path):

    with open(data_path) as f:

        data = f.read().split("\n")

        data = [normalizeString(line, to_ascii=False) for line in data]
        return data

def decode_batch(decoder_input, decoder_hidden_state, encoder_output):
    if isinstance(s2s, S2S_attention.S2S):
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state, encoder_output)
    else:
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state)


src_vocab = Vocab.load("/data/rrjin/NMT/tmpdata/src.spa.vocab")
tgt_vocab = Vocab.load("/data/rrjin/NMT/tmpdata/tgt.en.vocab")

src_data = read_data(args.test_src_path)
tgt_data = read_data(args.test_tgt_path)

device = args.device

for model_path in glob.glob(args.model_prefix + "*"):

    print("Load model from {}".format(model_path))

    s2s = (load_model(model_path)).to(device)

    s2s.eval()

    pred_data = []

    for line in src_data:

        line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

        # inputs: (input_length,)
        inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

        # inputs: (input_length, 1)
        inputs = inputs.view(-1, 1)

        encoder_output, encoder_hidden_state = s2s.encoder(inputs)

        if s2s.encoder.bidirectional_:

            if s2s.encoder.rnn_type == "lstm":

                # hn: (num_layers * num_directions, batch_size, hidden_size)
                # cn: (num_layers * num_directions, batch_size, hidden_size)
                hn, cn = encoder_hidden_state

                # hn: (num_layers, batch_size, num_directions * hidden_size)
                hn = hn.view(-1, 2, hn.size(1), hn.size(2))
                hn = torch.cat([hn[:, 0, :, :], hn[:, 1, :, :]], dim=2)

                # cn: (num_layers, batch_size, num_directions * hidden_size)
                cn = cn.view(-1, 2, cn.size(1), cn.size(2))
                cn = torch.cat([cn[:, 0, :, :], cn[:, 1, :, :]], dim=2)
                encoder_hidden_state = (hn, cn)

            else:
                encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                                 encoder_hidden_state.size(2))

                # decoder_hidden_state: (num_layers, batch_size, num_directions * hidden_size)
                encoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :],
                                                  encoder_hidden_state[:, 1, :, :]], dim=2)

        decoder_hidden_state = encoder_hidden_state

        decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.end_token)]], device=device)

        max_length = (inputs.size(0) - 2) * 3

        pred_line = []

        for i in range(max_length):

            # decoder_output: (1, 1, vocab_size)
            # decoder_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
            decoder_output, decoder_hidden_state = decode_batch(decoder_input, decoder_hidden_state,
                                                                encoder_output)

            # pred: (1, 1)
            pred = torch.argmax(decoder_output, dim=2)

            if tgt_vocab.get_token(pred[0, 0].item()) == tgt_vocab.end_token:

                break

            decoder_input = pred

            pred_line.append(tgt_vocab.get_token(pred[0, 0].item()))

        pred_data.append(pred_line)

    if not os.path.exists(args.translation_output_dir):
        os.makedirs(args.translation_output_dir)

    _, model_name = os.path.split(model_path)

    p = os.path.join(args.translation_output_dir, model_name + "_translations.txt")

    with open(p, "w") as f:

        f.write("Load model from {}\n".format(model_path))

        for tgt, pred in zip(tgt_data, pred_data):

            pred = " ".join(pred)

            f.write(tgt + "\n")
            f.write(pred + "\n")
            f.write("\n")

        bleu_value = corpus_bleu([[line.split()] for line in tgt_data], pred_data)
        f.write(str(bleu_value))

    print(bleu_value)
