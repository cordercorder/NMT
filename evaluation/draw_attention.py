import torch
import torch.nn.functional as F
import argparse
from utils.Vocab import Vocab
from utils.process import normalizeString
from models import S2S_attention
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_attention(attention, sentence, predicted_sentence, picture_path):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(picture_path)


parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--load", required=True)
parser.add_argument("--val_src_path", required=True)
parser.add_argument("--val_tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--picture_path", required=True)

args, unknown = parser.parse_known_args()

device = args.device

s2s = (torch.load(args.load)).to(device)

if not isinstance(s2s, S2S_attention.S2S):

    raise Exception("The model don't have attention mechanism")

src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

with open(args.val_tgt_path) as f:

    data = f.read().split("\n")
    tgt_data = [normalizeString(line) for line in data]

with torch.no_grad():

    s2s.eval()

    with open(args.val_src_path) as f:

        data = f.read().split("\n")

        for line in data:

            line = " ".join([src_vocab.start_token, normalizeString(line), src_vocab.end_token])

            inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

            # (input_length, 1)
            inputs = inputs.view(-1, 1)

            max_length = (inputs.size(0) - 2) * 3

            # encoder_output: (input_length, 1, num_directions * hidden_size)
            # encoder_hidden_state: (num_layers * num_directions, 1, hidden_size)
            encoder_output, encoder_hidden_state = s2s.encoder(inputs)

            # decoder_input: (1, 1)
            decoder_input = torch.tensor([[src_vocab.get_index(src_vocab.start_token)]], device=device)


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
                    decoder_hidden_state = (hn, cn)

                else:
                    encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                                     encoder_hidden_state.size(2))

                    # decoder_hidden_state: (num_layers, batch_size, num_directions * hidden_size)
                    decoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :],
                                                      encoder_hidden_state[:, 1, :, :]], dim=2)

            attention_weight = torch.zeros(max_length, inputs.size(0))

            pred_line = []

            for i in range(max_length):

                # atten: (batch_size, input_length, 1)
                decoder_output, decoder_hidden_state, atten = s2s.decoder.decode_batch(decoder_input, decoder_hidden_state,
                                                                                       encoder_output,
                                                                                       return_attention_weight=True)
                # (1, 1, vocab_size)
                pred = F.softmax(decoder_output, dim=2)

                # (1, 1)
                decoder_input = torch.argmax(pred, dim=2)

                if tgt_vocab.get_token(decoder_input[0, 0].item()) == tgt_vocab.end_token:

                    break

                pred_line.append(tgt_vocab.get_token(decoder_input[0, 0].item()))

                atten = atten.view(-1)

                attention_weight[i] = atten

            attention_weight = attention_weight[:len(pred_line)]

            plot_attention(attention_weight, line.split(), pred_line, args.picture_path)

            print(line)
            print(" ".join(pred_line))
            print()
