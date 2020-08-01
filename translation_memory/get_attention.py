import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import os
from utils.tools import load_transformer, read_data
from utils.Vocab import Vocab
from models import transformer


def plot_attention(attention, sentence, predicted_sentence, num_heads, num_rows, num_cols, picture_path):
    fig = plt.figure(figsize=(25, 30))
    font_dict = {"fontsize": 12}

    for i in range(num_heads):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        im = ax.matshow(attention[i], cmap="viridis")
        plt.colorbar(im)

        ax.set_xticklabels([""] + sentence, fontdict=font_dict, rotation=90)
        ax.set_yticklabels([""] + predicted_sentence, fontdict=font_dict)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    directory, picture_name = os.path.split(picture_path)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    plt.savefig(picture_path)
    plt.close()


@torch.no_grad()
def translate(s2s: transformer.S2S, line: str, src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # src: (input_length, )
    src = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

    # src: (1, input_length)
    src = src.unsqueeze(0)
    src_mask = s2s.make_src_mask(src)

    encoder_src = s2s.encoder(src, src_mask)

    tgt_list = [tgt_vocab.get_index(tgt_vocab.start_token)]
    decoder_input = None
    encoder_attention = None

    max_length = src.size(1) * 3

    for i in range(max_length):

        if decoder_input is None:
            decoder_input = torch.tensor([tgt_list], device=device)

        tgt_mask = s2s.make_tgt_mask(decoder_input)

        tgt = s2s.decoder.token_embedding(decoder_input) * s2s.decoder.scale
        tgt = s2s.decoder.pos_embedding(tgt)

        # attention: (batch_size, num_heads, input_length1, input_length2)
        for layer in s2s.decoder.layers:
            tgt, self_attention, encoder_attention = layer(tgt, encoder_src, tgt_mask, src_mask, return_attention=True)

        # output: (batch_size, tgt_input_length, tgt_vocab_size)
        output = s2s.decoder.linear(tgt)

        pred = torch.argmax(output, dim=-1)[0, -1]
        tgt_list.append(pred.item())
        if tgt_vocab.get_token(pred.item()) == tgt_vocab.end_token:
            break

        decoder_input = torch.cat([decoder_input, pred.unsqueeze(0).unsqueeze(1)], dim=1)

    pred_line = [tgt_vocab.get_token(index) for index in tgt_list[1:]]
    encoder_attention = encoder_attention.squeeze(0)
    return pred_line, encoder_attention


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--test_src_path", required=True)
    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)
    parser.add_argument("--picture_directory", required=True)

    args, unknown = parser.parse_known_args()

    device = args.device

    src_data = read_data(args.test_src_path)

    src_vocab = Vocab.load(args.src_vocab_path)
    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    max_src_len = max(len(line.split()) for line in src_data) + 2
    max_tgt_len = max_src_len * 3

    padding_value = src_vocab.get_index(src_vocab.mask_token)

    assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

    s2s = load_transformer(args.load, len(src_vocab), max_src_len, len(tgt_vocab), max_tgt_len, padding_value)

    s2s.eval()

    pred_data = []

    for i, line in enumerate(src_data):
        pred_line, attention = translate(s2s, line, src_vocab, tgt_vocab, device)

        picture_path = os.path.join(args.picture_directory, "attention" + str(i) + ".jpg")
        plot_attention(attention, [src_vocab.start_token] + line.split() + [src_vocab.end_token], pred_line,
                       8, 4, 2, picture_path)
        pred_data.append(pred_line)


if __name__ == "__main__":
    main()
