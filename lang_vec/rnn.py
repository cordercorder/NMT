import argparse
import torch
from utils.Vocab import Vocab
from utils.tools import load_model, read_data, write_data
from evaluation.S2S_translation import get_initial_decoder_hidden_state, decode_batch
from lang_vec.lang_vec_tools import load_lang_vec
from subprocess import call


@torch.no_grad()
def translate(line, s2s, src_vocab, tgt_vocab, lang_vec, device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    line = line.split()

    lang_token = line[1]

    assert lang_token.startswith("<") and lang_token.endswith(">")

    # lang_encoding: (embedding_size, )
    lang_encoding = torch.tensor(lang_vec[lang_token], device=device)

    # inputs: (input_length,)
    inputs = torch.tensor([src_vocab.get_index(token) for token in line], device=device)

    # inputs: (input_length, 1)
    inputs = inputs.view(-1, 1)

    # add language vector
    # input_embedding: (input_length, 1, embedding_size)
    input_embedding = s2s.encoder.embedding(inputs) + lang_encoding

    encoder_output, encoder_hidden_state = s2s.encoder.rnn(input_embedding)

    decoder_hidden_state = get_initial_decoder_hidden_state(s2s, encoder_hidden_state)

    decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

    max_length = (inputs.size(0) - 2) * 3

    pred_line = []

    for i in range(max_length):

        # decoder_output: (1, 1, vocab_size)
        # decoder_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
        decoder_output, decoder_hidden_state = decode_batch(s2s, decoder_input, decoder_hidden_state,
                                                            encoder_output)

        # pred: (1, 1)
        pred = torch.argmax(decoder_output, dim=2)

        if tgt_vocab.get_token(pred[0, 0].item()) == tgt_vocab.end_token:
            break

        decoder_input = pred

        pred_line.append(tgt_vocab.get_token(pred[0, 0].item()))

    return pred_line


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)
    parser.add_argument("--test_src_path", required=True)
    parser.add_argument("--test_tgt_path", required=True)
    parser.add_argument("--lang_vec_path", required=True)
    parser.add_argument("--translation_output", required=True)

    args, unknown = parser.parse_known_args()

    src_vocab = Vocab.load(args.src_vocab_path)
    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    src_data = read_data(args.test_src_path)

    lang_vec = load_lang_vec(args.lang_vec_path)  # lang_vec: dict

    device = args.device

    s2s = load_model(args.load, device=device)

    s2s.eval()

    pred_data = []
    for line in src_data:
        pred_data.append(translate(line, s2s, src_vocab, tgt_vocab, lang_vec, device))

    write_data(pred_data, args.translation_output)

    pred_data_tok_path = args.translation_output + ".tok"

    tok_command = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(args.translation_output, pred_data_tok_path)
    call(tok_command, shell=True)

    bleu_calculation_command = "perl /data/rrjin/NMT/scripts/multi-bleu.perl {} < {}".format(args.test_tgt_path,
                                                                                             pred_data_tok_path)
    call(bleu_calculation_command, shell=True)


if __name__ == "__main__":
    main()
