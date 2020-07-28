import argparse
import torch
from utils.Vocab import Vocab
from utils.tools import load_model, read_data, write_data, load_transformer
from evaluation.S2S_translation import get_initial_decoder_hidden_state, decode_batch
from lang_vec.lang_vec_tools import load_lang_vec
from models import S2S_basic, S2S_attention, transformer
from subprocess import call


@torch.no_grad()
def translate_rnn(line: str, line_number: int, s2s: S2S_basic.S2S or S2S_attention.S2S, src_vocab: Vocab, tgt_vocab: Vocab,
                  lang_vec: dict, device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    line = line.split()

    lang_token = line[1]

    assert lang_token.startswith("<") and lang_token.endswith(">")

    # inputs: (input_length,)
    inputs = torch.tensor([src_vocab.get_index(token) for token in line], device=device)

    # inputs: (input_length, 1)
    inputs = inputs.view(-1, 1)

    if lang_token.startswith("<") and lang_token.endswith(">"):
        # add language vector
        # input_embedding: (input_length, 1, embedding_size)
        # lang_encoding: (embedding_size, )
        lang_encoding = torch.tensor(lang_vec[lang_token], device=device)
        input_embedding = s2s.encoder.embedding(inputs) + lang_encoding
    else:
        input_embedding = s2s.encoder.embedding(inputs)
        print("line {} does not add language embedding".format(line_number))

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


@torch.no_grad()
def translate_transformer(line: str, line_number: int, s2s: transformer.S2S, src_vocab: Vocab,
                          tgt_vocab: Vocab, lang_vec: dict, device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    line = line.split()

    max_length = (len(line) - 2) * 3

    lang_token = line[1]

    # inputs: (input_length, )
    src = torch.tensor([src_vocab.get_index(token) for token in line], device=device)
    # inputs: (1, input_length)
    src = src.view(1, -1)

    src_mask = s2s.make_src_mask(src)

    src = s2s.encoder.token_embedding(src) * s2s.encoder.scale

    # src: (1, input_length, d_model)
    src = s2s.encoder.pos_embedding(src)

    if lang_token.startswith("<") and lang_token.endswith(">"):
        # lang_encoding: (d_model, )
        lang_encoding = torch.tensor(lang_vec[lang_token], device=device)
        src = src + lang_encoding

    else:
        print("line {} does not add language embedding".format(line_number))

    for layer in s2s.encoder.layers:
        src, self_attention = layer(src, src_mask)

    del self_attention

    encoder_src = src

    tgt = None

    pred_line = [tgt_vocab.get_index(tgt_vocab.start_token)]

    for i in range(max_length):

        if tgt is None:
            tgt = torch.tensor([pred_line], device=device)

        tgt_mask = s2s.make_tgt_mask(tgt)

        # output: (1, tgt_input_length, vocab_size)
        output = s2s.decoder(tgt, encoder_src, tgt_mask, src_mask)

        # (1, tgt_input_length)
        pred = torch.argmax(output, dim=-1)[0, -1]

        if tgt_vocab.get_token(pred.item()) == tgt_vocab.end_token:
            break

        tgt = torch.cat([tgt, pred.unsqueeze(0).unsqueeze(1)], dim=1)
        pred_line.append(pred.item())

    pred_line = [tgt_vocab.get_token(index) for index in pred_line[1:]]
    return pred_line


def translate(line: str, line_number: int, s2s: S2S_basic.S2S or S2S_attention.S2S or transformer.S2S, src_vocab: Vocab,
              tgt_vocab: Vocab, lang_vec: dict, device: torch.device):

    if isinstance(s2s, transformer.S2S):
        return translate_transformer(line, line_number, s2s, src_vocab, tgt_vocab, lang_vec, device)
    else:
        return translate_rnn(line, line_number, s2s, src_vocab, tgt_vocab, lang_vec, device)


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
    parser.add_argument("--transformer", action="store_true")

    args, unknown = parser.parse_known_args()

    src_vocab = Vocab.load(args.src_vocab_path)
    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    src_data = read_data(args.test_src_path)

    lang_vec = load_lang_vec(args.lang_vec_path)  # lang_vec: dict

    device = args.device

    print("load from {}".format(args.load))

    if args.transformer:

        max_src_length = max(len(line) for line in src_data) + 2
        max_tgt_length = max_src_length * 3
        padding_value = src_vocab.get_index(src_vocab.mask_token)
        assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

        s2s = load_transformer(args.load, len(src_vocab), max_src_length, len(tgt_vocab), max_tgt_length,
                               padding_value, device=device)

    else:
        s2s = load_model(args.load, device=device)

    s2s.eval()

    pred_data = []
    for i, line in enumerate(src_data):
        pred_data.append(translate(line, i, s2s, src_vocab, tgt_vocab, lang_vec, device))

    write_data(pred_data, args.translation_output)

    pred_data_tok_path = args.translation_output + ".tok"

    tok_command = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(args.translation_output, pred_data_tok_path)
    call(tok_command, shell=True)

    bleu_calculation_command = "perl /data/rrjin/NMT/scripts/multi-bleu.perl {} < {}".format(args.test_tgt_path,
                                                                                             pred_data_tok_path)
    call(bleu_calculation_command, shell=True)


if __name__ == "__main__":
    main()
