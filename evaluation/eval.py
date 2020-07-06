import torch
import torch.nn.functional as F
import argparse
from utils.Vocab import Vocab
from utils.process import normalizeString, load_model
from nltk.translate.bleu_score import corpus_bleu
from utils.Hypothesis import Hypothesis
from models import S2S_attention, S2S_basic
import copy


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

s2s = (load_model(args.load)).to(device)

print("Load from: {}".format(args.load))

print("Output is: {}".format(args.translation_output))

if isinstance(s2s, S2S_basic.S2S):
    print("Basic Model!")

elif isinstance(s2s, S2S_attention.S2S):

    print("Attention Model")
else:
    raise Exception("Error!")


def decode_batch(decoder_input, decoder_hidden_state, encoder_output):
    if isinstance(s2s, S2S_attention.S2S):
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state, encoder_output)
    else:
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state)


src_vocab = Vocab.load(args.src_vocab_path)
tgt_vocab = Vocab.load(args.tgt_vocab_path)

pred_data = []

with open(args.test_tgt_path) as f:

    data = f.read().split("\n")
    tgt_data = [normalizeString(line) for line in data]

with torch.no_grad():

    s2s.eval()

    with open(args.test_src_path) as f:

        data = f.read().split("\n")

    for line in data:

        line = " ".join([src_vocab.start_token, normalizeString(line, to_ascii=False), src_vocab.end_token])

        # print(line)
        # inputs: (input_length,)
        inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

        max_length = (inputs.size(0) -2) * 3

        # inputs: (input_length, 1)
        inputs = inputs.view(-1, 1)

        # encoder_output: (input_length, 1, num_directions * hidden_size)
        # encoder_hidden_state: (num_layers * num_directions, 1, hidden_size)
        encoder_output, encoder_hidden_state = s2s.encoder(inputs)

        # decoder_input: (1, 1)
        decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

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

        hypothesis_list = [Hypothesis(decoder_hidden_state, decoder_input)]

        complete_hypothesis_list = []

        pred_list = [None] * args.beam_size
        decoder_hidden_state_list = [None] * args.beam_size

        # import time
        #
        # start_time = time.time()

        while len(hypothesis_list) > 0:

            new_hypothesis_list = []

            # for optimization, store the result calculating by GPU

            for i, hypothesis in enumerate(hypothesis_list):

                if len(hypothesis) >= max_length:
                    continue

                if tgt_vocab.get_token(hypothesis.decoder_input) == tgt_vocab.end_token:
                    continue

                decoder_input = torch.tensor([[hypothesis.decoder_input]], device=device)

                decoder_hidden_state = hypothesis.decoder_hidden_state

                decoder_output , decoder_hidden_state = decode_batch(decoder_input, decoder_hidden_state,
                                                                     encoder_output)

                decoder_hidden_state_list[i] = decoder_hidden_state
                # pred: (1, 1, vocab_size)
                pred = F.softmax(decoder_output, dim=2)

                # pred: (vocab_size)
                pred = torch.squeeze(pred)

                pred = pred.log2()

                pred_list[i] = pred.tolist()

            for i, hypothesis in enumerate(hypothesis_list):

                if len(hypothesis) >= max_length:
                    complete_hypothesis_list.append((hypothesis.pred_index_list, hypothesis.score))
                    continue

                if tgt_vocab.get_token(hypothesis.decoder_input) == tgt_vocab.end_token:
                    complete_hypothesis_list.append((hypothesis.pred_index_list[:-1], hypothesis.score))
                    continue

                pred = pred_list[i]

                # start_time = time.time()

                for j in range(len(pred)):

                    new_hypothesis = Hypothesis()
                    new_hypothesis.score = hypothesis.score + pred[j]

                    new_hypothesis.decoder_input = j

                    new_hypothesis.prev_hypothesis = hypothesis
                    new_hypothesis.decoder_hidden_state = decoder_hidden_state_list[i]

                    new_hypothesis_list.append(new_hypothesis)

                # print("Time: {} seconds".format(time.time() - start_time))


            new_hypothesis_list.sort(key=lambda item: -item.score)

            new_hypothesis_list = new_hypothesis_list[:args.beam_size]

            for hypothesis in new_hypothesis_list:

                hypothesis.pred_index_list = copy.deepcopy(hypothesis.prev_hypothesis.pred_index_list)

                hypothesis.pred_index_list.append(hypothesis.decoder_input)

            hypothesis_list = new_hypothesis_list

        # print("Time: {} seconds".format(time.time() - start_time))

        max_score_id = 0
        for i in range(len(complete_hypothesis_list)):

            if complete_hypothesis_list[i][1] > complete_hypothesis_list[max_score_id][1]:

                max_score_id = i

        pred_line_index = complete_hypothesis_list[max_score_id][0]

        # print(pred_line_index)
        pred_line = [tgt_vocab.get_token(index) for index in pred_line_index]

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