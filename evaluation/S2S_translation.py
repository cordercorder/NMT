import torch
from utils.combine_bidir_state import combine_bidir_hidden_state
from models import S2S_attention, S2S_basic, transformer
from utils.Hypothesis import Hypothesis
from utils.Vocab import Vocab
from typing import List
import torch.nn.functional as F
import copy


def decode_batch(s2s: S2S_attention.S2S or S2S_basic.S2S, decoder_input: torch.tensor,
                 decoder_hidden_state: torch.tensor, encoder_output: torch.tensor):
    if isinstance(s2s, S2S_attention.S2S):
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state, encoder_output)
    else:
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state)


def greedy_decoding(s2s: S2S_basic.S2S or S2S_attention.S2S or transformer.S2S, line: str,
                    src_vocab: Vocab, tgt_vocab:Vocab, device: torch.device, is_transformer: bool=False):

    if is_transformer:
        return greedy_decoding_transformer(s2s, line, src_vocab, tgt_vocab, device)

    return greedy_decoding_rnn(s2s, line, src_vocab, tgt_vocab, device)


@torch.no_grad()
def greedy_decoding_transformer(s2s: transformer.S2S, line: str, src_vocab: Vocab, tgt_vocab: Vocab,
                                device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # src: (input_length,)
    src = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

    # src: (1, input_length)
    src = src.unsqueeze(0)
    src_mask = s2s.make_src_mask(src)

    encoder_src = s2s.encoder(src, src_mask)

    tgt = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

    pred_list = []
    max_length = src.size(1) * 3

    for i in range(max_length):

        # tgt: (1, i + 1)
        tgt_mask = s2s.make_tgt_mask(tgt)

        # output: (batch_size, input_length, vocab_size)
        output = s2s.decoder(tgt, encoder_src, tgt_mask, src_mask)
        # output: (batch_size, vocab_size)
        output = output[:, -1, :]

        # pred: (batch_size, )
        pred = torch.argmax(output, dim=-1)

        if tgt_vocab.get_token(pred.item()) == tgt_vocab.end_token:
            break

        tgt = torch.cat([tgt, pred.unsqueeze(1)], dim=1)

        pred_list.append(pred.item())

    pred_line = [tgt_vocab.get_token(index) for index in pred_list]
    return pred_line


@torch.no_grad()
def greedy_decoding_rnn(s2s: S2S_basic.S2S or S2S_attention.S2S, line: str, src_vocab: Vocab,
                        tgt_vocab: Vocab, device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # inputs: (input_length,)
    inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

    # inputs: (input_length, 1)
    inputs = inputs.view(-1, 1)

    encoder_output, encoder_hidden_state = s2s.encoder(inputs)

    decoder_hidden_state = combine_bidir_hidden_state(s2s, encoder_hidden_state)

    decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

    max_length = inputs.size(0) * 3

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
def beam_search_decoding(s2s: S2S_basic.S2S or S2S_attention.S2S, line: str, src_vocab: Vocab,
                         tgt_vocab: Vocab, beam_size: int, device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # print(line)
    # inputs: (input_length,)
    inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

    max_length = inputs.size(0) * 3

    # inputs: (input_length, 1)
    inputs = inputs.view(-1, 1)

    # encoder_output: (input_length, 1, num_directions * hidden_size)
    # encoder_hidden_state: (num_layers * num_directions, 1, hidden_size)
    encoder_output, encoder_hidden_state = s2s.encoder(inputs)

    # decoder_input: (1, 1)
    decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

    decoder_hidden_state = combine_bidir_hidden_state(s2s, encoder_hidden_state)

    hypothesis_list = [Hypothesis(decoder_hidden_state, decoder_input)]

    complete_hypothesis_list = []

    pred_list = [None] * beam_size
    decoder_hidden_state_list = [None] * beam_size

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

            decoder_output, decoder_hidden_state = decode_batch(s2s, decoder_input, decoder_hidden_state,
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

        new_hypothesis_list = new_hypothesis_list[:beam_size]

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

    return pred_line


@torch.no_grad()
def beam_search_rnn(s2s: S2S_attention.S2S or S2S_basic.S2S, line: str, src_vocab: Vocab, tgt_vocab: Vocab,
                    beam_size: int, device: torch.device):

    # batch_size == beam_size

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # inputs: (input_length, beam_size)
    inputs = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.expand(-1, beam_size)

    encoder_output, encoder_hidden_state = s2s.encoder(inputs)

    # decoder_input: (1, beam_size)
    decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)
    decoder_input = decoder_input.expand(-1, beam_size)

    # decoder_hidden_state: (num_layers, beam_size, hidden_size)
    decoder_hidden_state = combine_bidir_hidden_state(s2s, encoder_hidden_state)

    max_length = inputs.size(0) * 3

    scores = torch.zeros(beam_size, device=device)

    complete_seqs = []
    complete_seqs_scores = []
    step = 1

    while step <= max_length:

        # output: (1, beam_size, vocab_size)
        # decoder_hidden_state: (num_layers, beam_size, hidden_size)
        output, decoder_hidden_state = decode_batch(s2s, decoder_input[-1].unsqueeze(0),
                                                    decoder_hidden_state, encoder_output)

        output = F.log_softmax(output, dim=-1)

        # sub_sentence_scores: (beam_size, vocab_size)
        sub_sentence_scores = scores.unsqueeze(1) + output.squeeze(0)

        if step == 1:
            pred_prob, pred_indices = sub_sentence_scores[0].topk(beam_size, dim=-1)
        else:
            # sub_sentence_scores: (beam_size * vocab_size)
            sub_sentence_scores = sub_sentence_scores.view(-1)
            pred_prob, pred_indices = sub_sentence_scores.topk(beam_size, dim=-1)

        # beam_id: (beam_size, )
        beam_id = pred_indices.floor_divide(len(tgt_vocab))

        # token_id: (beam_size, )
        token_id = pred_indices % len(tgt_vocab)

        # decoder_input[-1][beam_id]: (beam_size, )
        # next_decoder_input: (step + 1, beam_size)
        # decoder_input: (step, beam_size)
        next_decoder_input = torch.cat([decoder_input[:, beam_id], token_id.unsqueeze(0)], dim=0)

        complete_indices = []

        for i, indices in enumerate(token_id):

            if tgt_vocab.get_token(indices.item()) == tgt_vocab.end_token:
                complete_indices.append(i)

        if len(complete_indices) > 0:
            complete_seqs.extend(next_decoder_input[:, complete_indices].tolist())

            complete_pred_indices = beam_id[complete_indices] * len(tgt_vocab) + token_id[complete_indices]

            if step == 1:
                complete_seqs_scores.extend(sub_sentence_scores[0][complete_pred_indices].tolist())

                if len(complete_pred_indices) == beam_size:
                    break

                sub_sentence_scores[0][complete_pred_indices] = -1e9
                pred_prob, pred_indices = sub_sentence_scores[0].topk(beam_size, dim=-1)
            else:
                complete_seqs_scores.extend(sub_sentence_scores[complete_pred_indices].tolist())

                if len(complete_pred_indices) == beam_size:
                    break

                sub_sentence_scores[complete_pred_indices] = -1e9
                pred_prob, pred_indices = sub_sentence_scores.topk(beam_size, dim=-1)

            beam_id = pred_indices.floor_divide(len(tgt_vocab))
            token_id = pred_indices % len(tgt_vocab)

            next_decoder_input = torch.cat([decoder_input[:, beam_id], token_id.unsqueeze(0)], dim=0)

        step += 1

        if isinstance(decoder_hidden_state, tuple):
            h, c = decoder_hidden_state
            h = h[:, beam_id]
            c = c[:, beam_id]
            decoder_hidden_state = (h, c)
        else:
            decoder_hidden_state = decoder_hidden_state[:, beam_id]

        decoder_input = next_decoder_input
        scores = pred_prob

    best_sentence_id = 0
    for i in range(len(complete_seqs_scores)):
        if complete_seqs_scores[i] > complete_seqs_scores[best_sentence_id]:
            best_sentence_id = i

    best_sentence = complete_seqs[best_sentence_id]

    best_sentence = [tgt_vocab.get_token(index) for index in best_sentence[1:-1]]

    return best_sentence


@torch.no_grad()
def beam_search_transformer(s2s: transformer.S2S, line: str, src_vocab: Vocab, tgt_vocab: Vocab, beam_size: int,
                            device: torch.device):

    line = " ".join([src_vocab.start_token, line, src_vocab.end_token])

    # src: (input_length,)
    src = torch.tensor([src_vocab.get_index(token) for token in line.split()], device=device)

    # src: (1, input_length)
    src = src.unsqueeze(0)
    src = src.expand(beam_size, -1)
    src_mask = s2s.make_src_mask(src)

    encoder_src = s2s.encoder(src, src_mask)

    max_length = src.size(1) * 3

    # tgt: (1, 1)
    tgt = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)

    # tgt: (beam_size, 1)
    tgt = tgt.expand(beam_size, -1)
    scores = torch.zeros(beam_size, device=device)

    complete_seqs = []
    complete_seqs_scores = []

    step = 1

    while step <= max_length:

        tgt_mask = s2s.make_tgt_mask(tgt)

        # output: (1 * beam_size, input_length, vocab_size)
        output = s2s.decoder(tgt, encoder_src, tgt_mask, src_mask)

        # output: (1 * beam_size, vocab_size)
        output = output[:, -1, :]

        # output: (1 * beam_size, vocab_size)
        output = F.log_softmax(output, dim=-1)

        # sub_sentence_scores: (1 * beam_size, vocab_size)
        sub_sentence_scores = output + scores.unsqueeze(1)

        if step == 1:
            pred_prob, pred_indices = sub_sentence_scores[0].topk(beam_size, dim=-1)
        else:
            # sub_sentence_scores: (beam_size * vocab_size)
            sub_sentence_scores = sub_sentence_scores.view(-1)
            pred_prob, pred_indices = sub_sentence_scores.topk(beam_size, dim=-1)

        # beam_id: (beam_size, )
        beam_id = pred_indices.floor_divide(len(tgt_vocab))
        # token_id: (beam_size, )
        token_id = pred_indices % len(tgt_vocab)

        # next_tgt: (beam_size, input_length + 1)
        next_tgt = torch.cat([tgt[beam_id], token_id.unsqueeze(1)], dim=1)

        complete_indices = []

        for i, indices in enumerate(token_id):

            if tgt_vocab.get_token(indices.item()) == tgt_vocab.end_token:
                complete_indices.append(i)

        if len(complete_indices) > 0:
            complete_seqs.extend(next_tgt[complete_indices].tolist())

            complete_pred_indices = beam_id[complete_indices] * len(tgt_vocab) + token_id[complete_indices]

            if step == 1:
                complete_seqs_scores.extend(sub_sentence_scores[0][complete_pred_indices].tolist())

                if len(complete_indices) == beam_size:
                    break

                sub_sentence_scores[0][complete_pred_indices] = -1e9
                pred_prob, pred_indices = sub_sentence_scores[0].topk(beam_size, dim=-1)
            else:
                complete_seqs_scores.extend(sub_sentence_scores[complete_pred_indices].tolist())

                if len(complete_indices) == beam_size:
                    break

                sub_sentence_scores[complete_pred_indices] = -1e9
                pred_prob, pred_indices = sub_sentence_scores.topk(beam_size, dim=-1)

            # beam_id: (beam_size, )
            beam_id = pred_indices.floor_divide(len(tgt_vocab))
            # token_id: (beam_size, )
            token_id = pred_indices % len(tgt_vocab)
            # next_tgt: (beam_size, input_length + 1)
            next_tgt = torch.cat([tgt[beam_id], token_id.unsqueeze(1)], dim=1)

        step += 1

        tgt = next_tgt
        scores = pred_prob

    best_sentence_id = 0
    for i in range(len(complete_seqs_scores)):
        if complete_seqs_scores[i] > complete_seqs_scores[best_sentence_id]:
            best_sentence_id = i

    best_sentence = complete_seqs[best_sentence_id]

    best_sentence = [tgt_vocab.get_token(index) for index in best_sentence[1:-1]]

    return best_sentence
