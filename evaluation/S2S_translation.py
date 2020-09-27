import torch
import torch.nn.functional as F

from typing import List

from utils.combine_bidir_state import combine_bidir_hidden_state
from models import S2S_attention, S2S_basic, transformer
from utils.Vocab import Vocab


def decode_batch(s2s: S2S_attention.S2S or S2S_basic.S2S, decoder_input: torch.tensor,
                 decoder_hidden_state: torch.tensor, encoder_output: torch.tensor):
    if isinstance(s2s, S2S_attention.S2S):
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state, encoder_output)
    else:
        return s2s.decoder.decode_batch(decoder_input, decoder_hidden_state)


@torch.no_grad()
def greedy_decoding(s2s: S2S_basic.S2S or S2S_attention.S2S or transformer.S2S, data_tensor: torch.tensor,
                    tgt_vocab: Vocab, device: torch.device, tgt_prefix: List[str] = None):

    if isinstance(s2s, transformer.S2S):
        return greedy_decoding_transformer(s2s, data_tensor, tgt_vocab, device, tgt_prefix)

    return greedy_decoding_rnn(s2s, data_tensor, tgt_vocab, device)


def convert_index_to_token(pred_list: List[List], tgt_vocab: Vocab, batch_size: int, end_token_index: int):

    # pred_line: List[List] (tgt_length, batch_size)
    pred_line = []

    for j in range(batch_size):
        line = []
        for i in range(len(pred_list)):
            if pred_list[i][j] == end_token_index:
                break
            line.append(tgt_vocab.get_token(pred_list[i][j]))
        pred_line.append(line)

    return pred_line


@torch.no_grad()
def greedy_decoding_transformer(s2s: transformer.S2S, data_tensor: torch.tensor, tgt_vocab: Vocab,
                                device: torch.device, tgt_prefix: List[str] = None):

    # src: (batch_size, input_length)
    src = data_tensor
    src_mask = s2s.make_src_mask(src)

    batch_size = src.size(0)

    encoder_src = s2s.encoder(src, src_mask)

    tgt = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)
    tgt = tgt.expand(batch_size, -1)

    # pred_list: List[List] (tgt_length, batch_size)
    pred_list = []

    if tgt_prefix is not None:
        # tgt_prefix_tensor: (batch_size, )
        tgt_prefix = [tgt_vocab.get_index(prefix_token) for prefix_token in tgt_prefix]
        tgt_prefix_tensor = torch.tensor(tgt_prefix, device=device)
        # tgt_prefix_tensor: (batch_size, 1)
        tgt_prefix_tensor = tgt_prefix_tensor.unsqueeze(1)
        # tgt: (batch_size, 2)
        tgt = torch.cat([tgt, tgt_prefix_tensor], dim=1)
        pred_list.append(tgt_prefix)

    max_length = src.size(1) * 3

    end_token_index = tgt_vocab.get_index(tgt_vocab.end_token)

    for i in range(0 if tgt_prefix is None else 1, max_length):

        # tgt: (batch_size, i + 1)
        tgt_mask = s2s.make_tgt_mask(tgt)

        # output: (batch_size, input_length, vocab_size)
        output = s2s.decoder(tgt, encoder_src, tgt_mask, src_mask)
        # output: (batch_size, vocab_size)
        output = output[:, -1, :]

        # pred: (batch_size, )
        pred = torch.argmax(output, dim=-1)

        if torch.all(pred == end_token_index).item():
            break

        tgt = torch.cat([tgt, pred.unsqueeze(1)], dim=1)

        pred_list.append(pred.tolist())

    return convert_index_to_token(pred_list, tgt_vocab, batch_size, end_token_index)


@torch.no_grad()
def greedy_decoding_rnn(s2s: S2S_basic.S2S or S2S_attention.S2S, data_tensor: torch.tensor,
                        tgt_vocab: Vocab, device: torch.device):

    # inputs: (input_length, batch_size)
    inputs = data_tensor

    batch_size = inputs.size(1)

    encoder_output, encoder_hidden_state = s2s.encoder(inputs)

    decoder_hidden_state = combine_bidir_hidden_state(s2s, encoder_hidden_state)

    decoder_input = torch.tensor([[tgt_vocab.get_index(tgt_vocab.start_token)]], device=device)
    decoder_input = decoder_input.expand(-1, batch_size)

    max_length = inputs.size(0) * 3

    # pred_list: List[List] (tgt_length, batch_size)
    pred_list = []

    end_token_index = tgt_vocab.get_index(tgt_vocab.end_token)

    for i in range(max_length):

        # decoder_output: (1, batch_size, vocab_size)
        # decoder_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
        decoder_output, decoder_hidden_state = decode_batch(s2s, decoder_input, decoder_hidden_state,
                                                            encoder_output)

        # pred: (1, batch_size)
        pred = torch.argmax(decoder_output, dim=2)

        if torch.all(pred == end_token_index).item():
            break

        decoder_input = pred

        pred_list.append(pred.squeeze(0).tolist())

    return convert_index_to_token(pred_list, tgt_vocab, batch_size, end_token_index)


@torch.no_grad()
def beam_search_decoding(s2s: S2S_basic.S2S or S2S_attention.S2S or transformer.S2S, data_tensor: torch.tensor,
                         tgt_vocab: Vocab, beam_size: int, device: torch.device):
    if isinstance(s2s, transformer.S2S):
        return beam_search_transformer(s2s, data_tensor, tgt_vocab, beam_size, device)

    else:
        return beam_search_rnn(s2s, data_tensor, tgt_vocab, beam_size, device)


@torch.no_grad()
def beam_search_rnn(s2s: S2S_attention.S2S or S2S_basic.S2S, data_tensor: torch.tensor, tgt_vocab: Vocab,
                    beam_size: int, device: torch.device):

    # batch_size == beam_size

    # inputs: (input_length, beam_size)
    inputs = data_tensor
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

    while True:

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

        if step == max_length:
            complete_seqs.extend(next_decoder_input.t().tolist())
            complete_seqs_scores.extend(pred_prob.tolist())
            break

        complete_indices = []

        for i, indices in enumerate(token_id):

            if tgt_vocab.get_token(indices.item()) == tgt_vocab.end_token:
                complete_indices.append(i)

        if len(complete_indices) > 0:
            complete_seqs.extend(next_decoder_input[:, complete_indices].t().tolist())

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
def beam_search_transformer(s2s: transformer.S2S, data_tensor: torch.tensor, tgt_vocab: Vocab, beam_size: int,
                            device: torch.device):

    # src: (1, input_length)
    src = data_tensor
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

    while True:

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

        if step == max_length:
            complete_seqs.extend(next_tgt.tolist())
            complete_seqs_scores.extend(pred_prob.tolist())
            break

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
