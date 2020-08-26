import logging
import argparse
import torch
import torch.nn as nn
from models import S2S_basic
from models import S2S_attention
from utils.data_loader import load_corpus_data, NMTDataset, collate
from torch.utils.data import DataLoader
from utils.tools import sort_src_sentence_by_length, save_model, load_model
import time
import math

logging.basicConfig(level=logging.DEBUG)


def train(args):

    device = torch.device(args.device)

    torch.cuda.set_device(device)

    src_data, src_vocab = load_corpus_data(args.src_path, args.src_language, args.start_token, args.end_token,
                                           args.mask_token, args.src_vocab_path, args.rebuild_vocab, args.unk,
                                           args.threshold)

    tgt_data, tgt_vocab = load_corpus_data(args.tgt_path, args.tgt_language, args.start_token, args.end_token,
                                           args.mask_token, args.tgt_vocab_path, args.rebuild_vocab, args.unk,
                                           args.threshold)

    logging.info("Source language vocab size: {}".format(len(src_vocab)))
    logging.info("Target language vocab size: {}".format(len(tgt_vocab)))


    assert len(src_data) == len(tgt_data)

    if args.sort_sentence_by_length:
        src_data, tgt_data = sort_src_sentence_by_length(list(zip(src_data, tgt_data)))

    if args.load:
        logging.info("Load existing model from {}".format(args.load))
        s2s, optimizer_state_dict = load_model(args.load, training=True, device=device)
        optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)
        optimizer.load_state_dict(optimizer_state_dict)

    else:
        if args.attention_size:

            logging.info("Attention Model")
            encoder = S2S_attention.Encoder(args.rnn_type, len(src_vocab), args.embedding_size, args.hidden_size,
                                            args.num_layers,args.dropout, args.bidirectional)
            attention = S2S_attention.BahdanauAttention(2 * args.hidden_size if args.bidirectional else args.hidden_size,
                                                        args.num_layers * (2 * args.hidden_size if args.bidirectional
                                                                           else args.hidden_size),
                                                        args.attention_size)
            decoder = S2S_attention.AttentionDecoder(args.rnn_type, len(tgt_vocab), args.embedding_size, args.embedding_size +
                                                     (2 * args.hidden_size if args.bidirectional else args.hidden_size),
                                                     2 * args.hidden_size if args.bidirectional else args.hidden_size,
                                                     args.num_layers, attention, args.dropout)
            s2s = S2S_attention.S2S(encoder, decoder).to(device)

        else:
            logging.info("Basic Model")
            encoder = S2S_basic.Encoder(args.rnn_type, len(src_vocab), args.embedding_size, args.hidden_size, args.num_layers,
                                        args.dropout, args.bidirectional)

            decoder = S2S_basic.Decoder(args.rnn_type, len(tgt_vocab), args.embedding_size,
                                        2 * args.hidden_size if args.bidirectional else args.hidden_size,
                                        args.num_layers, args.dropout)

            s2s = S2S_basic.S2S(encoder, decoder).to(device)

        optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)

    s2s.train()

    padding_value = src_vocab.get_index(args.mask_token)

    assert padding_value == tgt_vocab.get_index(args.mask_token)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)

    train_data = NMTDataset(src_data, tgt_data)

    # release cpu memory
    del src_data
    del tgt_data

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=lambda batch: collate(batch, padding_value))

    for i in range(args.start_epoch, args.end_epoch):

        epoch_loss = 0.0

        start_time = time.time()

        steps = 0

        for j, (input_batch, target_batch) in enumerate(train_loader):

            batch_loss = s2s.train_batch(input_batch.to(device, non_blocking=True),
                                         target_batch.to(device, non_blocking=True), criterion, optimizer)

            epoch_loss += batch_loss

            steps += 1

        epoch_loss /= steps

        torch.save(save_model(s2s, optimizer, args), "{}_{}_{}".format(args.checkpoint, i, steps))
        logging.info("Epoch: {}, time: {} seconds, loss: {}".format(i, time.time() - start_time, epoch_loss))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", required=True)
    parser.add_argument("--src_language", required=True)
    parser.add_argument("--tgt_language", required=True)
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--tgt_path", required=True)
    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)
    parser.add_argument("--rnn_type", required=True, choices=["rnn", "RNN", "lstm", "LSTM", "gru", "GRU"])
    parser.add_argument("--embedding_size", required=True, type=int)
    parser.add_argument("--hidden_size", required=True, type=int)
    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)

    # use attention or not
    parser.add_argument("--attention_size", type=int)
    parser.add_argument("--load")

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--start_token", default="<s>")
    parser.add_argument("--end_token", default="<e>")
    parser.add_argument("--unk", default="UNK")
    parser.add_argument("--threshold", default=0, type=int)
    parser.add_argument("--mask_token", default="<mask>")

    parser.add_argument("--rebuild_vocab", action="store_true")
    parser.add_argument("--sort_sentence_by_length", action="store_true")

    args, unknown = parser.parse_known_args()
    train(args)


if __name__ == "__main__":
    main()
