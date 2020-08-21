import logging
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import load_corpus_data, NMTDataset, collate
from utils.tools import sort_src_sentence_by_length, save_transformer, load_transformer
from models import transformer

logging.basicConfig(level=logging.DEBUG)


def train(args):

    device = args.device

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

    logging.info("Transformer")

    max_src_len = max(len(line) for line in src_data)
    max_tgt_len = max(len(line) for line in tgt_data)

    padding_value = src_vocab.get_index(args.mask_token)

    assert padding_value == tgt_vocab.get_index(args.mask_token)

    if args.load:

        logging.info("Load existing model from {}".format(args.load))
        s2s, optimizer_state_dict = load_transformer(args.load, len(src_vocab), max_src_len, len(tgt_vocab),
                                                     max_tgt_len, padding_value, training=True, device=device)
        optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)
        optimizer.load_state_dict(optimizer_state_dict)

    else:
        logging.info("New model")
        encoder = transformer.Encoder(len(src_vocab), max_src_len, args.d_model, args.num_layers, args.num_heads,
                                      args.d_ff, args.dropout, device)

        decoder = transformer.Decoder(len(tgt_vocab), max_tgt_len, args.d_model, args.num_layers, args.num_heads,
                                      args.d_ff, args.dropout, device)

        s2s = transformer.S2S(encoder, decoder, padding_value, device).to(device)

        s2s.init_parameters()

        optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)

    s2s.train()

    criterion = nn.CrossEntropyLoss(ignore_index=padding_value)

    train_data = NMTDataset(src_data, tgt_data)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=lambda batch: collate(batch, padding_value, batch_first=True))

    STEPS = len(range(0, len(src_data), args.batch_size))
    save_model_steps = max(int(STEPS * args.save_model_steps), 1)

    for i in range(args.start_epoch, args.end_epoch):

        epoch_loss = 0.0

        start_time = time.time()

        steps = 0

        for j, (input_batch, target_batch) in enumerate(train_loader):

            batch_loss = s2s.train_batch(input_batch.to(device, non_blocking=True),
                                         target_batch.to(device, non_blocking=True),
                                         criterion, optimizer)

            epoch_loss += batch_loss

            steps += 1

            if steps % save_model_steps == 0:
                torch.save(save_transformer(s2s, optimizer, args), args.checkpoint + "_" + str(i) + "_" + str(steps))
                ppl = math.exp(batch_loss)
                logging.info("Batch loss: {}, batch perplexity: {}".format(batch_loss, ppl))

        epoch_loss /= steps

        epoch_ppl = math.exp(epoch_loss)

        torch.save(save_transformer(s2s, optimizer, args), args.checkpoint + "__{}_{:.6f}".format(i, epoch_loss))
        logging.info("Epoch: {}, time: {} seconds, loss: {}, perplexity: {}".format(i, time.time() - start_time,
                                                                                    epoch_loss,
                                                                                    epoch_ppl))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", required=True)
    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--d_model", required=True, type=int)
    parser.add_argument("--num_heads", required=True, type=int)
    parser.add_argument("--d_ff", required=True, type=int)

    parser.add_argument("--src_language", required=True)
    parser.add_argument("--tgt_language", required=True)
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--tgt_path", required=True)
    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--load")

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--start_token", default="<s>")
    parser.add_argument("--end_token", default="<e>")
    parser.add_argument("--unk", default="UNK")
    parser.add_argument("--threshold", default=0, type=int)
    parser.add_argument("--save_model_steps", default=0.3, type=float)
    parser.add_argument("--mask_token", default="<mask>")

    parser.add_argument("--rebuild_vocab", action="store_true")
    parser.add_argument("--sort_sentence_by_length", action="store_true")

    args, unknown = parser.parse_known_args()

    train(args)


if __name__ == "__main__":
    main()
