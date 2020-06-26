import argparse
import torch
import torch.nn as nn
from models import S2S_basic
from utils.data_loader import load_corpus_data, batch_data
import random
import time

parser = argparse.ArgumentParser()

parser.add_argument("--device", required=True)
parser.add_argument("--src_language", required=True)
parser.add_argument("--tgt_language", required=True)
parser.add_argument("--src_path", required=True)
parser.add_argument("--tgt_path", required=True)
parser.add_argument("--src_vocab_path", required=True)
parser.add_argument("--tgt_vocab_path", required=True)
parser.add_argument("--rnn_type", required=True)
parser.add_argument("--embedding_size", required=True, type=int)
parser.add_argument("--hidden_size", required=True, type=int)
parser.add_argument("--num_layers", required=True, type=int)
parser.add_argument("--checkpoint", required=True)

parser.add_argument("--epoch", default=10)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--bidirectional", default=True, type=bool)
parser.add_argument("--dropout", default=0, type=int)
parser.add_argument("--start_token", default="<s>")
parser.add_argument("--end_token", default="<e>")
parser.add_argument("--unk", default="UNK")
parser.add_argument("--threshold", default=0, type=int)
parser.add_argument("--save_model_steps", default=0.1, type=float)

args, unknown = parser.parse_known_args()

device = torch.device(args.device)

src_data, src_vocab = load_corpus_data(args.src_path, args.src_language, args.start_token, args.end_token,
                                       args.src_vocab_path, args.unk, args.threshold)

tgt_data, tgt_vocab = load_corpus_data(args.tgt_path, args.tgt_language, args.start_token, args.end_token,
                                       args.tgt_vocab_path, args.unk, args.threshold)

assert len(src_data) == len(tgt_data)

train_order = list(range(0, len(src_data), args.batch_size))

encoder = S2S_basic.Encoder(args.rnn_type, len(src_vocab), args.embedding_size, args.hidden_size, args.num_layers,
                            args.dropout, args.bidirectional)

decoder = S2S_basic.Decoder(args.rnn_type, len(tgt_vocab), args.embedding_size,
                            2 * args.hidden_size if args.bidirectional else args.hidden_size, args.num_layers,
                            args.dropout)

s2s = S2S_basic.S2S(encoder, decoder).to(device)

optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)

criterion = nn.CrossEntropyLoss(reduction="none")

padding_value = src_vocab.get_index(args.end_token)

assert padding_value == tgt_vocab.get_index(args.end_token)

STEPS = len(train_order)

for i in range(args.epoch):

    random.shuffle(train_order)

    epoch_loss = 0.0

    start_time = time.time()

    steps = 0

    for input_batch, target_batch in batch_data(src_data, tgt_data, train_order, args.batch_size,
                                                padding_value, device):

        # output: (input_length, batch_size, vocab_size)
        # target_batch: (input_length, batch_size)
        output = s2s(input_batch, target_batch, True)

        batch_loss = 0.0

        for j in range(1, target_batch.size(0) - 1):
            # tmp_input_batch: (batch_size, vocab_size)
            # tmp_target_batch: (batch_size, )
            tmp_input_batch = output[j]
            tmp_target_batch = target_batch[j]
            mask = torch.logical_not(tmp_target_batch == padding_value)
            # tmp_loss: (batch_size, )
            tmp_loss = criterion(tmp_input_batch, tmp_target_batch)

            tmp_loss *= mask

            batch_loss += torch.sum(tmp_loss)

        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        steps += 1

        if steps == max(int(STEPS * args.save_model_steps), 1):
            torch.save({"epoch": i, "step": steps, "model_state": s2s.state_dict()}, args.checkpoint + "_" +
                       str(i) + "_" + str(steps))

    torch.save({"epoch": i, "epoch_loss": epoch_loss, "model_state": s2s.state_dict()}, args.checkpoint +
               "__{}_{:.6f}".format(i, epoch_loss))
    print("Epoch: {}, time: {} seconds, loss: {}".format(i, time.time() - start_time, epoch_loss))
