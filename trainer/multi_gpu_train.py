import sys
sys.path.append("../")

import argparse
import torch
import torch.nn as nn
from models import S2S_basic
from models import S2S_attention
from utils.data_loader import load_corpus_data, NMTDataset, collate
from torch.utils.data import DataLoader
from utils.process import sort_src_sentence_by_length, save_model
import random
import time
import math
import os


parser = argparse.ArgumentParser()

parser.add_argument("--device_id", nargs="+")
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

# use attention or not
parser.add_argument("--attention_size", type=int)

parser.add_argument("--epoch", default=10)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--bidirectional", default=True, type=bool)
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--start_token", default="<s>")
parser.add_argument("--end_token", default="<e>")
parser.add_argument("--unk", default="UNK")
parser.add_argument("--threshold", default=0, type=int)
parser.add_argument("--save_model_steps", default=0.3, type=float)
parser.add_argument("--teacher_forcing_ratio", default=0.5, type=float)
parser.add_argument("--mask_token", default="<mask>")

args, unknown = parser.parse_known_args()

src_data, src_vocab = load_corpus_data(args.src_path, args.src_language, args.start_token, args.end_token,
                                       args.mask_token, args.src_vocab_path, args.unk, args.threshold)

tgt_data, tgt_vocab = load_corpus_data(args.tgt_path, args.tgt_language, args.start_token, args.end_token,
                                       args.mask_token, args.tgt_vocab_path, args.unk, args.threshold)

print("Source language vocab size: {}".format(len(src_vocab)))
print("Target language vocab size: {}".format(len(tgt_vocab)))

os.environ["CUDA_VISIBLE_DEVICES"]=",".join(str(idx) for idx in args.device_id)

torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
device = torch.device("cuda", local_rank)

print(local_rank)

assert len(src_data) == len(tgt_data)

src_data, tgt_data = sort_src_sentence_by_length(list(zip(src_data, tgt_data)))

if args.attention_size:

    print("Attention Model")
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
    print("Basic Model")
    encoder = S2S_basic.Encoder(args.rnn_type, len(src_vocab), args.embedding_size, args.hidden_size, args.num_layers,
                                args.dropout, args.bidirectional)

    decoder = S2S_basic.Decoder(args.rnn_type, len(tgt_vocab), args.embedding_size,
                                2 * args.hidden_size if args.bidirectional else args.hidden_size,
                                args.num_layers, args.dropout)

    s2s = S2S_basic.S2S(encoder, decoder).to(device)

s2s = nn.parallel.DistributedDataParallel(s2s, device_ids=[local_rank])

print("Multi Gpu training: {}".format(local_rank))

optimizer = torch.optim.Adam(s2s.parameters(), args.learning_rate)

criterion = nn.CrossEntropyLoss(reduction="none")

padding_value = src_vocab.get_index(args.mask_token)

assert padding_value == tgt_vocab.get_index(args.mask_token)

train_data = NMTDataset(src_data, tgt_data)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

train_loader = DataLoader(train_data, args.batch_size, shuffle=False, sampler=train_sampler,
                          collate_fn=lambda batch: collate(batch, padding_value, device), drop_last=True)

STEPS = len(range(0, len(src_data), args.batch_size))
save_model_steps = max(int(STEPS * args.save_model_steps), 1)

for i in range(args.epoch):

    epoch_loss = 0.0

    start_time = time.time()

    steps = 0

    word_count = 0

    use_teacher_forcing_list = [True if random.random() >= args.teacher_forcing_ratio else False for j in range(STEPS)]

    for j, (input_batch, target_batch) in enumerate(train_loader):

        batch_loss = s2s.module.train_batch(input_batch, target_batch, padding_value, criterion,
                                            use_teacher_forcing_list[j])

        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        steps += 1

        batch_word_count = target_batch.size(0) * target_batch.size(1)

        word_count += batch_word_count

        if steps % save_model_steps == 0:
            if local_rank == 0:
                torch.save(save_model(s2s, args.attention_size), args.checkpoint + "_" + str(i) + "_" + str(steps))
            batch_loss_value = batch_loss.item()
            ppl = math.exp(batch_loss_value / batch_word_count)
            print("Batch loss: {}, batch perplexity: {}, local rank: {}".format(batch_loss_value, ppl, local_rank))


    epoch_loss /= word_count
    if local_rank == 0:
        torch.save(save_model(s2s, args.attention_size), args.checkpoint + "__{}_{:.6f}".format(i, epoch_loss))
    print("Epoch: {}, time: {} seconds, loss: {}, local rank: {}".format(i, time.time() - start_time, epoch_loss,
                                                                         local_rank))