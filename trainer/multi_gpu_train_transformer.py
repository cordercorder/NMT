import logging
import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils.data_loader import load_corpus_data, NMTDataset, collate
from utils.tools import (sort_src_sentence_by_length, save_transformer, load_transformer,
                         setup_seed, get_optimizer, build_transformer)
from utils.Criterion import LabelSmoothingLoss

logging.basicConfig(level=logging.DEBUG)


def train(local_rank, args):

    setup_seed(args.seed)

    rank = args.nr * args.gpus + local_rank

    saved_model_dir, _ = os.path.split(args.checkpoint)

    if not os.path.isdir(saved_model_dir):
        os.makedirs(saved_model_dir)

    src_data, src_vocab = load_corpus_data(args.src_path, args.src_language, args.start_token, args.end_token,
                                           args.mask_token, args.src_vocab_path, args.rebuild_vocab, args.unk,
                                           args.threshold)

    tgt_data, tgt_vocab = load_corpus_data(args.tgt_path, args.tgt_language, args.start_token, args.end_token,
                                           args.mask_token, args.tgt_vocab_path, args.rebuild_vocab, args.unk,
                                           args.threshold)

    args.src_vocab_size = len(src_vocab)
    args.tgt_vocab_size = len(tgt_vocab)

    logging.info("Source language vocab size: {}".format(len(src_vocab)))
    logging.info("Target language vocab size: {}".format(len(tgt_vocab)))

    assert len(src_data) == len(tgt_data)

    if args.sort_sentence_by_length:
        src_data, tgt_data = sort_src_sentence_by_length(list(zip(src_data, tgt_data)))

    logging.info("Transformer")

    max_src_len = max(len(line) for line in src_data)
    max_tgt_len = max(len(line) for line in tgt_data)

    args.max_src_len = max_src_len
    args.max_tgt_len = max_tgt_len

    padding_value = src_vocab.get_index(args.mask_token)

    assert padding_value == tgt_vocab.get_index(args.mask_token)
    args.padding_value = padding_value

    logging.info("Multi GPU training")

    dist.init_process_group(backend="nccl", init_method=args.init_method, rank=rank,
                            world_size=args.world_size)

    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)

    if args.load:

        logging.info("Load existing model from {}".format(args.load))
        s2s, optimizer_state_dict = load_transformer(args, training=True, device=device)
        s2s = nn.parallel.DistributedDataParallel(s2s, device_ids=[local_rank])
        optimizer = get_optimizer(s2s.parameters(), args)
        optimizer.load_state_dict(optimizer_state_dict)

    else:
        logging.info("New model")
        s2s = build_transformer(args, device)
        s2s.init_parameters()
        s2s = nn.parallel.DistributedDataParallel(s2s, device_ids=[local_rank])
        optimizer = get_optimizer(s2s.parameters(), args)

    s2s.train()

    if args.label_smoothing:
        logging.info("Label Smoothing!")
        criterion = LabelSmoothingLoss(args.label_smoothing, padding_value)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=padding_value)

    train_data = NMTDataset(src_data, tgt_data)

    # release cpu memory
    del src_data
    del tgt_data

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=False, sampler=train_sampler, drop_last=True,
                              pin_memory=True, collate_fn=lambda batch: collate(batch, padding_value, batch_first=True))

    for i in range(args.start_epoch, args.end_epoch):

        train_sampler.set_epoch(i)

        epoch_loss = 0.0

        start_time = time.time()

        steps = 0

        for j, (input_batch, target_batch) in enumerate(train_loader):

            if args.update_freq == 1:
                need_update = True
            else:
                need_update = True if (j + 1) % args.update_freq == 0 else False

            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            output = s2s(input_batch, target_batch[:, :-1])
            del input_batch
            output = output.view(-1, output.size(-1))
            target_batch = target_batch[:, 1:].contiguous().view(-1)

            batch_loss = criterion(output, target_batch)
            del target_batch
            del output

            # synchronize all processes
            # Gradient synchronization communications take place during the backward pass and overlap
            # with the backward computation. When the backward() returns, param.grad already contains
            # the synchronized gradient tensor.
            dist.barrier()
            batch_loss.backward()

            if need_update:
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = batch_loss.item()

            epoch_loss += batch_loss

            steps += 1

        if (steps + 1) % args.update_freq != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= steps

        epoch_ppl = math.exp(epoch_loss)

        logging.info("Epoch: {}, time: {} seconds, loss: {}, perplexity: {}, local rank: {}".format(i, time.time() -
                                                                                                    start_time,
                                                                                                    epoch_loss,
                                                                                                    epoch_ppl,
                                                                                                    local_rank))
        if local_rank == 0:
            torch.save(save_transformer(s2s, optimizer, args), "{}_{}_{}".format(args.checkpoint, i, steps))

    torch.save(save_transformer(s2s, optimizer, args), args.checkpoint + "_rank{}".format(local_rank))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_id", nargs="+", type=int)
    parser.add_argument("--nodes", default=1, type=int, help="total number of nodes")
    parser.add_argument("--nr", default=0, type=int, help="rank of current nodes")
    parser.add_argument("--init_method", default="tcp://localhost:12321")

    parser.add_argument("--num_layers", required=True, type=int)
    parser.add_argument("--d_model", required=True, type=int)
    parser.add_argument("--num_heads", required=True, type=int)
    parser.add_argument("--d_ff", required=True, type=int)
    parser.add_argument("--encoder_max_rpe", default=0, type=int)
    parser.add_argument("--decoder_max_rpe", default=0, type=int)

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
    parser.add_argument("--optim_method", choices=["fix_learning_rate", "adam_inverse_sqrt"],
                        default="fix_learning_rate")
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_updates", default=4000, type=int)
    parser.add_argument("--warmup_init_lr", default=1e-7, type=float)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--start_token", default="<s>")
    parser.add_argument("--end_token", default="<e>")
    parser.add_argument("--unk", default="UNK")
    parser.add_argument("--threshold", default=0, type=int)
    parser.add_argument("--mask_token", default="<mask>")
    parser.add_argument("--label_smoothing", default=0.1, type=int)
    parser.add_argument("--share_dec_pro_emb", type=bool, default=True, help="share decoder input and project embedding")

    parser.add_argument("--seed", default=998244353, type=int)
    parser.add_argument("--update_freq", default=1, type=int)

    parser.add_argument("--rebuild_vocab", action="store_true")
    parser.add_argument("--sort_sentence_by_length", action="store_true")

    args, unknown = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in args.device_id)

    args.gpus = len(args.device_id)

    args.world_size = args.gpus * args.nodes
    mp.spawn(train, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    main()
