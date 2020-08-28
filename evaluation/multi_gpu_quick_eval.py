import os
import glob
import logging
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data import DataLoader
from typing import List
from subprocess import call

from utils.tools import read_data, write_data, load_transformer, load_model, SrcData
from utils.data_loader import convert_data_to_index, pad_data
from utils.Vocab import Vocab
from evaluation.S2S_translation import greedy_decoding, beam_search_decoding

logging.basicConfig(level=logging.DEBUG)


class DataPartition:

    def __init__(self, data: List[List], num_process: int):

        assert num_process > 0

        self.data = data
        self.num_process = num_process

        # self.partitions List[List[List]]
        self.partitions = []

        block_size = len(self.data) // num_process

        for i in range(num_process):

            if i == 0:
                self.partitions.append(self.data[:block_size])
            else:
                self.partitions.append(self.data[i*block_size: (i+1)*block_size])

        if len(self.data) % num_process != 0:
            self.partitions[-1].extend(self.data[num_process * block_size:])

    def dataset(self, process_id: int):
        return SrcData(self.partitions[process_id])


def evaluation(local_rank, args):
    rank = args.nr * args.gpus + local_rank
    dist.init_process_group(backend="nccl", init_method=args.init_method, rank=rank,
                            world_size=args.world_size)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    src_data = read_data(args.test_src_path)

    max_src_len = max(len(line.split()) for line in src_data) + 2
    max_tgt_len = max_src_len * 3
    logging.info("max src sentence length: {}".format(max_src_len))

    src_vocab = Vocab.load(args.src_vocab_path)
    tgt_vocab = Vocab.load(args.tgt_vocab_path)

    padding_value = src_vocab.get_index(src_vocab.mask_token)

    assert padding_value == tgt_vocab.get_index(tgt_vocab.mask_token)

    src_data = convert_data_to_index(src_data, src_vocab)

    dataset = DataPartition(src_data, args.world_size).dataset(rank)

    logging.info("dataset size: {}, rank: {}".format(len(dataset), rank))

    data_loader = DataLoader(dataset=dataset, batch_size=(args.batch_size if args.batch_size else 1),
                             shuffle=False, pin_memory=True, drop_last=False,
                             collate_fn=lambda batch: pad_data(batch, padding_value,
                                                               batch_first=(True if args.transformer else False)))

    if args.beam_size:
        logging.info("Beam size: {}".format(args.beam_size))

    if args.is_prefix:
        args.model_load = args.model_load + "*"

    for model_path in glob.glob(args.model_load):
        logging.info("Load model from: {}, rank: {}".format(model_path, rank))

        if args.transformer:

            s2s = load_transformer(model_path, len(src_vocab), max_src_len, len(tgt_vocab), max_tgt_len, padding_value,
                                   device=device)

        else:
            s2s = load_model(model_path, device=device)

        s2s.eval()

        if args.record_time:
            import time
            start_time = time.time()

        pred_data = []

        for data in data_loader:
            if args.beam_size:
                pred_data.append(beam_search_decoding(s2s, data.to(device, non_blocking=True), tgt_vocab,
                                                      args.beam_size, device))
            else:
                pred_data.extend(greedy_decoding(s2s, data.to(device, non_blocking=True), tgt_vocab, device))

        if args.record_time:
            end_time = time.time()
            logging.info("Time spend: {} seconds, rank: {}".format(end_time - start_time, rank))

        if not os.path.isdir(args.translation_output_dir):
            os.makedirs(args.translation_output_dir)

        _, model_name = os.path.split(model_path)

        if args.beam_size:
            translation_file_name_prefix = "{}_beam_size{}".format(model_name, args.beam_size)
        else:
            translation_file_name_prefix = model_name

        p = os.path.join(args.translation_output_dir, "{}_translations.rank{}".format(translation_file_name_prefix,
                                                                                      rank))

        write_data(pred_data, p)

        if args.need_tok:

            # replace '@@ ' with ''
            p_tok = os.path.join(args.translation_output_dir,
                                 "{}_translations_tok.rank{}".format(translation_file_name_prefix, rank))

            tok_command = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(p, p_tok)

            call(tok_command, shell=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", nargs="+", type=int)

    parser.add_argument("--nodes", default=1, type=int, help="total number of nodes")
    parser.add_argument("--nr", default=0, type=int, help="rank of current nodes")
    parser.add_argument("--init_method", default="tcp://localhost:12321")

    parser.add_argument("--model_load", required=True)
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--is_prefix", action="store_true")

    parser.add_argument("--test_src_path", required=True)
    parser.add_argument("--test_tgt_path", required=True)
    parser.add_argument("--src_vocab_path", required=True)
    parser.add_argument("--tgt_vocab_path", required=True)

    parser.add_argument("--translation_output_dir", required=True)
    parser.add_argument("--beam_size", type=int)

    parser.add_argument("--record_time", action="store_true")
    parser.add_argument("--need_tok", action="store_true")

    parser.add_argument("--batch_size", type=int)

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in args.device_id)

    args.gpus = len(args.device_id)

    args.world_size = args.gpus * args.nodes

    mp.spawn(evaluation, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    main()
