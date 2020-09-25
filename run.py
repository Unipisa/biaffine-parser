# -*- coding: utf-8 -*-

import torch
import argparse
import os
from parser.cmds import Evaluate, Predict, Train
from parser.utils.logging import init_logger, logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--file', '-f', default='exp/ptb',
                               help='path to saved model files')
        subparser.add_argument('--preprocess', '-p', action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--device', '-d', default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=16, type=int,
                               help='max num of threads')
        subparser.add_argument('--tree', action='store_true',
                               help='whether to ensure well-formedness')
        subparser.add_argument('--proj', action='store_true',
                               help='whether to projectivise the data')
        subparser.add_argument('--verbose', action='store_false',
                               help='print verbose logs')
    args = parser.parse_args()

    init_logger(logger, verbose=args.verbose)
    logger.info(f"Set the max num of threads to {args.threads}")
    logger.info(f"Set the seed for generating random numbers to {args.seed}")
    logger.info(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.fields = os.path.join(args.file, 'fields')
    args.model = os.path.join(args.file, 'model')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
