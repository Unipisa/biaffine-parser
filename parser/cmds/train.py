# -*- coding: utf-8 -*-

import math
from datetime import datetime, timedelta
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.metric import Metric

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

class Train(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--ftrain', default='data/ptb/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/ptb/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/ptb/test.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/glove.6B.100d.txt',
                               help='path to pretrained embeddings')
        subparser.add_argument('--lower', action='store_true', # Attardi
                               help='whether to turn words to lowercase')
        subparser.add_argument('--unk', default='unk',
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--max-sent-length', default=512, type=int,
                               help='max tokenized sentence length (longer ones are discarded)')

        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        train = Corpus.load(args.ftrain, self.fields, args.max_sent_length)
        dev = Corpus.load(args.fdev, self.fields, args.max_sent_length)
        test = Corpus.load(args.ftest, self.fields, args.max_sent_length)

        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(train.buckets)} buckets")

        print("Create the model")
        self.model = Model(args, mask_token_id=self.FEAT.mask_token_id)
        if self.WORD:
            self.model.load_pretrained(self.WORD.embed)
        print(f"{self.model}\n")
        self.model = self.model.to(args.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        if args.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(),
                                   args.lr,
                                   (args.mu, args.nu),
                                   args.epsilon,
                                   args.decay)
            training_steps = len(train.loader) // self.args.accumulation_steps \
                             * self.args.epochs
            warmup_steps = math.ceil(training_steps * self.args.warmup_steps_ratio)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=training_steps)
        else:
            self.optimizer = Adam(self.model.parameters(),
                                  args.lr,
                                  (args.mu, args.nu),
                                  args.epsilon)
            self.scheduler = ExponentialLR(self.optimizer,
                                           args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            print(f"Epoch {epoch} / {args.epochs}:")
            loss, train_metric = self.train(train.loader)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev.loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test.loader)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric and epoch > args.patience//10:
                best_e, best_metric = epoch, dev_metric
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
        self.model = Model.load(args.model)
        loss, metric = self.evaluate(test.loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
