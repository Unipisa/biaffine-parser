# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.common import bos, pad, unk
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import Field, SubwordField
from parser.utils.fn import ispunct, numericalize
from parser.utils.metric import AttachmentMetric
from parser.utils.vocab import FieldVocab

import torch
import torch.nn as nn


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=args.lower)
            if args.feat == 'char':
                self.FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos,
                                         fix_len=args.fix_len, tokenize=list)
            elif args.feat == 'bert':
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
                self.FEAT = SubwordField('bert',
                                         tokenizer=tokenizer,
                                         fix_len=args.fix_len)
                self.bos = self.FEAT.bos and bos
                self.WORD.bos = self.bos # ensure representations of the same length
                if hasattr(tokenizer, 'vocab'):
                    self.FEAT.vocab = tokenizer.vocab
                else:
                    self.FEAT.vocab = FieldVocab(tokenizer.unk_token_id,
                                                 {tokenizer._convert_id_to_token(i): i
                                                  for i in range(len(tokenizer))})
                args.feat_pad_index = self.FEAT.pad_index # so that it is saved correctly. Attardi
            else:
                self.FEAT = Field('tags', bos=self.bos)
            self.ARC = Field('arcs', bos=self.bos, use_vocab=False,
                             fn=numericalize)
            self.REL = Field('rels', bos=self.bos)
            if args.feat == 'bert':
                if args.n_embed:
                    self.fields = CoNLL(FORM=(self.WORD, self.FEAT),
                                        HEAD=self.ARC, DEPREL=self.REL)
                else:
                    self.fields = CoNLL(FORM=self.FEAT,
                                        HEAD=self.ARC, DEPREL=self.REL)
                    self.WORD = None
            elif args.feat == 'char':
                self.fields = CoNLL(FORM=(self.WORD, self.FEAT),
                                    HEAD=self.ARC, DEPREL=self.REL)
            else:
                self.fields = CoNLL(FORM=self.WORD, CPOS=self.FEAT,
                                    HEAD=self.ARC, DEPREL=self.REL)

            train = Corpus.load(args.ftrain, self.fields, max_sent_length=args.max_sent_length)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            if self.WORD:
                self.WORD.build(train, args.min_freq, embed)
            self.FEAT.build(train)
            self.REL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat in ('char', 'bert'):
                if isinstance(self.fields.FORM, tuple):
                    self.WORD, self.FEAT = self.fields.FORM
                else:
                    self.WORD, self.FEAT = None, self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = torch.tensor([i for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(args.device) if self.WORD else []

        if self.WORD:
            args.update({
                'n_words': self.WORD.vocab.n_init,
                'pad_index': self.WORD.pad_index,
                'unk_index': self.WORD.unk_index,
                'bos_index': self.WORD.bos_index,
        })
        args.update({
            'n_feats': len(self.FEAT.vocab),
            'n_rels': len(self.REL.vocab),
            'feat_pad_index': self.FEAT.pad_index,
        })

        print(f"Override the default configs\n{args}")
        print("Features:")
        if self.WORD:
            print(f"   {self.WORD}")
        print(f"   {self.FEAT}\n   {self.ARC}\n   {self.REL}")

    def train(self, loader):
        self.model.train()

        total_loss, metric = 0, AttachmentMetric()
        accumulation_steps = max(1, self.args.accumulation_steps)

        # gradient accumulation attempt:
        # @see https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
        for step, batch in enumerate(loader):
            if self.WORD:
                words, feats, arcs, rels = batch
                mask = words.ne(self.model.pad_index)
            else:
                words, (feats, arcs, rels) = None, batch
                mask = feats[:,:,0].ne(self.model.pad_index)

            # ignore the BOS token at the start of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            if isinstance(self.model, nn.DataParallel) and len(self.model.device_ids) > 1:
                loss = loss.mean()
            loss /= accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            if (step+1) % accumulation_steps == 0:  # Wait for several backward steps
                self.optimizer.step()               # Now we can do an optimizer step
                self.scheduler.step()
                self.optimizer.zero_grad()          # Reset gradients tensors

                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
                # ignore all punctuation if not specified
                if words is not None and not self.args.punct:
                    mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
                if self.args.evaluate_in_training:                 # Evaluate the model when we...
                    metric(arc_preds, rel_preds, arcs, rels, mask) # ...have no gradients accumulated
            total_loss += loss.item()
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for batch in loader:
            if self.WORD:
                words, feats, arcs, rels = batch
                mask = words.ne(self.model.pad_index)
            else:
                feats, arcs, rels = batch
                words = None
                mask = feats[:,:,0].ne(self.model.pad_index)
            # ignore the BOS token at the start of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            # ignore all punctuation if not specified
            if words is not None and self.puncts is not None:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        arcs, rels, probs = [], [], []
        for batch in loader:
            if self.WORD:
                words, feats = batch
                mask = words.ne(self.model.pad_index)
            else:
                feats = batch[0]
                words = None
                mask = feats[:,:,0].ne(self.model.pad_index)
            # ignore the BOS token at the start of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                probs = s_arc.softmax(-1).gather(-1, arc_preds.unsqueeze(-1))
                probs.extend(probs.squeeze(-1)[mask].split(lens))
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        probs = [[round(p, 4) for p in seq.tolist()] for seq in probs]

        return arcs, rels, probs
