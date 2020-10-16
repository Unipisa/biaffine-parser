# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
import torch
from torch.cuda import memory_allocated

from .scalar_mix import ScalarMix
from .dropout import TokenDropout

from parser.utils.logging import get_logger

logger = get_logger(__name__)


class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False,
                 mask_token_id=0, token_dropout=0.0, mix_dropout=0.0,
                 use_hidden_states=True, use_attentions=False,
                 attention_head=0, attention_layer=8):
        """
        A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.

        While mainly tailored to provide input preparation and post-processing for the BERT model,
        it is also compatiable with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.

        :param model (str): path or name of the pretrained model.
        :param n_layers: number of layers from the model to use.
        If 0, use all layers.
        :param n_out (int): the requested size of the embeddings.
            If 0, use the size of the pretrained embedding model
        :param token_dropout (float): replace input wordpieces with [MASK] with this probability.
            Not done if parameter is 0.0.
        :param requires_grad: whether to fine tune the embeddings.
        :param mask_token_id: the value of the [MASK] token to use for dropped tokens.
        :param mix_dropout: drop layers with this probability when comuting their
            weighted average with ScalarMix.
        :param use_hidden_states: use the output hidden states from bert if True, or else
            the outputs.
        :param use_attentions: extract attention weights.
        :param attention_head: which attention head to use.
        :param attention_layer: which attention layer weights to return.

    .. _transformers:
        https://github.com/huggingface/transformers
        """

        super().__init__()

        config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                            output_attentions=use_attentions)
        self.bert = AutoModel.from_pretrained(model, config=config)
        self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pad_index = config.pad_token_id
        self.requires_grad = requires_grad
        self.use_hidden_states = use_hidden_states
        self.mask_token_id = mask_token_id
        self.use_attentions = use_attentions
        self.attention_layer = attention_layer
        self.head = attention_head

        self.token_dropout = TokenDropout(token_dropout, mask_token_id) if token_dropout else None
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        if self.hidden_size != self.n_out:
            self.projection = nn.Linear(self.hidden_size, self.n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"bert={self.bert}, "
        s += f"scalar_mix={self.scalar_mix}, "
        if self.use_attentions:
            s += f"use_attentions={self.use_attentions}, "
        if self.hidden_size != self.n_out:
            s += f"projection={self.projection}, "
        s += f"pad_index={self.pad_index}, "
        s += f"mask_token_id={self.mask_token_id}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, subwords):
        batch_size, seq_len, fix_len = subwords.shape
        if self.token_dropout:
            subwords = self.token_dropout(subwords)
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))

        if not self.requires_grad:
            self.bert.eval()
        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), True)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), True)
        if subwords.shape[1] > self.bert.config.max_position_embeddings:
            logger.warn(f"Tokenized sequence is longer than the transformer can handle: "
                        f"({subwords.shape[1]} > {self.bert.config.max_position_embeddings})")
        # return the hidden states of all layers
        # print('<BERT, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        outputs = self.bert(subwords, attention_mask=bert_mask.float()) # float for XLNET
        # print('BERT>, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        if self.use_hidden_states:
            bert = outputs[-2] if self.use_attentions else outputs[-1]
            # [n_layers, batch_size, n_subwords, hidden_size]
            bert = bert[-self.n_layers:]
            # [batch_size, n_subwords, hidden_size]
            bert = self.scalar_mix(bert)
        else:
            bert = outputs[0]
        # [batch_size, n_subwords]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size)
        embed = embed.masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        embed = embed.sum(2) / bert_lens.unsqueeze(-1) # sum wordpieces
        seq_attn = None
        if self.use_attentions:
            # (a list of layers) = [ [batch, num_heads, sent_len, sent_len] ]
            attns = outputs[-1]
            # [batch, n_subwords, n_subwords]
            attn = attns[self.attention_layer][:,self.head,:,:] # layer 9 represents syntax
            # squeeze out multiword tokens
            mask2 = ~mask
            mask2[:,:,0] = True # keep first column
            sub_masks = pad_sequence(mask2[mask].split(lens.tolist()), True)
            seq_mask = torch.einsum('bi,bj->bij', sub_masks, sub_masks) # outer product
            seq_lens = seq_mask.sum((1,2))
            # [batch_size, seq_len, seq_len]
            sub_attn = attn[seq_mask].split(seq_lens.tolist())
            # fill a tensor [batch_size, seq_len, seq_len]
            seq_attn = attn.new_zeros(batch_size, seq_len, seq_len)
            for i, attn_i in enumerate(sub_attn):
                size = sub_masks[i].sum(0)
                attn_i = attn_i.view(size, size)
                # size = min(size, self.n_attentions) # FIXME: n_attentions=1
                seq_attn[i,:size,:size] = attn_i
        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed, seq_attn

