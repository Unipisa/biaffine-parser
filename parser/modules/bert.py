# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoConfig

from .scalar_mix import ScalarMix
from .dropout import TokenDropout

class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False,
                 mask_token_id=0, token_dropout=0.0, mix_dropout=0.0, use_hidden_states=True):
        """
        :param model: path or name of the pretrained model.
        :param n_layers: number of layers from the model to use.
        If 0, use all layers.
        :param n_out: the requested size of the embeddings.
        If 0, use the size of the pretrained embedding model
        :param token_dropout: replace input wordpieces with [MASK] with this probability.
        Not done if parameter is 0.0.
        :param requires_grad: whether to fine tune the embeddings.
        :param mask_token_id: the value of the [MASK] token to use for dropped tokens.
        :param mix_dropout: drop layers with this probability when comuting their
        weighted average with ScalarMix.
        :param use_hidden_states: use the output hidden states from bert if True, or else
        the outputs.
        """

        super(BertEmbedding, self).__init__()

        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(model, config=config)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers if n_layers else self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out if n_out else self.hidden_size
        self.pad_index = config.pad_token_id
        self.requires_grad = requires_grad
        self.use_hidden_states = use_hidden_states
        self.mask_token_id = mask_token_id

        self.token_dropout = TokenDropout(token_dropout, mask_token_id) if token_dropout else None
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        if self.hidden_size != self.n_out:
            self.projection = nn.Linear(self.hidden_size, self.n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"
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
        # return the hidden states of all layers
        outputs = self.bert(subwords, attention_mask=bert_mask.float()) # float for XLNET
        if self.use_hidden_states:
            bert = outputs[-1]
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
        embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed

