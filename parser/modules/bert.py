# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
import torch

from .scalar_mix import ScalarMix
from .dropout import TokenDropout

class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False,
                 mask_token_id=0, token_dropout=0.0, mix_dropout=0.0,
                 use_hidden_states=True, n_attentions=0, attention_layer=8):
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
        :param n_attentions: attention weights to return.
        """

        super(BertEmbedding, self).__init__()

        config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                            output_attentions=n_attentions!=0)
        self.bert = AutoModel.from_pretrained(model, config=config)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers if n_layers else self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out if n_out else self.hidden_size
        self.pad_index = config.pad_token_id
        self.requires_grad = requires_grad
        self.use_hidden_states = use_hidden_states
        self.mask_token_id = mask_token_id
        self.n_attentions = n_attentions
        self.attention_layer = attention_layer
        self.head = 0

        self.token_dropout = TokenDropout(token_dropout, mask_token_id) if token_dropout else None
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        if self.hidden_size != self.n_out:
            self.projection = nn.Linear(self.hidden_size, self.n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"n_attentions={self.n_attentions}, "
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
        # return the hidden states of all layers
        outputs = self.bert(subwords, attention_mask=bert_mask.float()) # float for XLNET
        if self.use_hidden_states:
            bert = outputs[-2] if self.n_attentions else outputs[-1]
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
        if self.n_attentions:
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
            sub_attn = attn[seq_mask].split(seq_lens.tolist())
            # fill a tensor [batch_size, seq_len, n_attentions]
            seq_attn = attn.new_zeros(batch_size, seq_len, self.n_attentions)
            for i, attn_i in enumerate(sub_attn):
                size = sub_masks[i].sum(0)
                attn_i = attn_i.view(size, size)
                size = min(size, self.n_attentions)
                seq_attn[i,:size,:size] = attn_i[:size, :size]
        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed, seq_attn

