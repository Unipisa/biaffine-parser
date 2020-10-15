# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from parser.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout
from parser.utils.alg import eisner
from parser.utils.fn import istree

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, Optional, Tuple, Any, List
import torch.nn.functional as F

class Model(nn.Module):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_feats (int):
            The size of the feat vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        feat (str):
            Specifies which type of additional feature to use: ``'char'`` | ``'bert'`` | ``'tag'``.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            ``'tag'``: POS tag embeddings.
            Default: ``'char'``.
        n_word_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        bert_fine_tune (bool):
            Weather to fine tune the BERT model.
            Deafult: False.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        token_dropout (float):
            The dropout ratio of tokens. Default: .0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_arc (int):
            Arc MLP size. Default: 500.
        n_mlp_rel  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        use_hidden_states (bool):
            Wethre to use hidden states rather than outputs from BERT.
            Default: True.
        use_attentions (bool):
            Wethre to use attention heads from BERT.
            Default: False.
        attention_head (int):
            Which attention head from BERT to use. Default: 0.
        attention_layer (int):
            Which attention layer from BERT to use; use all if 0. Default: 6.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, args, mask_token_id=0):
        super().__init__()

        self.args = args
        if args.n_embed:
            # the embedding layer
            self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                           embedding_dim=args.n_embed)
            self.unk_index = args.unk_index
        else:
            self.word_embed = None
        if args.feat == 'char':
            self.feat_embed = CharLSTM(n_chars=args.n_feats,
                                       n_embed=args.n_char_embed,
                                       n_out=args.n_feat_embed,
                                       pad_index=args.feat_pad_index)
            self.pad_index = args.pad_index
        elif args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed,
                                            requires_grad=args.bert_fine_tune,
                                            mask_token_id=mask_token_id,
                                            token_dropout=args.token_dropout,
                                            mix_dropout=args.mix_dropout,
                                            use_hidden_states=args.use_hidden_states,
                                            use_attentions=args.use_attentions,
                                            attention_layer=args.attention_layer)
            #self.args.n_mlp_arc = self.feat_embed.bert.config.max_position_embeddings
            self.args.n_feat_embed = self.feat_embed.n_out # taken from the model
            self.args.n_bert_layers = self.feat_embed.n_layers # taken from the model
            self.pad_index = self.feat_embed.pad_index     # taken from the model
            self.args.pad_index = self.pad_index           # update
        else:
            self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
                                           embedding_dim=args.n_feat_embed)
            self.pad_index = args.pad_index
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        if args.n_lstm_layers:
            # the lstm layer
            self.lstm = BiLSTM(input_size=args.n_embed+args.n_feat_embed,
                               hidden_size=args.n_lstm_hidden,
                               num_layers=args.n_lstm_layers,
                               dropout=args.lstm_dropout)
            self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
            mlp_input_size = args.n_lstm_hidden*2
        else:
            self.lstm = None
            mlp_input_size = args.n_embed + args.n_feat_embed

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)

        # transformer attention
        if args.use_attentions:
            self.attn_mix = nn.Parameter(torch.randn(1)) #2)) # 1))

        # # distance
        # self.args.distance = False # DEBUG
        # if self.args.distance:
        #     self.distance = DeepBiaffine(mlp_input_size, mlp_input_size, self.args.deep_biaff_hidden_dim, 1, dropout=args.mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()


    def extra_repr(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total parameters: {total_params}\n" \
            f"Trainable parameters: {trainable_params}"


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self


    def forward(self, words: torch.Tensor,
                feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (~torch.LongTensor):
                Feat indices.
                If feat is ``'char'`` or ``'bert'``, the size of feats should be ``[batch_size, seq_len, fix_len]``.
                if ``'tag'``, the size is ``[batch_size, seq_len]``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """

        # words, feats are the first two items in the batch from TextDataLoader.__iter__()
        if words is None:
            words = feats[:,:,0] # drop subpiece dimension
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # feat_embed: [batch_size, seq_len, n_feat_embed]
        # attn: [batch_size, seq_len, seq_len]
        feat_embed, attn = self.feat_embed(feats)
        if self.word_embed:
            ext_words = words
            # set the indices larger than num_embeddings to unk_index
            if hasattr(self, 'pretrained'):
                ext_mask = words.ge(self.word_embed.num_embeddings)
                ext_words = words.masked_fill(ext_mask, self.unk_index)

            # get outputs from embedding layers
            word_embed = self.word_embed(ext_words)
            if hasattr(self, 'pretrained'):
                word_embed += self.pretrained(words)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), dim=-1)
        else:
            embed = self.embed_dropout(feat_embed)[0]

        if self.lstm:
            # print('PAD:', self.pad_index, words, feats, embed, lens) # DEBUG
            x = pack_padded_sequence(embed, lens, True, False)
            x, _ = self.lstm(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.lstm_dropout(x)
        else:
            x = embed

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # mix bert attentions
        if attn is not None:
            s_arc += self.attn_mix * attn
            # s_rel += self.attn_mix[1] * attn.unsqueeze(-1)

        # # head-dependent distance
        # if self.args.distance:
        #     # @see https://arxiv.org/pdf/1901.10457.pdf
        #     arange = torch.arange(words.size(1), device=words.device)
        #     head_offset = arange.view(1, 1, -1).expand(words.size(0), -1, -1) \
        #                   - arange.view(1, -1, 1).expand(words.size(0), -1, -1)
        #     dist_scores = self.distance(x, x).squeeze(3)
        #     dist_pred = 1 + F.softplus(dist_scores)
        #     dist_target = torch.abs(head_offset)
        #     dist_cauchy = -torch.log(1 + (dist_target.float() - dist_pred)**2/2)
        #     s_arc += dist_cauchy.detach()
        # else:
        #     dist_cauchy = None

        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        # Lower the diagonal, because the head of a word can't be itself.
        s_arc += torch.diag(s_arc.new(seq_len).fill_(float('-inf')))

        return s_arc, s_rel #, dist_cauchy


    def loss(self, s_arc: torch.Tensor, s_rel: torch.Tensor,
             arcs: torch.Tensor, rels: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
             # dist_cauchy: torch.Tensor = None
        r"""
        Computes the arc and tag loss for a sequence given gold heads and tags.
        Args:
        s_arc : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tags_count),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        s_rel : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tags_count),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for each word.
        rels : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for each word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        # dist_cauchy: ``torch.Tensor``, optional.
        #     log of distance probability distribution of head offsets.
        Returns
        -------
        loss : ``torch.Tensor``.
            The sum of the cross-entropy losses from the arcs and rels predictions.
        """

        # heads = arcs
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        # select the predicted relations towards the correct heads
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        loss = arc_loss + rel_loss

        # if dist_cauchy is not None:
        #     #dist_cauchy = torch.gather(dist_cauchy[:, 1:], 2, heads.unsqueeze(2))
        #     dist_cauchy = torch.gather(dist_cauchy, 2, heads.unsqueeze(2))
        #     loss -= dist_cauchy.sum()

        return loss


    def decode(self, s_arc: torch.Tensor, s_rel: torch.Tensor,
               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        # select the most likely arcs
        arc_preds = s_arc.argmax(-1)
        if self.args.tree:
            # ensure the arcs form a tree
            bad = [not istree(seq[:i+1], self.args.proj)
                   for i, seq in zip(lens.tolist(), arc_preds.tolist())]
            if any(bad):
                arc_preds[bad] = eisner(s_arc[bad], mask[bad])
        # select the most likely rels
        rel_preds = s_rel.argmax(-1)
        # choose those corresponding to the predicted arcs
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
