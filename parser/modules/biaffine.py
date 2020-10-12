# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


# class PairwiseBilinear(nn.Module):
#     r""" A bilinear module that deals with broadcasting for efficient memory usage.
#     Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
#     Output: tensor of size (N x L1 x L2 x O)"""

#     def __init__(self, input1_size, input2_size, output_size, bias=True):
#         super().__init__()

#         self.input1_size = input1_size
#         self.input2_size = input2_size
#         self.output_size = output_size

#         self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
#         self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0

#     def forward(self, input1, input2):
#         input1_size = list(input1.size())
#         input2_size = list(input2.size())
#         output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

#         # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
#         intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
#         # (N x L2 x D2) -> (N x D2 x L2)
#         input2 = input2.transpose(1, 2)
#         # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
#         output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
#         # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
#         output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

#         return output


# class PairwiseBiaffineScorer(nn.Module):
#     def __init__(self, input1_size, input2_size, output_size):
#         super().__init__()
#         self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

#         self.W_bilin.weight.data.zero_()
#         self.W_bilin.bias.data.zero_()

#     def forward(self, input1, input2):
#         input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1)
#         input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)
#         return self.W_bilin(input1, input2)


# class DeepBiaffine(nn.Module):
#     def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0):
#         super().__init__()
#         self.W1 = nn.Linear(input1_size, hidden_size)
#         self.W2 = nn.Linear(input2_size, hidden_size)
#         self.hidden_func = hidden_func
#         self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input1, input2):
#         return self.scorer(self.dropout(self.hidden_func(self.W1(input1))), self.dropout(self.hidden_func(self.W2(input2))))
