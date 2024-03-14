import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from functools import reduce
from model.RecurrentVisionTransformer import FullSelfAttention
from model.base import MLP, Attention

from torch.nn import init
from typing import Optional, Tuple

class Tokenizer(nn.Module):
    """
    Convolution tokenizer as described in https://arxiv.org/pdf/2104.05704.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, args, stride, **kwargs):
        super().__init__()
        args = Namespace(**vars(args), **kwargs)
        self.args = args
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//stride)

        self.apply(self.init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(F.relu(self.conv(x)), kernel_size=2)
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
    
class SeqPool(nn.Module):
    """
    Sequence Pooling as described in https://arxiv.org/pdf/2104.05704.pdf
    """

    def __init__(self, embd_size, args, **kwargs):
        super().__init__()
        args = Namespace(**vars(args), **kwargs)
        self.args = args
        self.linear = nn.Linear(embd_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = self.linear(x)
        xl = xl.permute(0, 2, 1)
        xl = F.softmax(xl, dim=-1)
        xl = (xl @ x).squeeze(1)
        return xl

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def init(self, drop_prob , scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder as described in https://arxiv.org/pdf/2104.05704.pdf
    """

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1, attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.self_attn = Attention(dim=embed_dim, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        # TODO: this does not work for some reason
        drop_path_rate = 0
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.pre_norm(x)))
        x = self.norm1(x)
        y = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = x + self.drop_path(self.dropout2(y))
        return x

class MemEHead(nn.Module):
    """
    FC Layer detection head.
    """
    def __init__(self, embd_size):
        super().__init__()
        self.linear = nn.Linear(embd_size, 2)

    def forward(self, x):
        return self.linear(x)

class MemEViT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**vars(args), **kwargs)
        self.args = args

        self.token1 = Tokenizer(args.in_channels, args.embd_size // 2, args.kernel_size, args, stride=2)
        self.token2 = Tokenizer(args.embd_size // 2, args.embd_size, args.kernel_size, args, stride=1)

        self.positional_emb = nn.Parameter(self.sinusoidal_embedding(args.num_tokens + 2 * args.num_mem_tokens, args.embd_size), requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)

        self.stage1 = TransformerEncoder(args.embd_size, args.heads, args.dim_feedforward, args.dropout, args.att_dropout, args.drop_path_rate)
        self.norm = nn.LayerNorm(args.embd_size)
        self.seqpool = SeqPool(args.embd_size, args)

        self.detection = MemEHead(args.embd_size)

        self.init_mem_tokens()

        self.apply(self.init_weight)


    def forward(self, x, h=None):
        B, N, C, H, W = x.size()

        outputs = []

        mem = self.mem_tokens # TODO: not sure this is correct
        mem = mem.repeat(B, 1, 1)

        for t in range(N):

            xt = x[:, t, :, :, :]
            xt = self.token1(xt)
            xt = self.token2(xt)

            xt = xt.view(B, -1, self.args.embd_size)

            # B x N x D -> B x (M + N + M) x D
            xt = torch.cat([mem, xt, mem], dim=1)


            xt = xt + self.positional_emb


            xt = self.dropout(xt)

            xt = self.stage1(xt)
            xt = self.norm(xt)

            # strip out M tokens
            mem = xt[:, -self.args.num_mem_tokens:, :]
            xt = xt[:, self.args.num_mem_tokens:self.args.num_mem_tokens + self.args.num_tokens, :]

            xt = self.seqpool(xt)
            final_output = self.detection(xt)
            outputs.append(final_output)
            

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, None
    
    def init_mem_tokens(self):
        if self.args.num_mem_tokens == 0:
            self.mem_tokens = None
        else:
            self.mem_tokens = [torch.randn(1, self.args.embd_size)] * self.args.num_mem_tokens
            self.mem_tokens = torch.cat(self.mem_tokens, dim=0).view(1, self.args.num_mem_tokens, -1)
            self.mem_tokens = torch.nn.Parameter(self.mem_tokens, requires_grad=True)
            self.register_parameter(param=self.mem_tokens, name='mem_tokens')
    
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)