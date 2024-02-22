import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from argparse import Namespace
import math
from typing import Type


class MaxGRU(nn.Module):
    """
    Adaptation of the MaxViT paper and RVT paper together with fast attention module, optimized for performance.
    """

    def __init__(self, stages):
        super().__init__()
        self.stages = stages
        pass

    def forward(self, x):
        pass

class MaxGRUBlock(nn.Module):
    """
    Block of the MaxGRU model. Idea from MaxViT and RVT paper.
    Conv -> FastAttention -> MLP -> GRU
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)

        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.attn = FastAttention(args.output_channels, mid_chn=args.mid_chn, out_chan=args.out_chan, norm_layer=args.norm_layer)
        self.mlp = MLP(args.output_channels, channel_last=False, expansion_ratio=args.expansion_ratio, act_layer=args.act_layer, gated=args.gated, bias=args.bias, drop_prob=args.drop_prob)
        self.gru = nn.GRU(input_size=36192, hidden_size=args.gru_hidden_size, num_layers=args.gru_layers, batch_first=True)
        self.detection = nn.Linear(args.gru_hidden_size, 2)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = self.conv(x)
        x = self.attn(x)
        x = self.mlp(x)
        x = x.view(B, N, -1)
        x, _ = self.gru(x)
        x = self.detection(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu', *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FastAttention(nn.Module):
    """
    """
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs):
        super(FastAttention, self).__init__()
        self.norm_layer = norm_layer
        mid_chn = int(in_chan/2)
        self.w_qs = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.init_weight()

    def forward(self, feat):
        # Expect shape N x C x H x W

        query = self.w_qs(feat)
        key   = self.w_ks(feat)
        value = self.w_vs(feat)

        N,C,H,W = feat.size()

        query_ = query.view(N,32,-1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_   = key.view(N,32,-1)
        key   = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N,C,-1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        return p_feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 act_layer: Type[nn.Module],
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            #inner_dim = round(inner_dim * 2 / 3)
            #inner_dim = math.ceil(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            #inner_dim = round(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                act_layer(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 act_layer: Type[nn.Module],
                 bias: bool = True):
        super().__init__()
        # Different activation functions / versions of the gated linear unit:
        # - ReGLU:  Relu
        # - SwiGLU: Swish/SiLU
        # - GeGLU:  GELU
        # - GLU:    Sigmoid
        # seem to be the most promising once.
        # Extensive quantitative eval in table 1: https://arxiv.org/abs/2102.11972
        # Section 2 for explanation and implementation details: https://arxiv.org/abs/2002.05202
        # NOTE: Pytorch has a native GLU implementation: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html?highlight=glu#torch.nn.GLU
        proj_out_dim = dim_out*2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = -1 if channel_last else 1

        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)


class MGRU(nn.Module):
    """
    Modified GRU architecture that works on feature maps.
    Inspired by the LSTM used in Recurrent Vision Transformer.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)

        self.dim = args.gates_dimension
        self.conv1x1 = nn.Conv2d(in_channels = args.in_channels * 2, out_channels = 2 * args.gates_dimension, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels = args.in_channels * 2, out_channels = args.gates_dimension, kernel_size=1)


    def forward(self, x, h):
        # input size: N x C x H x W
        # hidden size: N x C x H x W
        xh = torch.cat([h, x], dim=1)
        r_u = self.conv1x1(xh)
        r, u = torch.split(r_u, self.dim, dim=1)
        r = F.sigmoid(r)
        u = F.sigmoid(u)

        xh = torch.cat([x, r * h], dim=1)
        c = self.conv_c(xh)
        c = F.relu(c)

        return u * h + (1 - u) * c
