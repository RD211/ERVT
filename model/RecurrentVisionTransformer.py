from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer
from model.base import DWSConvLSTM2d, GLU, MLP, LinearHead, LayerScale

class SelfAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)


    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x

class FullSelfAttention(nn.Module):
    def __init__(self, channels: int, partition_size: int, dim_head: int = 32):
        super().__init__()

        self.channels = channels
        self.partition_size = partition_size
        self.ln = nn.LayerNorm(channels)
        self.ls = LayerScale(channels)
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()

        x = self.ln(x)  # Apply layer normalization
        attn_output = self.mhsa(x.view(B,H*W,C)).view(B,H,W,C)  # Apply self-attention
        attn_output = self.ls(attn_output)  # Scale the attention output

        return attn_output

class RVTBlock(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.fsa = FullSelfAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head)

        self.ln = nn.LayerNorm(args.output_channels)

        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)

        self.lstm = DWSConvLSTM2d(dim=args.output_channels)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv)

        x_bsa = self.fsa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        x_unflattened = x_bsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (c, h))

        return x_unflattened, c

class RVT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**vars(args), **kwargs)
        self.args = args

        act_layer_to_activation = {
            "gelu": nn.GELU,
            "relu": nn.ReLU
        }

        def replace_act_layer(args):
            args["mlp_act_layer"] = act_layer_to_activation[args["mlp_act_layer"]]
            return args

        self.stages = nn.ModuleList([
            RVTBlock(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.n_time_bins,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x, lstm_states=None):
        B, N, C, H, W = x.size()

        if lstm_states == None:
            lstm_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                lstm_state = lstm_states[i] if lstm_states[i] is not None else (None, None)
                xt, c = stage(xt, lstm_state[0], lstm_state[1])

                # We save it for the next time step
                lstm_states[i] = (c, xt)


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, lstm_states
