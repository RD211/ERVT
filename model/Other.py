from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer
from model.base import DWSConvLSTM2d, GLU, MLP, LinearHead, LayerScale
import random
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
    def __init__(self, channels: int, dim_head: int = 32):
        super().__init__()

        self.channels = channels
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()

        attn_output = self.mhsa(x.view(B,H*W,C)).view(B,H,W,C)  # Apply self-attention

        return attn_output

class RVTBlock(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv1 = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(args.output_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)

        self.fsa = FullSelfAttention(channels=args.output_channels, dim_head=args.dim_head)

        self.ln1= nn.LayerNorm(args.output_channels)
        self.ln11 = nn.LayerNorm(args.output_channels)
        self.ln2= nn.LayerNorm(args.output_channels)

        self.mlpb = MLP2(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)

        self.lstm = DWSConvLSTM2d(dim=args.output_channels)
    # input (80x60x9)
    # conv1 (40x30x32)
    # conv2 (20x15x32)
    # fsa (20x15x32)
    # mlpb (20x15x32)
    # lstm (20x15x32)
        
    # conv1 (10x7x64)
    # conv2 (5x4x64)
    # fsa (5x4x64)
    # mlpb (5x4x64)
    # lstm (5x4x64)
        
    # Detection head with multiple layers
    
    # add more layers in mlp
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 2 - convs
        x_conv = self.conv1(x)
        x_conv = self.relu(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln1(x_conv)

        x_conv = x_conv.permute(0, 3, 1, 2)
        x_conv = self.conv2(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln11(x_conv)

        x_bsa = self.fsa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln2(x_bsa)
        x_bsa = self.mlpb(x_bsa)
        
        x_unflattened = x_bsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (h, c))

        return x_unflattened, c

class MLP2(nn.Module):
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

        bottleneck = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=inner_dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
            act_layer(),
        )
        self.net = nn.Sequential(
            proj_in,
            bottleneck,
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)



class LinearHead2(nn.Module):
    """
    FC Layer detection head.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**vars(args), **kwargs)
        self.args = args

        stride_factor = reduce(lambda x, y: x * y, [s["stride"] for s in args.stages])
        dim = int(args.stages[-1]["output_channels"] * ((args.sensor_width * args.spatial_factor) // stride_factor // stride_factor) * ((args.sensor_height * args.spatial_factor) // stride_factor // stride_factor))

        self.linear1 = nn.Linear(640, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    
class RVT2(nn.Module):
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
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.in_channels,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead2(args)

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
                lstm_states[i] = (xt, c)


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            if len(outputs) > 0:
                final_output = final_output + outputs[-1]
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, lstm_states
